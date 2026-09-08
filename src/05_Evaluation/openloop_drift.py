"""
openloop_drift.py
=================
Open-loop drift of a trained checkpoint over a full leach cycle.

The network is seeded with the true Se at hour 0 and then stepped forward on
its own predictions, with no plant and no controller to re-anchor it, under the
run's own recorded irrigation schedule. Errors compound. This is not one-step
teacher-forced error and not validation loss, both of which stay low while
drift separates candidates.

Closed-loop control re-anchors to the true plant state every day, which hides
drift, so open loop is the discriminating measure for ranking checkpoints.
Closed-loop tracking is the right measure for the chosen model, not for
choosing it.

WHAT IT COMPUTES
  drift(t)   mean over runs of RMSE across the 221 nodes between predicted and
             true Se at hour t
  headline   mean drift over the horizon, lower is better
  one-step   teacher-forced one-step RMSE, for contrast
  per-run    the full drift matrix, plus median, IQR, 90th and 99th
             percentiles, worst run and its id, and soil per run

Truth is read from the dataset h5, so no simulator call is needed. Scaling and
dimensions come from preprocess.py, so the input layout is identical to
training: [ Se[:,t] (221) | R[t+1] (1, scaled) | soil (4) ], soil order
[Ks, alpha, n_vgm, theta_s]. Predictions are clamped to [0, 1], the inference
convention, which means this harness cannot see an out-of-range prediction.

FLAGS
  --ckpt        trained checkpoint. Required.
  --tag         label for this run, used in the output filename. Required.
  --split       val | test. Default val.
  --horizon     hours to step. Default 599, the full run.
  --out         output folder. Default: beside the checkpoint.

Writes sweep_drift_<tag>.json and an ASCII table to the terminal.

Usage, from 05_Evaluation:
  python -u openloop_drift.py --ckpt ../04_Training/models/pgml_lam0p4_s42.pt --tag s42_lam0p4

--h5 and --prep default through paths.py to data/dataset.h5 and reference/.

Status: stable
Depends on: preprocess.py, paths.py, torch, numpy, h5py

Changelog
---------
2026.09.07  v2.1  Repository release. Renamed from rollout_drift.py. --h5 and
                  --prep default through paths.py. Header rewritten.
2026.08.19  v2.0  Loader rewritten for the rebuilt checkpoint format and the
                  sigmoid output layer, refusing rather than falling back
                  silently. The per-step clamp would otherwise hide an
                  unbounded head and a wrong network would produce a plausible
                  drift curve. --split now defaults to val.
2026.08.01  v1.2  Per-run drift retained, so the spread across runs is visible
                  and not only its mean. Added after a seed-to-seed CV of 63 to
                  99 percent on the headline, large enough that the ranking
                  cannot be read from a mean alone. Reports the median, IQR,
                  90th and 99th percentiles, the worst run and its id, and the
                  count above a threshold. Per-run values are keyed by run id
                  and soil is written per run, so a tail can be traced to a
                  corner of the parameter box. The curve and the headline are
                  unchanged to the digit.
2026.07.30  v1.1  R indexing corrected to match preprocess.py v2.6. Both the
                  rollout and the teacher-forced path now take the rate over
                  the step, R_scaled[:, t+1]. H = min(horizon, T-1), so t+1
                  never exceeds T-1. Drift figures produced before this version
                  are superseded.
2026.07.29  v1.0  Initial open-loop rollout-drift harness. Vectorised over all
                  test runs (one GPU state matrix stepped in parallel). Open-loop
                  drift in a sequential horizon loop; teacher-forced one-step
                  RMSE vectorised and chunked (VRAM-safe). Writes
                  sweep_drift_<tag>.json for plot_batch_sweep.py. ASCII-only
                  output for the cp1252 terminal.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import h5py
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
import paths
paths.add_stage_paths()

from preprocess import (INPUT_DIM, OUTPUT_DIM, load_scalers, apply_scaler,
                        SOIL_PARAM_KEYS)


# =============================================================================
# MODEL LOADER  (import the trainer's MLP; rebuild locally if the import path
# is not available, mirroring pgml_closed_loop.py so this file is robust to
# where it is launched from)
# =============================================================================
def load_model(ckpt_path, device, width=512, depth=4):
    """Loads either checkpoint format, and refuses to guess.

    Rebuilt checkpoints differ from the withdrawn ones in three ways, and the
    previous loader absorbed all three in silence:

      FORMAT. Withdrawn checkpoints carry "args" and "model_state". Rebuilt
      ones carry "model", with lam, seed and output_activation at the top level
      and no "args" at all.

      OUTPUT LAYER. Rebuilt models end in a sigmoid. It has no parameters, so
      building the network without it would load the weights CLEANLY and then
      predict outside [0, 1]. In this file the clamp at each rollout step would
      hide that, and a drift curve computed from clamped predictions is a
      number rather than a measurement. Hence the refusal below.

      The network class is rebuilt here from the checkpoint's own recorded
      width, depth and output activation, so no import from the trainer is
      needed.
    """
    import torch.nn as nn
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "model_state" in ck and "args" in ck:
        fmt = "withdrawn"
        state = ck["model_state"]
        w, d = int(ck["args"]["width"]), int(ck["args"]["depth"])
        act = "none"
    elif "model" in ck:
        fmt = "rebuild"
        state = ck["model"]
        w, d = int(ck.get("width", width)), int(ck.get("depth", depth))
        act = ck.get("output_activation")
        if act is None:
            raise SystemExit(
                f"{ckpt_path}: rebuilt checkpoint with no output_activation. "
                "Refusing to guess: an unbounded head where a sigmoid belongs "
                "predicts outside [0, 1], and the rollout clamp would hide it.")
    else:
        raise SystemExit(f"{ckpt_path}: unrecognised checkpoint format, "
                         f"keys {sorted(ck.keys())}")

    L = [nn.Linear(INPUT_DIM, w), nn.BatchNorm1d(w), nn.ReLU()]
    for _ in range(d - 1):
        L += [nn.Linear(w, w), nn.BatchNorm1d(w), nn.ReLU()]
    L.append(nn.Linear(w, OUTPUT_DIM))
    if act == "sigmoid":
        L.append(nn.Sigmoid())
    elif act != "none":
        raise SystemExit(f"{ckpt_path}: unknown output_activation {act!r}")

    class MLP(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    model = MLP(L)
    model.load_state_dict(state, strict=True)
    ck["_fmt"], ck["_act"] = fmt, act
    print(f"checkpoint : {os.path.basename(os.path.dirname(ckpt_path))}/"
          f"{os.path.basename(ckpt_path)}   format={fmt}   output={act}")
    return model.to(device).eval(), ck


# =============================================================================
# DATA LOADER  (test runs only, straight from the h5; truth is already stored)
# Returns GPU tensors:
#   Se_true : (n_runs, 221, T)   effective saturation, ground truth
#   R_scaled: (n_runs, T)        scaled irrigation rate (network input units)
#   soil    : (n_runs, 4)        scaled [Ks, alpha, n_vgm, theta_s]
# =============================================================================
def load_test_tensors(h5_path, prep_dir, scalers, device, split="val"):
    # Default is val, not test. The 615 test soils are the held-out set and
    # were used once, for the selected configuration only.
    split_path = os.path.join(prep_dir, "split.json")
    with open(split_path, "r") as fh:
        test_ids = json.load(fh)[split]

    Se_list, R_list, soil_list, kept, dropped = [], [], [], [], []
    with h5py.File(h5_path, "r") as f:
        for r in test_ids:
            if r not in f:
                dropped.append(r)
                continue
            grp = f[r]
            Se = grp["Se_out"][:].astype(np.float64)   # (221, T)
            R  = grp["R_out"][:].astype(np.float64)     # (T,)
            if np.any(np.isnan(Se)) or np.any(np.isnan(R)):
                dropped.append(r)
                continue
            soil = np.array([float(grp.attrs[k]) for k in SOIL_PARAM_KEYS],
                            dtype=np.float64)            # [Ks, alpha, n_vgm, theta_s]
            Se_list.append(Se)
            R_list.append(R)
            soil_list.append(soil)
            kept.append(r)

    if dropped:
        print(f"  WARNING dropped {len(dropped)} test run(s) (missing/NaN): "
              f"{dropped[:5]}{' ...' if len(dropped) > 5 else ''}")

    Se_true = np.stack(Se_list, axis=0).astype(np.float32)          # (n,221,T)
    R_raw   = np.stack(R_list,  axis=0).astype(np.float64)          # (n,T)
    soil_raw = np.stack(soil_list, axis=0).astype(np.float64)       # (n,4)

    # scale R and each soil column with the SAME scaler used in training
    R_scaled = apply_scaler(R_raw, "R_out", scalers)                # (n,T) f32
    soil_scaled = np.zeros_like(soil_raw, dtype=np.float32)
    for j, k in enumerate(SOIL_PARAM_KEYS):
        soil_scaled[:, j] = apply_scaler(soil_raw[:, j], k, scalers)

    Se_true = torch.from_numpy(Se_true).to(device)
    R_scaled = torch.from_numpy(R_scaled.astype(np.float32)).to(device)
    soil = torch.from_numpy(soil_scaled).to(device)
    # v1.2: soil_raw returned as well, in physical units, so a run in the tail
    # can be tied to its parameters without reopening the h5.
    return Se_true, R_scaled, soil, kept, soil_raw


# =============================================================================
# OPEN-LOOP ROLLOUT  (sequential in time; all runs stepped in parallel)
# Seed with truth at t=0, then feed predictions forward. drift[t] is the mean
# over runs of the node-RMSE between predicted and true Se at hour t.
# =============================================================================
def open_loop_drift(model, Se_true, R_scaled, soil, horizon, device):
    n, Nz, T = Se_true.shape
    H = min(horizon, T - 1)
    drift = torch.zeros(H + 1, device=device)          # drift[0] = 0 by construction
    # v1.2: keep the per-run error as well as its mean. err is already computed
    # per run inside the loop and was being collapsed immediately; this only
    # stores what was previously discarded, so the curve is bit-identical.
    per_run = torch.zeros(n, H + 1, device=device)
    state = Se_true[:, :, 0].clone()                    # (n,221)
    with torch.no_grad():
        for t in range(H):
            # R_scaled[:, t+1] is the rate applied over the step t -> t+1
            # (see preprocess.py v2.6). Index is safe: H = min(horizon, T-1),
            # so t+1 <= T-1.
            x = torch.cat([state, R_scaled[:, t + 1:t + 2], soil], dim=1)   # (n,226)
            state = model(x).clamp(0.0, 1.0)                            # (n,221) Se(t+1)
            err = torch.sqrt(torch.mean((state - Se_true[:, :, t + 1]) ** 2, dim=1))
            per_run[:, t + 1] = err
            drift[t + 1] = err.mean()
    # per-run headline: mean over the horizon for each run, same reduction the
    # headline scalar applies, but taken before the average over runs
    per_run_headline = per_run[:, 1:H + 1].mean(dim=1)
    return drift.cpu().numpy(), per_run_headline.cpu().numpy(), H


# =============================================================================
# TEACHER-FORCED ONE-STEP RMSE  (the "val loss" view, for contrast)
# Feed TRUE Se at every step; measure one-step error. Vectorised and chunked so
# the (n*H, 226) forward stays VRAM-safe.
# =============================================================================
def teacher_forced_onestep(model, Se_true, R_scaled, soil, H, device, chunk=65536):
    n, Nz, T = Se_true.shape
    Se_x = Se_true[:, :, 0:H].permute(0, 2, 1).reshape(-1, Nz)          # (n*H,221)
    Se_y = Se_true[:, :, 1:H + 1].permute(0, 2, 1).reshape(-1, Nz)      # (n*H,221)
    R_x  = R_scaled[:, 1:H + 1].reshape(-1, 1)                          # (n*H,1) rate over t -> t+1
    soil_x = soil.unsqueeze(1).expand(-1, H, -1).reshape(-1, soil.shape[1])  # (n*H,4)
    X = torch.cat([Se_x, R_x, soil_x], dim=1)                          # (n*H,226)

    total, m = 0.0, X.shape[0]
    with torch.no_grad():
        for i in range(0, m, chunk):
            pred = model(X[i:i + chunk]).clamp(0.0, 1.0)
            err = torch.sqrt(torch.mean((pred - Se_y[i:i + chunk]) ** 2, dim=1))
            total += err.sum().item()
    return total / m


def main():
    p = argparse.ArgumentParser(
        description="Open-loop drift of a trained checkpoint")
    p.add_argument("--ckpt", required=True, help="Trained checkpoint (.pt)")
    p.add_argument("--h5",   default=None,
                   help="Dataset h5. Default: data/dataset.h5")
    p.add_argument("--prep", default=None,
                   help="Folder with scaler.json and split.json. Default: reference/")
    p.add_argument("--tag",  required=True, help="Label for this run (e.g. b1024)")
    p.add_argument("--out",  default=None, help="Dir for sweep_drift_<tag>.json (default: checkpoint dir)")
    p.add_argument("--split", default="val", choices=["val", "test"],
                   help="Held-out split to step through. Default val.")
    p.add_argument("--horizon", type=int, default=599,
                   help="Hours to step. Default 599, the full run.")
    args = p.parse_args()

    if args.h5 is None:
        args.h5 = paths.require_h5()
    if args.prep is None:
        args.prep = paths.REFERENCE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out if args.out else os.path.dirname(os.path.abspath(args.ckpt))

    print("=" * 68)
    print("OPEN-LOOP ROLLOUT DRIFT")
    print("=" * 68)
    print(f"device : {device}")
    if device.type == "cuda":
        print(f"gpu    : {torch.cuda.get_device_name(0)}")
    print(f"ckpt   : {args.ckpt}")
    print(f"tag    : {args.tag}")

    if not os.path.isfile(args.ckpt):
        print(f"ERROR: checkpoint not found: {args.ckpt}. Aborting.")
        return

    model, ck = load_model(args.ckpt, device)
    scalers = load_scalers(args.prep)
    _a    = ck.get("args", {})
    batch = ck.get("batch_size", _a.get("batch_size"))
    lr    = ck.get("lr",         _a.get("lr"))
    lam   = ck.get("lam",        _a.get("lam"))
    print(f"batch  : {batch}   lr : {lr}   lam : {lam}")

    _t0 = time.perf_counter()
    Se_true, R_scaled, soil, kept, soil_raw = load_test_tensors(
        args.h5, args.prep, scalers, device, split=args.split)
    n, Nz, T = Se_true.shape
    print(f"{args.split:<7}: {n} runs, Se {Nz} nodes x {T} hours "
          f"({Se_true.element_size() * Se_true.nelement() / 1e6:.0f} MB on {device})")

    drift, per_run, H = open_loop_drift(model, Se_true, R_scaled, soil,
                                        args.horizon, device)
    tf_onestep = teacher_forced_onestep(model, Se_true, R_scaled, soil, H, device)
    eval_seconds = time.perf_counter() - _t0

    headline = float(np.mean(drift[1:H + 1]))     # mean drift over the horizon
    final = float(drift[H])

    # -- v1.2 distribution across test runs -----------------------------------
    # headline is the mean of per_run by construction; asserted below rather
    # than assumed, because if it ever stops holding the two numbers are
    # measuring different things and the comparison to the median is void.
    pr = np.asarray(per_run, dtype=np.float64)
    if not np.isclose(float(pr.mean()), headline, rtol=1e-5, atol=0.0):
        print(f"  WARNING per-run mean {pr.mean():.8f} does not match headline "
              f"{headline:.8f}. The median comparison below is not meaningful.")
    q = np.percentile(pr, [25, 50, 75, 90, 99])
    worst_i = int(np.argmax(pr))
    dist = {
        "mean": float(pr.mean()),
        "median": float(q[1]),
        "sd": float(pr.std(ddof=1)),
        "p25": float(q[0]), "p75": float(q[2]),
        "p90": float(q[3]), "p99": float(q[4]),
        "iqr": float(q[2] - q[0]),
        "min": float(pr.min()), "max": float(pr.max()),
        "worst_run": kept[worst_i],
        "worst_soil": {k: float(soil_raw[worst_i, j])
                       for j, k in enumerate(SOIL_PARAM_KEYS)},
        # a run 10x the median is doing something categorically different from
        # drifting; counted so a tail cannot hide inside an average
        "n_above_10x_median": int((pr > 10.0 * q[1]).sum()),
        "n_above_0p1": int((pr > 0.1).sum()),
        "mean_over_median": float(pr.mean() / q[1]) if q[1] > 0 else float("nan"),
    }

    marks = [h for h in (1, 24, 72, 168, 336, H) if h <= H]
    checkpoints = {str(h): float(drift[h]) for h in marks}

    print("\n" + "-" * 68)
    print("open-loop drift (node-RMSE, mean over runs), by horizon hour")
    print("-" * 68)
    print(f"{'hour':>8}{'drift':>14}")
    for h in marks:
        print(f"{h:>8}{drift[h]:>14.6f}")
    print("-" * 68)
    print(f"{'HEADLINE mean drift over horizon':<44}{headline:>14.6f}")
    print(f"{'final drift (hour ' + str(H) + ')':<44}{final:>14.6f}")
    print(f"{'teacher-forced one-step RMSE (val-loss view)':<44}{tf_onestep:>14.6f}")
    print(f"{'eval seconds':<44}{eval_seconds:>14.1f}")
    print("-" * 68)

    print("\n" + "-" * 68)
    print(f"distribution of per-run drift across {n} test runs")
    print("-" * 68)
    print(f"{'mean (the headline)':<44}{dist['mean']:>14.6f}")
    print(f"{'median':<44}{dist['median']:>14.6f}")
    print(f"{'sd across runs':<44}{dist['sd']:>14.6f}")
    print(f"{'p25 / p75 (IQR ' + format(dist['iqr'], '.4f') + ')':<44}"
          f"{dist['p25']:>7.4f}{dist['p75']:>7.4f}")
    print(f"{'p90':<44}{dist['p90']:>14.6f}")
    print(f"{'p99':<44}{dist['p99']:>14.6f}")
    print(f"{'worst run  ' + str(dist['worst_run']):<44}{dist['max']:>14.6f}")
    print(f"{'runs above 10x median':<44}{dist['n_above_10x_median']:>14d}")
    print(f"{'runs above 0.1':<44}{dist['n_above_0p1']:>14d}")
    print(f"{'mean / median':<44}{dist['mean_over_median']:>14.2f}")
    print("-" * 68)
    # The one line that decides how the sweep is read. A ratio near 1 means the
    # headline represents a typical run; a large ratio means it represents a
    # tail, and the ranking must then be taken on the median.
    if dist["mean_over_median"] > 2.0:
        print("READ: mean is set by a tail. Rank on the median, not the mean.")
    else:
        print("READ: mean is representative of a typical run.")
    print("-" * 68)

    summary = {
        "tag": args.tag,
        "ckpt": os.path.abspath(args.ckpt),
        "batch": batch, "lr": lr, "lam": lam,
        "n_test": n, "horizon": H,
        "drift_headline_mean": headline,
        "drift_final": final,
        "onestep_teacher_forced": tf_onestep,
        "drift_checkpoints": checkpoints,
        "drift_curve": [float(v) for v in drift],
        # v1.2
        "drift_distribution": dist,
        "drift_per_run": {kept[i]: float(pr[i]) for i in range(len(kept))},
        "eval_seconds": eval_seconds,
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"sweep_drift_{args.tag}.json")
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote  : {out_path}")


if __name__ == "__main__":
    main()
