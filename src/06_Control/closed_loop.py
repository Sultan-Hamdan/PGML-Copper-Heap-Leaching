r"""
closed_loop.py
==============
Runs the PGML closed loop across many soils at once.

The network is the controller's internal model. Each day it sweeps candidate
irrigation rates, steps itself 24 hours ahead per candidate, scores the
predicted profile against the setpoint, and applies the argmin. The numerical
solver then advances one real day and the network is re-anchored to the plant
state.

OBJECTIVE
  Squared deviation of Se from the setpoint, averaged over the 221 depth nodes.
  Unnormalised, unweighted, no irrigation penalty. Olivares 2025 Eq. 32.

REPORTED
  primary    that quantity summed over every one of the tf*24 hours, so
             intra-day excursions are not hidden
  secondary  summed over the tf end-of-day profiles only

ACTION SET, per soil
  r_min = scaler R_out minimum, 0.1623
  r_max = min(0.30 * Ks, safety * R_stable, scaler R_out max)
  R_stable is measured per soil by rmax_ceiling.py. The ceiling file's id list
  is checked against the soil list element by element before r_max is consumed
  positionally.

SOIL SETS
  --split val    614 validation soils
  --split test   615 test soils, held out
  --soil_table   a constructed table from build_extrapolation_soils.py
  The list is frozen to a JSON on first run and reused, so a re-run cannot
  redraw a different set. Each run starts from its own Se_i as a uniform
  profile.

Usage, from 06_Control:
    python -u closed_loop.py --limit_ckpts 1 --tf 3 --n_soils 64
    python -u closed_loop.py
    python -u closed_loop.py --split test --ckpt_glob "pgml_lam0p4_s*.pt"

Outputs, to --out_dir:
    soils_<split>.json     frozen soil list and per-soil parameters
    cl_<tag>.json          per-checkpoint result, per-soil arrays
    summary_<split>.csv    one row per checkpoint

--root, --h5, --ceiling and --sweep_dir default through paths.py.

Status: stable
Depends on: column_solver.py, preprocess.py, paths.py, torch, numpy, h5py

Changelog
---------
2026.09.07  v1.5  Repository release. Renamed from closed_loop_pr8.py. R_flood
                  renamed to R_stable; ceiling file renamed to rmax_<split>.json;
                  imports and paths routed through paths.py. Checkpoint tag now
                  falls back to the file stem when the parent folder is models/.
2026.08.24  v1.4  --soil_table for the extrapolation cells, reading constructed
                  soils instead of the dataset. Ceiling and output are named per
                  cell, so a constructed run cannot fall back on the validation
                  ceiling and 24 soil sets cannot overwrite one result.
2026.08.12  v1.3  Checkpoint loader rewritten for the rebuilt format: different
                  key layout, sigmoid output layer, one folder per run all named
                  best.pt. Each would have been absorbed silently before.
2026.08.06  v1.2  --sweep_dir added; the checkpoint directory had been hardwired.
2026.08.01  v1.1  Per-soil stability ceiling, NaN abort, convergence reporting.
                  r_max is min(0.95*R_stable, 0.30*Ks, scaler R_out max). The
                  stability limit binds on 4 of 64 validation soils, all at the
                  bottom of the n envelope with dry starts.
2026.08.01  v1.0  Initial. Soil-batched closed loop over the lambda-sweep
                  checkpoints. Hourly and daily
                  Hourly and daily accumulations computed in the same pass, primary
                  declared in advance. Soil list frozen to JSON. ASCII only.
"""

import argparse
import csv
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths

DEF_ROOT = paths.REPO_ROOT
DEF_H5 = paths.H5

SOIL_KEYS = ["Ks", "alpha", "n_vgm", "theta_s"]


# ---------------------------------------------------------------------------
# soil set
# ---------------------------------------------------------------------------
def load_soil_table(path):
    """Read a constructed soil table. Returns (ids, soil dict, area, cell).

    Extrapolation soils are constructed rather than drawn from the design, so
    they have no h5 entry and their parameters travel with them. This mirrors
    rmax_ceiling.py v2.3, which reads the same file, so the ceiling
    and the run see one source.

    ORDER IS A CONTRACT. "ids" and "rows" are written in the same order by
    build_extrapolation_soils.py and preserved here, because the ceiling guard
    below compares id lists element by element and then consumes r_max
    positionally. That chain is what keeps each soil on its own stability limit.
    """
    with open(path, "r") as fh:
        tab = json.load(fh)
    rows = tab["rows"]
    ids = [r["id"] for r in rows]
    if ids != list(tab["ids"]):
        raise RuntimeError(
            "the soil table's ids do not match its rows in order. Rebuild it.")
    soil = {k: np.asarray([float(r[k]) for r in rows], dtype=np.float64)
            for k in SOIL_KEYS + ["Se_i", "theta_r"]}
    return ids, soil, float(tab["area"]), tab["cell"]


def build_soil_set(h5_path, split_path, split_key, n_soils, draw_seed, frozen_path):
    """Return (ids, soil dict of arrays). Freezes the list on first call."""
    import h5py

    if os.path.isfile(frozen_path):
        with open(frozen_path, "r") as fh:
            frozen = json.load(fh)
        print(f"soil set   : REUSED from {frozen_path}")
        print(f"             frozen on {frozen['frozen_utc']}, "
              f"draw_seed={frozen['draw_seed']}, n={len(frozen['ids'])}")
        ids = frozen["ids"]
    else:
        with open(split_path, "r") as fh:
            split = json.load(fh)
        pool = split[split_key]
        if n_soils and n_soils < len(pool):
            rng = np.random.default_rng(draw_seed)
            idx = rng.choice(len(pool), size=n_soils, replace=False)
            ids = [pool[int(i)] for i in sorted(idx)]
        else:
            ids = list(pool)
        frozen = {
            "split_key": split_key,
            "draw_seed": draw_seed,
            "n_pool": len(pool),
            "n_drawn": len(ids),
            "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "ids": ids,
        }
        os.makedirs(os.path.dirname(frozen_path), exist_ok=True)
        with open(frozen_path, "w") as fh:
            json.dump(frozen, fh, indent=2)
        print(f"soil set   : FROZEN to {frozen_path}  (n={len(ids)} of {len(pool)})")

    soil = {k: [] for k in SOIL_KEYS + ["Se_i", "theta_r"]}
    with h5py.File(h5_path, "r") as f:
        area = float(f.attrs["area"])
        for rid in ids:
            g = f[rid]
            for k in SOIL_KEYS + ["Se_i", "theta_r"]:
                soil[k].append(float(g.attrs[k]))
    soil = {k: np.asarray(v, dtype=np.float64) for k, v in soil.items()}
    return ids, soil, area


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def load_mlp(ckpt_path, in_dim, out_dim, device):
    """Loads either checkpoint format.

    The rebuild changed three things that a permissive loader would absorb in
    silence, so each is checked rather than defaulted:

      FORMAT. Withdrawn checkpoints carry an "args" dict and "model_state".
      Rebuilt ones carry "model" and put lam, seed and output_activation at the
      top level, with no "args" at all. Width and depth are not recorded in the
      rebuilt format because they never varied, so they come from the CLI.

      OUTPUT LAYER. Rebuilt models end in a sigmoid, which has no parameters.
      Building the network without it would load the weights CLEANLY and then
      predict outside [0, 1], where Se**m with m up to 13 diverges. The
      controller would score candidates against a broken forward model and
      report a number rather than an error. So the activation is read from the
      checkpoint and refused if absent.

      The network is rebuilt from the checkpoint's own recorded width, depth
      and output activation. The old fallback silently built
      an unbounded network when that import failed, which is exactly the
      failure above. There is no fallback here.
    """
    import torch
    import torch.nn as nn

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "model_state" in ck and "args" in ck:
        fmt = "withdrawn"
        a = ck["args"]
        state = ck["model_state"]
        w, d = int(a["width"]), int(a["depth"])
        act = "none"
        meta_lam = float(a.get("lam", float("nan")))
        meta_seed = int(a.get("seed", -1))
        meta_tag = str(a.get("tag", Path(ckpt_path).stem))
        meta_val = float(ck.get("val_loss", float("nan")))
    elif "model" in ck:
        fmt = "rebuild"
        state = ck["model"]
        w, d = int(ck.get("width", 512)), int(ck.get("depth", 4))
        act = ck.get("output_activation")
        if act is None:
            raise SystemExit(
                f"{ckpt_path}: rebuilt checkpoint with no output_activation. "
                "Refusing to guess: an unbounded head where a sigmoid belongs "
                "produces predictions outside [0, 1] and a silently wrong "
                "controller.")
        meta_lam = float(ck.get("lam", float("nan")))
        meta_seed = int(ck.get("seed", -1))
        # Sweep checkpoints are one folder per run, all named best.pt, so the
        # folder carries the identity. Released checkpoints sit flat in models/
        # under their own names, so fall back to the stem there.
        _parent = Path(ckpt_path).parent.name
        meta_tag = (Path(ckpt_path).stem if _parent == "models"
                    else _parent)
        meta_val = float(ck.get("monitor", float("nan")))
    else:
        raise SystemExit(f"{ckpt_path}: unrecognised checkpoint format, "
                         f"keys {sorted(ck.keys())}")

    L = [nn.Linear(in_dim, w), nn.BatchNorm1d(w), nn.ReLU()]
    for _ in range(d - 1):
        L += [nn.Linear(w, w), nn.BatchNorm1d(w), nn.ReLU()]
    L.append(nn.Linear(w, out_dim))
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
    # strict: a missing or unexpected key means the architecture does not match
    # the weights, and loading anyway would score a different model than the
    # one that was trained.
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    meta = {"lam": meta_lam, "seed": meta_seed, "tag": meta_tag,
            "epoch": int(ck.get("epoch", -1)), "val_loss": meta_val,
            "width": w, "depth": d,
            "format": fmt, "output_activation": act}
    return model, meta


# ---------------------------------------------------------------------------
# one day of control, batched over soils
# ---------------------------------------------------------------------------
def decide_R(model, Se_now, cand_R, cand_R_scaled, soil_scaled, se_sp, D_H,
             device, chunk):
    """Se_now (S,221). cand_R (S,C) physical. Returns R (S,) physical."""
    import torch
    S, C = cand_R.shape
    best = np.empty(S, dtype=np.float64)
    for lo in range(0, S, chunk):
        hi = min(lo + chunk, S)
        s = hi - lo
        Se0 = np.repeat(Se_now[lo:hi], C, axis=0)                 # (s*C, 221)
        Se = torch.as_tensor(Se0, dtype=torch.float32, device=device)
        suf = torch.zeros(s * C, 5, dtype=torch.float32, device=device)
        suf[:, 0] = torch.as_tensor(cand_R_scaled[lo:hi].reshape(-1),
                                    dtype=torch.float32, device=device)
        suf[:, 1:] = torch.as_tensor(np.repeat(soil_scaled[lo:hi], C, axis=0),
                                     dtype=torch.float32, device=device)
        with torch.no_grad():
            for _h in range(D_H):
                Se = model(torch.cat([Se, suf], dim=1)).clamp(0.0, 1.0)
            score = ((Se - se_sp) ** 2).mean(dim=1).reshape(s, C)
            j = torch.argmin(score, dim=1).cpu().numpy()
        best[lo:hi] = cand_R[lo:hi][np.arange(s), j]
    return best


def plant_day(theta0, R, soil, area, Dz, Dt, D_H, n_iter, H, diag=True):
    """Advance every soil one real day. Returns (Se (S,221,24), diag dict)."""
    from column_solver import run_batch, ON_GPU
    Q = (R / 100.0 * area).reshape(-1, 1)                          # (S,1) m3/day
    out = run_batch(Q,
                    Ks=soil["Ks"], theta_s=soil["theta_s"],
                    theta_r=soil["theta_r"], alpha=soil["alpha"],
                    n=soil["n_vgm"], H=H, Area=area, tf=1,
                    Dz=Dz, Dt=Dt, D_H=D_H, n_iter=n_iter,
                    theta0=theta0, diag=diag)
    Se, dg = (out[0], out[-1]) if diag else (out, None)
    if ON_GPU:
        import cupy as cp
        Se = cp.asnumpy(Se)
    return np.asarray(Se, dtype=np.float64), dg


def run_control(model, soil, soil_scaled, scal, area, args, device, r_max_in):
    S = soil["Ks"].shape[0]
    Nz1 = int(args.H * 100 / args.Dz) + 1

    r_min = float(scal["R_out"]["min"])
    r_cap = float(scal["R_out"]["max"])
    # Candidate ceiling. Two bounds, both per soil:
    #   operational cap      0.30 * Ks      (rearranged from Q_ub = 0.30*Ks*A/100)
    #   stability ceiling    0.95 * R_stable (measured by rmax_ceiling.py)
    #   scaler ceiling       R_out max      (keeps the scaled input inside [0,1])
    # The stability ceiling is a property of the SOIL, measured against the plant,
    # so every lambda case gets an identical action set and the closed-loop
    # comparison stays clean. It binds on 4 of 64 validation soils.
    r_max = np.minimum(np.minimum(r_max_in, 0.30 * soil["Ks"]), r_cap)
    w = np.linspace(0.0, 1.0, args.candidates)[None, :]            # (1,C)
    cand_R = r_min + (r_max[:, None] - r_min) * w                  # (S,C)
    cand_scaled = (cand_R - r_min) / (r_cap - r_min)

    Se_now = np.repeat(soil["Se_i"][:, None], Nz1, axis=1)         # (S,221)
    theta = Se_now * (soil["theta_s"] - soil["theta_r"])[:, None] \
        + soil["theta_r"][:, None]

    R_hist = np.zeros((S, args.tf))
    eq32_hourly_cum = np.zeros(S)
    eq32_daily = np.zeros((S, args.tf))
    meanSe_daily = np.zeros((S, args.tf))
    div_final = np.zeros(S, dtype=np.int64)     # hours out of bounds, last pass
    max_resid = np.zeros(S)
    n_ceiling_hits = np.zeros(S, dtype=np.int64)

    t0 = time.perf_counter()
    for day in range(args.tf):
        R = decide_R(model, Se_now, cand_R, cand_scaled, soil_scaled,
                     args.se_sp, args.D_H, device, args.chunk_soils)
        Se_day, dg = plant_day(theta, R, soil, area, args.Dz, args.Dt,
                               args.D_H, args.n_iter, args.H)       # (S,221,24)
        # HARD ABORT on non-finite. A NaN quietly poisons a mean and the
        # run still "completes"; that is worse than crashing because it
        # can be reported by accident. Never let it through.
        if not np.isfinite(Se_day).all():
            bad = np.where(~np.isfinite(Se_day).all(axis=(1, 2)))[0]
            raise RuntimeError(
                f"non-finite plant output on day {day+1}, soils "
                f"{bad.tolist()[:10]} (indices), R="
                f"{np.round(R[bad][:10], 4).tolist()}. The stability ceiling did "
                f"not hold. Do NOT use these numbers.")
        if dg is not None:
            div_final += dg["n_div_final"].astype(np.int64)
            max_resid = np.maximum(max_resid, dg["max_residual"])
        n_ceiling_hits += (R >= r_max - 1e-9).astype(np.int64)
        dev2 = (Se_day - args.se_sp) ** 2
        eq32_hourly_cum += dev2.mean(axis=1).sum(axis=1)            # sum over 24 h
        Se_end = Se_day[:, :, -1]
        eq32_daily[:, day] = ((Se_end - args.se_sp) ** 2).mean(axis=1)
        meanSe_daily[:, day] = Se_end.mean(axis=1)
        R_hist[:, day] = R
        theta = Se_end * (soil["theta_s"] - soil["theta_r"])[:, None] \
            + soil["theta_r"][:, None]
        Se_now = Se_end                                             # RE-ANCHOR
        if args.heartbeat:
            el = time.perf_counter() - t0
            print(f"    day {day+1:>3}/{args.tf}  {el:6.1f}s  "
                  f"mean Se {Se_end.mean():.4f}  "
                  f"mean R {R.mean():7.3f}", flush=True)

    settle = np.full(S, -1, dtype=int)
    for s in range(S):
        ok = np.abs(meanSe_daily[s] - args.se_sp) < args.settle_tol
        for d in range(args.tf):
            if ok[d:].all():
                settle[s] = d + 1
                break
    return dict(R=R_hist, eq32_hourly_cum=eq32_hourly_cum,
                eq32_daily=eq32_daily, meanSe_daily=meanSe_daily,
                settle=settle, cand_r_max=r_max, div_final=div_final,
                max_resid=max_resid, n_ceiling_hits=n_ceiling_hits,
                wall_s=time.perf_counter() - t0)


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="PGML closed-loop control run")
    p.add_argument("--root", default=DEF_ROOT)
    p.add_argument("--h5", default=DEF_H5)
    p.add_argument("--split", default="val", choices=["val", "test"],
                   help="val = selection set. test = held-out set.")
    p.add_argument("--soil_table", default="",
                   help="a constructed soil table from "
                        "build_extrapolation_soils.py. "
                        "Parameters are read from the table itself, so no h5 "
                        "is opened and --split, --n_soils and --draw_seed are "
                        "ignored. Requires --ceiling naming that cell's own "
                        "rmax_<cell>.json.")
    p.add_argument("--n_soils", type=int, default=0,
                   help="0 = all soils in the split")
    p.add_argument("--draw_seed", type=int, default=42)
    p.add_argument("--ceiling", default="",
                   help="rmax_<split>.json from rmax_ceiling.py. "
                        "Default: rmax_<split>.json in cwd.")
    p.add_argument("--no_ceiling", action="store_true",
                   help="run with the 0.30*Ks cap only. Not for selection: "
                        "the plant can return non-converged profiles.")
    p.add_argument("--sweep_dir", default="",
                   help="Directory holding the checkpoints. Default is "
                        "04_Training/models. "
                        "Point it elsewhere to score a different set of arms "
                        "against the same soils, as W4 does.")
    p.add_argument("--ckpt_glob", default="*.pt")
    p.add_argument("--limit_ckpts", type=int, default=0)
    p.add_argument("--out_dir", default="")
    p.add_argument("--candidates", type=int, default=200)
    p.add_argument("--chunk_soils", type=int, default=128)
    p.add_argument("--tf", type=int, default=25)
    p.add_argument("--se_sp", type=float, default=0.70)
    p.add_argument("--settle_tol", type=float, default=0.01)
    p.add_argument("--Dz", type=float, default=2.5)
    p.add_argument("--Dt", type=float, default=1.0 / 24.0)
    p.add_argument("--D_H", type=int, default=24)
    p.add_argument("--H", type=float, default=5.50)
    p.add_argument("--n_iter", type=int, default=20)
    p.add_argument("--heartbeat", action="store_true")
    args = p.parse_args()

    root = args.root
    prep_dir = paths.REFERENCE
    train_dir = os.path.join(root, "04_Training")
    sweep_dir = (args.sweep_dir if args.sweep_dir
                 else os.path.join(train_dir, "models"))
    paths.add_stage_paths()

    from preprocess import INPUT_DIM, OUTPUT_DIM, load_scalers, apply_scaler
    from preprocess import SOIL_PARAM_KEYS
    import torch

    if list(SOIL_PARAM_KEYS) != SOIL_KEYS:
        print(f"ABORT: preprocess SOIL_PARAM_KEYS {list(SOIL_PARAM_KEYS)} "
              f"does not match this harness {SOIL_KEYS}. Fix before running.")
        return

    out_dir = args.out_dir or os.path.join(
        paths.RESULTS, "closed_loop",
        "extrapolation" if args.soil_table else args.split)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print("PGML CLOSED LOOP")
    print("=" * 78)
    print(f"device     : {device}")
    if args.soil_table:
        print("split      : none. Constructed soils.")
        print("             Neither the selection set nor the held-out set.")
    else:
        print(f"split      : {args.split}  "
              f"({'selection' if args.split == 'val' else 'held-out'})")
    print(f"out_dir    : {out_dir}")

    scal = load_scalers(prep_dir)
    if args.soil_table:
        try:
            ids, soil, area, cell = load_soil_table(args.soil_table)
        except RuntimeError as e:
            print(f"ABORT: {e}")
            return
        print(f"soil table : {args.soil_table}")
        print(f"             cell {cell}, {len(ids)} rows, area {area:.1f} m2")
    else:
        cell = None
        # The shipped soil list lives in reference/. A subset run (--n_soils)
        # draws its own and freezes it beside the results instead, so the
        # shipped list is never overwritten.
        frozen = (os.path.join(out_dir, f"soils_{args.split}.json")
                  if args.n_soils
                  else paths.reference(f"soils_{args.split}.json"))
        ids, soil, area = build_soil_set(
            args.h5, os.path.join(prep_dir, "split.json"), args.split,
            args.n_soils, args.draw_seed, frozen)
    S = len(ids)

    # ------------------------------------------------------------ ceiling
    # A constructed run must NEVER silently fall back on rmax_val.json.
    # The guard below would catch the mismatch, but naming the cell's own file
    # by default removes the chance of pointing at the wrong one to begin with.
    ceil_path = a_ceiling = args.ceiling or (
        os.path.join(paths.EXTRAP_CEILINGS, f"rmax_{cell}.json")
        if args.soil_table
        else paths.reference(f"rmax_{args.split}.json"))
    if args.no_ceiling:
        r_max_in = np.full(S, np.inf)
        ceil_meta = {"mode": "0.30Ks only, no stability ceiling"}
        print("ceiling    : 0.30*Ks ONLY -- not valid for a selection run")
    else:
        if not os.path.isfile(ceil_path):
            print(f"ABORT: no ceiling file at {ceil_path}. Run "
                  f"rmax_ceiling.py first, or pass --no_ceiling "
                  f"(diagnostic only).")
            return
        with open(ceil_path, "r") as fh:
            cj = json.load(fh)
        if list(cj["ids"]) != list(ids):
            print(f"ABORT: ceiling soil list does not match the frozen soil "
                  f"list.\n  ceiling n={len(cj['ids'])}  soils n={len(ids)}\n"
                  f"  Regenerate the ceiling for THIS soil set.")
            return
        r_max_in = np.asarray(cj["r_max"], dtype=np.float64)
        ceil_meta = {k: cj[k] for k in
                     ("created_utc", "safety", "div_tol", "res_tol",
                      "start_state", "n_R", "criterion") if k in cj}
        n_bind = int((r_max_in < 0.30 * soil["Ks"] - 1e-9).sum())
        print(f"ceiling    : {ceil_path}")
        print(f"             stability limit binds on {n_bind} of {S} "
              f"soils; 0.30*Ks governs the rest")

    soil_scaled = np.stack(
        [apply_scaler(soil[k], k, scal) for k in SOIL_KEYS], axis=1
    ).astype(np.float32)

    print(f"soils      : {S}")
    print(f"  Ks       : {soil['Ks'].min():.2f} to {soil['Ks'].max():.2f}")
    print(f"  alpha    : {soil['alpha'].min():.5f} to {soil['alpha'].max():.5f}")
    print(f"  n_vgm    : {soil['n_vgm'].min():.4f} to {soil['n_vgm'].max():.4f}")
    print(f"  theta_s  : {soil['theta_s'].min():.4f} to {soil['theta_s'].max():.4f}")
    print(f"  Se_i     : {soil['Se_i'].min():.4f} to {soil['Se_i'].max():.4f}")
    print(f"setpoint   : {args.se_sp}   days: {args.tf}   "
          f"candidates: {args.candidates}   chunk: {args.chunk_soils}")
    print(f"R grid     : [{scal['R_out']['min']:.4f}, per-soil "
          f"min(0.30*Ks, {scal['R_out']['max']:.4f})]")

    # recursive: rebuilt checkpoints live one folder deep and are all named
    # best.pt, so a flat glob finds nothing and a flat name is not unique.
    cks = sorted(glob.glob(os.path.join(sweep_dir, args.ckpt_glob),
                           recursive=True))
    if args.limit_ckpts:
        cks = cks[:args.limit_ckpts]
    print(f"checkpoints: {len(cks)}\n")

    rows = []
    for i, ck in enumerate(cks):
        model, meta = load_mlp(ck, INPUT_DIM, OUTPUT_DIM, device)
        # meta["tag"] is the file stem for withdrawn checkpoints and the parent
        # folder name for rebuilt ones. Using the stem alone would name every
        # rebuilt output cl_best.json and overwrite eighteen results into one.
        name = meta["tag"]
        print("-" * 78)
        print(f"[{i+1}/{len(cks)}] {name}   lam={meta['lam']}  "
              f"seed={meta['seed']}  best_epoch={meta['epoch']}  "
              f"out={meta['output_activation']}  fmt={meta['format']}")
        try:
            res = run_control(model, soil, soil_scaled, scal, area, args,
                               device, r_max_in)
        except RuntimeError as e:
            print(f"    ABORTED: {e}")
            continue

        cum_h = res["eq32_hourly_cum"]
        cum_d = res["eq32_daily"].sum(axis=1)
        print(f"    PRIMARY   cum hourly : mean {cum_h.mean():.6e}  "
              f"median {np.median(cum_h):.6e}  max {cum_h.max():.6e}")
        print(f"    secondary cum daily  : mean {cum_d.mean():.6e}  "
              f"median {np.median(cum_d):.6e}")
        print(f"    terminal mean Se : {res['meanSe_daily'][:, -1].mean():.4f}   "
              f"settling day (median of resolved): "
              f"{int(np.median(res['settle'][res['settle'] > 0])) if (res['settle'] > 0).any() else -1}")
        print(f"    unresolved soils : {(res['settle'] < 0).sum()} of {S}     "
              f"wall {res['wall_s']:.1f}s")
        print(f"    convergence      : soils with any non-converged hour "
              f"{int((res['div_final'] > 0).sum())} of {S}   "
              f"max residual {res['max_resid'].max():.3e}")
        print(f"    ceiling-limited  : {int((res['n_ceiling_hits'] > 0).sum())} "
              f"soils hit their r_max on at least one day")

        payload = dict(
            checkpoint=name, split=args.split, n_soils=S,
            lam=meta["lam"], seed=meta["seed"], tag=meta["tag"],
            best_epoch=meta["epoch"], val_loss=meta["val_loss"],
            se_sp=args.se_sp, tf=args.tf, candidates=args.candidates,
            metric_primary="cum_eq32_hourly", metric_secondary="cum_eq32_daily",
            cell=cell,
            soil_table=args.soil_table or None,
            soil_ids=ids,
            cum_eq32_hourly=cum_h.tolist(),
            cum_eq32_daily=cum_d.tolist(),
            eq32_daily=res["eq32_daily"].tolist(),
            meanSe_daily=res["meanSe_daily"].tolist(),
            R=res["R"].tolist(),
            settle_day=res["settle"].tolist(),
            cand_r_max=res["cand_r_max"].tolist(),
            div_final=res["div_final"].tolist(),
            max_resid=res["max_resid"].tolist(),
            n_ceiling_hits=res["n_ceiling_hits"].tolist(),
            ceiling_source=ceil_path if not args.no_ceiling else None,
            ceiling_meta=ceil_meta,
            wall_s=res["wall_s"],
        )
        out_name = f"cl_{cell}_{name}.json" if cell else f"cl_{name}.json"
        with open(os.path.join(out_dir, out_name), "w") as fh:
            json.dump(payload, fh)

        rows.append(dict(
            cell=cell,
            checkpoint=name, lam=meta["lam"], seed=meta["seed"], n_soils=S,
            cum_eq32_hourly_mean=float(cum_h.mean()),
            cum_eq32_hourly_median=float(np.median(cum_h)),
            cum_eq32_hourly_sd=float(cum_h.std(ddof=1)),
            cum_eq32_hourly_max=float(cum_h.max()),
            cum_eq32_daily_mean=float(cum_d.mean()),
            cum_eq32_daily_median=float(np.median(cum_d)),
            terminal_meanSe=float(res["meanSe_daily"][:, -1].mean()),
            unresolved=int((res["settle"] < 0).sum()),
            soils_nonconverged=int((res["div_final"] > 0).sum()),
            max_residual=float(res["max_resid"].max()),
            soils_at_ceiling=int((res["n_ceiling_hits"] > 0).sum()),
            wall_s=float(res["wall_s"]),
        ))

    if rows:
        csv_path = os.path.join(out_dir, f"summary_{args.split}.csv")
        with open(csv_path, "w", newline="") as fh:
            wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wtr.writeheader()
            for r in rows:
                wtr.writerow(r)
        print("\n" + "=" * 78)
        print(f"wrote {csv_path}")
        print("This script does not select. The rule applied was: a lambda")
        print("case wins only if it beats the others at all three seeds;")
        print("cases within one across-seed SD are a plateau; with no")
        print("separation, lambda = 0 ships by parsimony.")

    print("\ndone.")


if __name__ == "__main__":
    main()
