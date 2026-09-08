"""
train.py
========
Trains an MLP to predict the effective saturation profile one hour ahead, from
the current profile, the irrigation rate over the step, and four soil
properties.

OBJECTIVE
  loss = (1 - lam) * data / data.detach() + lam * phys / phys.detach()

  Convex, so the coefficients sum to one and lam is a proportion. Each term is
  divided by its own detached value, so both are order one and parity sits at
  lam 0.5. The physics term is the Gardner residual from physics_loss.py.
  There is no gradient clipping; the gradient norm is logged per step as a
  distribution rather than a mean.

SELECTION
  Checkpointing, early stopping and the plateau schedule all key off one
  quantity: mean squared error against the true next Se on the validation
  soils, unnormalised, unweighted, in effective-saturation units. It contains
  no lam, so runs at different lam are comparable.

LOGGED PER EPOCH
  Both loss terms, the gradient norm distribution, and the fraction of
  predicted nodes outside [0, 1].

FLAGS
  --lam                 physics weight, 0 to 1. Required.
  --out                 output folder for this run. Required.
  --normalisation       ratio | none. "none" combines the raw terms, so parity
                        moves during the run and the lam grid must be spaced
                        accordingly.
  --output_activation   none | sigmoid. Se is bounded in [0, 1] by definition;
                        a sigmoid enforces that. Bandai 2021.
  --train_frac          fraction of training runs, nested subsets, independent
                        of --seed. Subsampled after the scalers are fitted on
                        the full train set.

ARCHITECTURE
  MLP width 512, depth 4, BatchNorm and ReLU. Adam, lr 0.004 at batch 4096,
  five-epoch linear warmup, 100 epochs.

Usage, from 04_Training:
  python train.py --lam 0.4 --seed 42 --out runs/sweep/s42_lam0p4

--h5 and --prep default through paths.py to data/dataset.h5 and reference/.

Status: stable
Depends on: physics_loss.py, preprocess.py, paths.py

Changelog
2026.08.17  v1.6  Comment corrected. The alpha_g conversion was
                  attributed to Olivares 2025. The constants 2.736 and
                  0.711 are this work's, fitted offline. Olivares
                  supplies the saturation-form relations carrying beta
                  and nothing else here. Comment only, no numerical
                  change.
2026.08.11  v1.5  --train_frac for the data ablation. Subsamples whole runs
                  from the assembled training tensor after the pipeline has
                  fitted its scalers on the full train set, so the input space
                  and the validation set are identical at every fraction.
                  Subsets are nested and independent of --seed.
2026.08.11  v1.4  Version stamped into log.json and the console header, so a
                  result can be traced to the code that produced it.
2026.08.11  v1.3  --normalisation flag, "ratio" or the unnormalised fallback,
                  both convex. The log now records the form and the effective
                  physics weight lam/(1 - lam) so runs under the two schemes can
                  be placed on one axis.
2026.08.11  v1.2  Table formatting: three significant figures rather than seven,
                  a rule every ten epochs, best-epoch marker at the right edge.
                  Silenced a UserWarning by taking .item() on the loss tensors.
2026.08.11  v1.1  Per-epoch timing and a running estimate of the remaining
                  time, both printed and recorded. The estimate uses a trailing
                  mean over ten epochs so one slow epoch does not swing it.
2026.08.11  v1.0  Initial training loop for the convex objective.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
import paths                                                       # noqa: E402
paths.add_stage_paths()

from physics_loss import physics_loss                              # noqa: E402
from preprocess import (build_dataloaders, load_scalers,           # noqa: E402
                        GPUBatchLoader, SAMPLES_PER_RUN,
                        INPUT_DIM, OUTPUT_DIM, SOIL_PARAM_KEYS)

# Gardner parameters follow from the sampled van Genuchten n by a fixed rule,
# not by fitting: beta = 3 + 2/(n - 2) is the conductivity exponent and
# alpha_g = 2.736 * (n - 2)**0.711 * alpha is this work's conversion, fitted
# offline over 512 Sobol soils. Olivares 2025 supplies the saturation-form
# relations carrying beta, not this conversion. Restated here rather than
# imported so the loop has no dependency on the fitting script.
# Stamped into every log so a result can be traced to the code that produced it.
# Numerics are unchanged from v1.0 for any run with --normalisation ratio: the
# intervening edits were timing, table formatting, and a flag whose default
# reproduces the earlier behaviour exactly.
__version__ = "1.6"

ALPHA_G_PREFACTOR = 2.736
ALPHA_G_EXPONENT = 0.711

# Width of the per-epoch table. Fixed here so header, rules and rows cannot
# drift apart when a column is added.
RULE = "-" * 116


# =============================================================================
# Training-set subsampling
#
# For the data ablation. Three properties are needed and none of them survive
# simply passing fewer runs into build_dataloaders:
#
#   SCALERS UNCHANGED. build_dataloaders fits the min-max scalers on whatever
#   train set it is given. Fitting them on a tenth of the runs would change the
#   network's inputs, and the ablation would then be comparing two different
#   input spaces rather than two data volumes. So the full pipeline runs first
#   and the subsampling happens afterwards, on the assembled tensors.
#
#   VALIDATION UNCHANGED. All 614 validation soils in every run, so the monitor
#   means the same thing at every fraction.
#
#   SUBSET BY RUN, NOT BY SAMPLE. The question is how many SOILS the network
#   has seen. Dropping random timesteps from every soil would leave all 2867
#   soils represented and test something else entirely. Each run occupies a
#   contiguous block of SAMPLES_PER_RUN rows, so whole blocks are selected.
#
# The subsets are nested and do not depend on --seed: the run order is shuffled
# once with a fixed generator and the first k taken. So the 10 percent set is a
# subset of the 30 percent set, and a seed sweep varies initialisation without
# also varying which soils were seen.
# =============================================================================
SUBSET_SEED = 20260811


class _TensorView:
    """Duck-types the two attributes GPUBatchLoader reads off a HeapDataset."""

    def __init__(self, X, Y):
        self.X, self.Y = X, Y

    def __len__(self):
        return self.X.shape[0]


def subsample_by_run(loader, frac, batch_size):
    if frac >= 1.0:
        return loader, None

    n_samples = len(loader.dataset)
    if n_samples % SAMPLES_PER_RUN != 0:
        raise SystemExit(
            f"train set has {n_samples} samples, not a whole number of runs "
            f"of {SAMPLES_PER_RUN}; run-block subsampling would be wrong")

    n_runs = n_samples // SAMPLES_PER_RUN
    keep = max(1, int(round(frac * n_runs)))

    g = np.random.default_rng(SUBSET_SEED)
    order = g.permutation(n_runs)[:keep]
    order.sort()                       # keep the original run order

    rows = np.concatenate([np.arange(i * SAMPLES_PER_RUN,
                                     (i + 1) * SAMPLES_PER_RUN)
                           for i in order])
    idx = torch.from_numpy(rows).to(loader.dataset.X.device)

    view = _TensorView(loader.dataset.X[idx], loader.dataset.Y[idx])
    info = {"n_runs_full": n_runs, "n_runs_kept": keep,
            "n_samples_kept": int(view.X.shape[0]),
            "subset_seed": SUBSET_SEED}
    print(f"[ablation] train set cut to {keep}/{n_runs} runs "
          f"({view.X.shape[0]:,} samples, frac {frac})")
    return GPUBatchLoader(view, batch_size=batch_size, shuffle=True), info


# =============================================================================
# Model
# =============================================================================
class MLP(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM,
                 width=512, depth=4, output_activation="none"):
        super().__init__()
        layers = [nn.Linear(input_dim, width), nn.BatchNorm1d(width), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.BatchNorm1d(width), nn.ReLU()]
        layers.append(nn.Linear(width, output_dim))
        if output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif output_activation != "none":
            raise ValueError(f"unknown output_activation: {output_activation}")
        self.net = nn.Sequential(*layers)
        self.output_activation = output_activation

    def forward(self, x):
        return self.net(x)


# =============================================================================
# Physics term
# The network works in effective saturation; the residual operator works in
# water content. The conversion needs the run's own theta_s, which arrives
# scaled in the input tail and has to be put back into physical units first.
# =============================================================================
def unscale(x_col, key, scalers):
    lo, hi = scalers[key]["min"], scalers[key]["max"]
    return x_col * (hi - lo) + lo


def physics_term(x_batch, y_pred, scalers, Dz, Dt):
    base = OUTPUT_DIM + 1
    soil = {k: unscale(x_batch[:, base + i:base + i + 1], k, scalers)
            for i, k in enumerate(SOIL_PARAM_KEYS)}

    mn = soil["n_vgm"] - 2.0
    beta = 3.0 + 2.0 / mn
    alpha_g = ALPHA_G_PREFACTOR * torch.pow(mn, ALPHA_G_EXPONENT) * soil["alpha"]

    R_phys = unscale(x_batch[:, OUTPUT_DIM:OUTPUT_DIM + 1], "R_out", scalers)

    theta_r = 0.0
    theta_s = soil["theta_s"]
    dth = theta_s - theta_r

    theta_t = theta_r + x_batch[:, :OUTPUT_DIM] * dth
    theta_pred = theta_r + y_pred * dth

    return physics_loss(theta_t, theta_pred, R_phys, theta_s, theta_r,
                        soil["Ks"], beta, alpha_g, Dz, Dt)


# =============================================================================
# Epochs
# =============================================================================
def train_epoch(model, loader, criterion, optimizer, device, scalers,
                lam, normalisation, Dz, Dt):
    model.train(True)
    tot_d, tot_p, n = 0.0, 0.0, 0
    norms = []

    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        y_pred = model(x_batch)
        data_l = criterion(y_pred, y_batch)
        phys_l = physics_term(x_batch, y_pred, scalers, Dz, Dt)

        # Convex form throughout. Under "ratio" each term carries its own
        # reference and detach() keeps that divisor out of the graph, so it acts
        # as a scale rather than as a second objective. Under "none" the raw
        # terms are combined and lam alone sets the balance.
        #
        # A term is SKIPPED, not multiplied by zero, when its coefficient is
        # zero. That matters at lam = 1, where the data term is deleted rather
        # than down-weighted, and it prevents a 0 * nan from a term that never
        # contributes.
        loss = torch.zeros((), device=device)
        if lam < 1.0:
            d = (data_l / data_l.detach().clamp(min=1e-30)
                 if normalisation == "ratio" else data_l)
            loss = loss + (1.0 - lam) * d
        if lam > 0.0:
            p_ = (phys_l / phys_l.detach().clamp(min=1e-30)
                  if normalisation == "ratio" else phys_l)
            loss = loss + lam * p_

        optimizer.zero_grad()
        loss.backward()

        # No clipping. The norm is measured only, never applied, so the
        # question of whether a cap is needed is settled by the record.
        with torch.no_grad():
            sq = sum(float((p.grad ** 2).sum()) for p in model.parameters()
                     if p.grad is not None)
        norms.append(sq ** 0.5)

        optimizer.step()

        tot_d += data_l.item()
        tot_p += phys_l.item()
        n += 1

    a = np.array(norms)
    return {
        "data": tot_d / n, "phys": tot_p / n,
        "grad_median": float(np.median(a)), "grad_p99": float(np.percentile(a, 99)),
        "grad_max": float(a.max()), "grad_mean": float(a.mean()),
        "steps": n,
    }


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, scalers, Dz, Dt):
    """Both terms unnormalised. The data figure here IS the selection monitor,
    so it must not carry lam or any scaling."""
    model.eval()
    tot_d, tot_p, n = 0.0, 0.0, 0
    out_lo, out_hi, n_val = 0, 0, 0

    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        y_pred = model(x_batch)
        tot_d += float(criterion(y_pred, y_batch))
        tot_p += float(physics_term(x_batch, y_pred, scalers, Dz, Dt))
        out_lo += int((y_pred < 0.0).sum())
        out_hi += int((y_pred > 1.0).sum())
        n_val += y_pred.numel()
        n += 1

    return {
        "data": tot_d / n, "phys": tot_p / n,
        "frac_below_zero": out_lo / n_val,
        "frac_above_one": out_hi / n_val,
    }


# =============================================================================
# Run
# =============================================================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    if args.h5 is None:
        args.h5 = paths.require_h5()
    if args.prep is None:
        args.prep = paths.REFERENCE

    train_loader, val_loader, _ = build_dataloaders(
        h5_path=args.h5, out_dir=args.prep, batch_size=args.batch_size)
    scalers = load_scalers(args.prep)

    train_loader, subset_info = subsample_by_run(
        train_loader, args.train_frac, args.batch_size)

    model = MLP(width=args.width, depth=args.depth,
                output_activation=args.output_activation).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=args.patience)

    print(f"train.py v{__version__}   device {device}")
    print(f"lam {args.lam}   seed {args.seed}   "
          f"output {args.output_activation}   norm {args.normalisation}")
    print(f"batch {args.batch_size}   lr {args.lr}   warmup {args.warmup_epochs}"
          f"   epochs {args.epochs}   clipping OFF")
    print(f"train_frac {args.train_frac}   "
          f"steps/epoch {len(train_loader)}   "
          f"val batches {len(val_loader)}")
    print()
    print(RULE)
    print(f"{'ep':>4}  {'MONITOR':>10}  {'val_phys':>10}  {'tr_data':>10}  "
          f"{'tr_phys':>10}  {'gn_med':>9}  {'gn_max':>9}  {'out%':>7}  "
          f"{'lr':>8}  {'s/ep':>6}  {'eta':>7}")
    print(RULE)

    best, best_ep, rows = float("inf"), -1, []
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        t_ep = time.time()
        if args.warmup_epochs > 0 and ep <= args.warmup_epochs:
            for g in optimizer.param_groups:
                g["lr"] = args.lr * ep / args.warmup_epochs

        tr = train_epoch(model, train_loader, criterion, optimizer, device,
                         scalers, args.lam, args.normalisation,
                         args.Dz, args.Dt)
        ev = eval_epoch(model, val_loader, criterion, device, scalers,
                        args.Dz, args.Dt)

        monitor = ev["data"]
        if ep > args.warmup_epochs:
            scheduler.step(monitor)
        lr_now = optimizer.param_groups[0]["lr"]

        if monitor < best:
            best, best_ep = monitor, ep
            torch.save({"model": model.state_dict(),
                        "epoch": ep, "monitor": monitor,
                        "lam": args.lam, "seed": args.seed,
                        "output_activation": args.output_activation},
                       out / "best.pt")

        frac_out = ev["frac_below_zero"] + ev["frac_above_one"]
        dt_ep = time.time() - t_ep
        # Estimate the remainder from a trailing mean rather than the last
        # epoch, so a single slow epoch does not swing the figure.
        recent = [r["epoch_s"] for r in rows[-9:]] + [dt_ep]
        eta_min = (args.epochs - ep) * statistics.fmean(recent) / 60.0

        rows.append({"epoch": ep, "monitor": monitor, "lr": lr_now,
                     "epoch_s": dt_ep, "train": tr, "val": ev})

        mark = "  <" if best_ep == ep else ""
        print(f"{ep:4d}  {monitor:10.3e}  {ev['phys']:10.3e}  "
              f"{tr['data']:10.3e}  {tr['phys']:10.3e}  "
              f"{tr['grad_median']:9.3f}  {tr['grad_max']:9.2f}  "
              f"{100 * frac_out:6.2f}%  {lr_now:8.2e}  "
              f"{dt_ep:6.1f}  {eta_min:6.1f}m{mark}")
        if ep % 10 == 0 and ep != args.epochs:
            print(RULE)

    log = {
        "script": Path(__file__).name, "version": __version__,
        "lam": args.lam, "seed": args.seed,
        "output_activation": args.output_activation,
        "normalisation": args.normalisation,
        "normalisation_note": (
            "per-batch loss ratio (loss / loss.detach())"
            if args.normalisation == "ratio"
            else "none; raw terms combined, lam alone sets the balance"),
        "form": "convex: (1 - lam) * data + lam * phys",
        "effective_physics_weight": (
            None if args.lam >= 1.0 else args.lam / (1.0 - args.lam)),
        "clipping": None,
        "monitor": "unnormalised validation data MSE, Se units",
        "batch_size": args.batch_size, "lr": args.lr,
        "warmup_epochs": args.warmup_epochs, "patience": args.patience,
        "width": args.width, "depth": args.depth, "epochs": args.epochs,
        "train_frac": args.train_frac, "subset": subset_info,
        "steps_per_epoch": len(train_loader),
        "Dz": args.Dz, "Dt": args.Dt,
        "best_epoch": best_ep, "best_monitor": best,
        "wall_s": time.time() - t0,
        "epoch_s_median": statistics.median(r["epoch_s"] for r in rows),
        "epoch_s_min": min(r["epoch_s"] for r in rows),
        "epoch_s_max": max(r["epoch_s"] for r in rows),
        "rows": rows,
    }
    (out / "log.json").write_text(json.dumps(log, indent=1))

    print(RULE)
    print(f"best epoch {best_ep}   monitor {best:.6e}")
    print(f"epoch time  median {log['epoch_s_median']:.2f}s   "
          f"min {log['epoch_s_min']:.2f}s   max {log['epoch_s_max']:.2f}s")
    print(f"wall {log['wall_s'] / 60:.1f} min "
          f"(includes preprocessing before the first epoch)")
    print(f"wrote {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", default=None,
                   help="Path to the dataset h5. Default: data/dataset.h5")
    p.add_argument("--prep", default=None,
                   help="Folder holding scaler.json and split.json. "
                        "Default: reference/")
    p.add_argument("--out", required=True,
                   help="Output folder for this run, e.g. runs/sweep/s42_lam0p4")

    p.add_argument("--lam", type=float, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_activation", default="none",
                   choices=["none", "sigmoid"])
    p.add_argument("--normalisation", default="ratio",
                   choices=["ratio", "none"])
    p.add_argument("--train_frac", type=float, default=1.0,
                   help="Fraction of TRAINING RUNS to keep. Validation and the "
                        "scalers are unaffected. Subsets are nested and do not "
                        "depend on --seed.")

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=0.004)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=10)

    p.add_argument("--width", type=int, default=512)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--Dz", type=float, default=2.5)
    p.add_argument("--Dt", type=float, default=1.0 / 24.0)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
