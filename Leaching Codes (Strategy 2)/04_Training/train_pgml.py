"""
train_pgml.py
=============
The single training script for BOTH surrogate models. The loss is convex and
normalised:

    total = (1 - lambda) * data_n + lambda * physics_n

  data_n       : data_loss   / DATA_MAX  (MSE vs VGM truth, scaled to ~[0,1])
  physics_n    : physics_loss / PHYS_MAX (Gardner residual, scaled to ~[0,1])
  DATA_MAX,
  PHYS_MAX     : untrained-network term scales, HARDWIRED as constants below
                 (measured once on strategy2.h5, seed 42, by measure_ranges.py)
  lambda       : balance weight in [0,1]  (0 = data only, 1 = physics only)

lambda selects the balance, using identical machinery (same MLP, clipping,
scheduler, data) so the only difference is the weighting -- a fair
PGML-vs-baseline comparison:
    lambda  > 0  ->  PGML         (data + Gardner physics)   -> saves pgml_best.pt
    lambda == 0  ->  MLP baseline (data only, physics off)   -> saves mlp_best.pt

This supersedes the old standalone train_mlp.py (now redundant: --lam 0 here is the fair MLP control,
sharing this script's clipping/scheduler).

STRATEGY-AGNOSTIC:
  This trainer reads whatever h5 --h5 points at. Physics constants come from the
  scaler file (--out) and the CLI args, not from a hardcoded dataset. Soil being
  fixed (S2) or per-run (S3) is a property of the file and the phys wiring below,
  not of this script's logic.

SPACE NOTE (important):
  Se enters this file already physical [0,1] (NOT scaled). Only R and the soil
  params are min-max scaled. So:
    - data_loss runs directly on Se (no un-scaling),
    - physics_loss needs R in PHYSICAL units, so R is un-scaled before the
      residual sees it,
    - Se -> theta via theta = theta_r + Se*(theta_s - theta_r).

LAMBDA (convex, [0,1], REQUIRED):
  Pass --lam in [0,1]: 0 = MLP control (data only), 1 = physics only. Both loss
  terms are first normalised by their untrained scale (hardwired below), so lambda
  is a pure balance, not a magnitude. The old additive scale-match default
  (~9.2) no longer applies and has been removed; --lam is required and a guard
  rejects anything outside [0,1].

Usage (Windows; backslash paths, ^ line continuation):
    python train_pgml.py ^
        --h5   "Data generation\\Python dataset\\<dataset>.h5" ^
        --out  "Leaching Codes\\03_Preprocessing" ^
        --save "Leaching Codes\\04_Training" ^
        --lam  0.5

Origin: PGML variant of train_mlp.py
Status: active development
Depends on: preprocess.py (build_dataloaders, load_scalers), physics_loss.py, torch

Changelog
---------
2026.07.22  v2.3  Hardwired DATA_MAX and PHYS_MAX as module constants instead of
                  reading ranges.json at runtime, so the sweep has no external
                  file to lose or get out of sync. Values rounded to 5 significant
                  figures (numerically identical to the JSON). Dropped the json
                  import and the ranges load block. Re-measure and update the two
                  constants for S3 (different dataset -> different scale).
2026.07.22  v2.2  Convex/normalised loss for the lambda sweep (Action 1). Loss
                  is now (1 - lam) * data_n + lam * phys_n, each term divided by
                  its untrained scale from ranges.json (measure_ranges.py). lam
                  now lives in [0,1] (0 = data only, 1 = physics only), NOT the
                  old additive 0..inf. Changes: re-added the json import (dropped
                  in v1.4); run_epoch now takes data_max/phys_max; a [0,1] guard
                  rejects bad --lam. --lam is now REQUIRED: the additive
                  scale-match default (~9.2) is meaningless once the terms are
                  normalised, so the auto branch and measure_lambda() were
                  removed. Convex normalisation constants are hardwired (v2.3).
2026.07.18  v2.1  Added optional --seed (fixes weight init, shuffle, and CUDA
                  RNG) so runs differing in one setting can be compared without
                  random-init noise confounding the result. Omit --seed for a
                  fresh random run; default behaviour unchanged.
2026.07.18  v1.4  Docstring/readability pass, no behaviour change. Made the
                  header strategy-agnostic (dropped the strategy1.h5 naming from
                  the space note and usage). Replaced the always-0.33 theta_s
                  ternary with a plain constant and an explicit note that per-run
                  soil is an S3 wiring change. Removed an unused json import.
2026.06.14  v1.3  Added optional --tag for the lambda sweep: tags the checkpoint
                  name so each lambda keeps its own file (--tag lam100 ->
                  pgml_lam100.pt) instead of every lam>0 run overwriting
                  pgml_best.pt. Backwards-compatible: no --tag = unchanged
                  mlp_best.pt / pgml_best.pt naming.
2026.06.13  v1.2  This file is now the SINGLE trainer for both models. Checkpoint
                  auto-named by lambda: mlp_best.pt (lam=0) vs pgml_best.pt
                  (lam>0). Supersedes standalone train_mlp.py (redundant: --lam 0
                  here is the fair MLP control with identical clipping/scheduler).
                  Training table widened for readability.
2026.06.13  v1.1  Added gradient-norm clipping (--clip, default 1.0) to stop
                  the physics term's sharp gradients diverging mid-training
                  (full run thrashed from ~epoch 13 without it). Usage
                  docstring made escape-safe.
2026.06.13  v1.0  Initial PGML training. MLP from train_mlp.py; combined
                  data + lambda*physics loss; lambda by scale-matching on the
                  first batch (--lam to override; --lam 0 = MLP control).
                  R un-scaled for the physics term; Se used directly.
"""

import os
import sys
import time
import argparse

import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "03_Preprocessing"))
from preprocess import build_dataloaders, load_scalers, INPUT_DIM, OUTPUT_DIM

from physics_loss import physics_loss


# =============================================================================
# CONVEX-SWEEP NORMALISATION CONSTANTS
# Each loss term is divided by its untrained scale so both live in ~[0,1] before
# the convex weighting. Measured once on strategy2.h5 (seed 42, full train set)
# by measure_ranges.py; hardwired here so the sweep has no external file to lose.
# Re-measure and update these two for S3 (different dataset -> different scale).
# =============================================================================
DATA_MAX = 0.44751    # mean MSE, untrained network
PHYS_MAX = 0.048672   # mean Gardner residual, untrained network


# =============================================================================
# MODEL  (identical to train_mlp.py: Linear -> BatchNorm -> ReLU x depth, no
#         output activation -- soft-penalty approach, no clamp/sigmoid)
# =============================================================================
class MLP(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM,
                 width=512, depth=4):
        super().__init__()
        layers = [nn.Linear(input_dim, width), nn.BatchNorm1d(width), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.BatchNorm1d(width), nn.ReLU()]
        layers.append(nn.Linear(width, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =============================================================================
# PHYSICS-TERM PLUMBING
# Pull the pieces the Gardner residual needs out of one (B, 226) input batch,
# un-scale R to physical units, convert Se -> theta.
# Input layout (from preprocess.py): [ Se[:,t] (221) | R (1, scaled) | soil (4) ]
# =============================================================================
def unscale_R(R_scaled, scalers):
    lo = scalers["R_out"]["min"]
    hi = scalers["R_out"]["max"]
    return R_scaled * (hi - lo) + lo


def physics_term(x_batch, y_pred, scalers, phys, device):
    """
    x_batch : (B, 226) network input  (Se(t) physical | R scaled | soil scaled)
    y_pred  : (B, 221) network output = predicted Se(t+1), physical
    Returns the scalar physics loss (mean-squared Gardner residual).
    """
    theta_s = phys["theta_s"]
    theta_r = phys["theta_r"]
    dth     = theta_s - theta_r

    Se_t = x_batch[:, :OUTPUT_DIM]                      # (B,221) physical Se(t)
    R_scaled = x_batch[:, OUTPUT_DIM:OUTPUT_DIM + 1]    # (B,1) scaled
    R_phys = unscale_R(R_scaled, scalers)               # (B,1) physical

    theta_t    = theta_r + Se_t   * dth                 # Se -> theta
    theta_pred = theta_r + y_pred * dth

    return physics_loss(
        theta_t, theta_pred, R_phys,
        theta_s, theta_r,
        phys["Ks"], phys["m_g"], phys["alpha_g"],
        phys["Dz"], phys["Dt"],
    )


# =============================================================================
# TRAINING LOOP  (loss = (1-lam)*data_n + lam*phys_n, convex and normalised)
# =============================================================================
def run_epoch(model, loader, data_criterion, optimizer, device,
              scalers, phys, lam, clip, data_max, phys_max, train=True):
    model.train(train)
    context = torch.enable_grad() if train else torch.no_grad()
    tot, tot_d, tot_p, n = 0.0, 0.0, 0.0, 0

    with context:
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(x_batch)                                         # 1. MLP predicts Se(t+1)
            data_l = data_criterion(y_pred, y_batch)                        # 2. data loss: MSE vs true Se
            phys_l = physics_term(x_batch, y_pred, scalers, phys, device)   # 3. physics loss: Gardner residual
            data_n = data_l / data_max                                      # -> [0,1]
            phys_n = phys_l / phys_max                                      # -> [0,1]
            loss   = (1.0 - lam) * data_n + lam * phys_n                    # 4. Combined objective (convex, normalised)

            if train:
                optimizer.zero_grad()                                       # 5. clear old gradients
                loss.backward()                                             # 6. compute gradients (backprop!)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)    # 7. clip gradients (the stabiliser)
                optimizer.step()                                            # 8. adjust weights downhill

            tot   += loss.item()
            tot_d += data_l.item()
            tot_p += phys_l.item()
            n     += 1

    return tot / n, tot_d / n, tot_p / n


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    # --- reproducibility ---
    # A seed fixes weight init and shuffle order, so two runs that differ only
    # in one setting (e.g. batch size) can be compared without random-init noise
    # confounding the result. Omit --seed for a fresh random run.
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        print(f"Seed  : {args.seed} (deterministic init and shuffle)")

    # --- data ---
    print("\n" + "=" * 60)
    print("Building DataLoaders")
    print("=" * 60)
    train_loader, val_loader, _ = build_dataloaders(
        h5_path=args.h5, out_dir=args.out, batch_size=args.batch_size,
    )
    scalers = load_scalers(args.out)

    # convex-sweep normalisation constants are hardwired at module level.
    print(f"Ranges   : DATA_MAX={DATA_MAX}  PHYS_MAX={PHYS_MAX}  (hardwired, seed 42)")

    # --- physics constants for the Gardner residual ---
    # S2 fixes the soil, so these are file-level: Ks is the single value the
    # scaler stores (min == max), theta_s / theta_r are Cariaga nominal, and the
    # Gardner fit (m_g, alpha_g) comes from args. In S3 the soil varies per run,
    # so Ks, m_g, alpha_g and theta_s must be wired PER-RUN into the residual
    # (physics_loss already accepts (B,1) tensors for them). This dict does not
    # do that yet -- it is the S2 fixed-soil case only.
    phys = {
        "theta_s": 0.33,                 # Cariaga nominal (fixed in S2)
        "theta_r": 0.0,
        "Ks":      scalers["Ks"]["max"], # file-level physical Ks (min == max in S2)
        "m_g":     args.m_g,             # Gardner m  (Olivares fit)
        "alpha_g": args.alpha_g,         # Gardner alpha (Olivares fit)
        "Dz":      args.Dz,
        "Dt":      args.Dt,
    }
    print(f"\nPhysics  : Ks={phys['Ks']}  m_g={phys['m_g']}  alpha_g={phys['alpha_g']}"
          f"  Dz={phys['Dz']}  Dt={phys['Dt']:.5f}")

    # --- model ---
    model = MLP(INPUT_DIM, OUTPUT_DIM, width=args.width, depth=args.depth).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel    : MLP  width={args.width}  depth={args.depth}")
    print(f"Params   : {n_params:,}")

    data_criterion = nn.MSELoss()

    # --- lambda (convex, required, must be in [0,1]) ---
    # The old additive scale-match default (~9.2) is meaningless once the terms
    # are normalised, so --lam is required and guarded to [0,1].
    lam = args.lam
    if not (0.0 <= lam <= 1.0):
        sys.exit(f"ERROR: convex lambda must be in [0,1], got {lam}. "
                 f"(This is the normalised form, not the old additive scale.)")
    print(f"\nLambda   : fixed = {lam:.3e}" + ("  (MLP control)" if lam == 0 else ""))

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    os.makedirs(args.save, exist_ok=True)
    # One script trains both models; name the checkpoint by what was trained.
    # --tag (optional) overrides the auto-name so a lambda SWEEP keeps each run
    # in its own file (e.g. --tag lam100 -> pgml_lam100.pt) instead of every
    # lam>0 run overwriting pgml_best.pt. No --tag = original behaviour.
    if args.tag:
        prefix = "mlp" if lam == 0 else "pgml"
        ckpt_name = f"{prefix}_{args.tag}.pt"
    else:
        ckpt_name = "mlp_best.pt" if lam == 0 else "pgml_best.pt"
    ckpt_path = os.path.join(args.save, ckpt_name)
    print(f"\nCheckpoint will be saved as: {ckpt_name}  ({'MLP baseline' if lam == 0 else 'PGML'})")

    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)
    print(f"{'Epoch':>8}   {'Train':>14}   {'Data':>14}   {'Physics':>14}   "
          f"{'Val':>14}   {'LR':>11}   {'Time':>8}")
    print("-" * 104)

    best_val = float("inf")
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        tr, tr_d, tr_p = run_epoch(model, train_loader, data_criterion, optimizer,
                                   device, scalers, phys, lam, args.clip,
                                   DATA_MAX, PHYS_MAX, train=True)
        va, _, _ = run_epoch(model, val_loader, data_criterion, optimizer,
                             device, scalers, phys, lam, args.clip,
                             DATA_MAX, PHYS_MAX, train=False)
        scheduler.step(va)
        lr = optimizer.param_groups[0]["lr"]
        dt = time.perf_counter() - t0

        print(f"{epoch:>8}   {tr:>14.6f}   {tr_d:>14.6f}   {tr_p:>14.3e}   "
              f"{va:>14.6f}   {lr:>11.2e}   {dt:>7.1f}s")

        if va < best_val:
            best_val = va
            no_improve = 0
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "val_loss": best_val, "lambda": lam, "args": vars(args),
            }, ckpt_path)
            print(f"         checkpoint saved  (val={best_val:.6f})")
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(f"\nEarly stopping: val loss did not improve for {args.patience} epochs.")
            break

    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"Best val loss : {best_val:.6f}")
    print(f"Lambda used   : {lam:.3e}")
    print(f"Checkpoint    : {ckpt_path}")
    print("=" * 60)


def parse_args():
    p = argparse.ArgumentParser(description="PGML training -- PGML-MPC CENG0057")
    # paths
    p.add_argument("--h5",   required=True)
    p.add_argument("--out",  required=True)
    p.add_argument("--save", required=True)
    # architecture
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--depth", type=int, default=4)
    # training
    p.add_argument("--epochs",     type=int,   default=200)
    p.add_argument("--batch_size", type=int,   default=1024)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--seed",       type=int,   default=None,
                   help="Random seed for reproducible init/shuffle. "
                        "Omit for a fresh random run.")
    p.add_argument("--patience",   type=int,   default=20)
    # physics-loss controls
    p.add_argument("--lam", type=float, required=True,
                   help="Convex physics weight in [0,1] (REQUIRED). "
                        "0 = MLP control (data only); 1 = physics only.")
    p.add_argument("--clip", type=float, default=1.0,
                   help="Max gradient norm (clipping). Default 1.0; stabilises the physics term.")
    p.add_argument("--tag", type=str, default=None,
                   help="Optional checkpoint name tag for a lambda sweep "
                        "(e.g. --tag lam0p4 -> pgml_lam0p4.pt). "
                        "Omit for the default mlp_best.pt / pgml_best.pt.")
    p.add_argument("--m_g",     type=float, default=16.4094, help="Gardner m")
    p.add_argument("--alpha_g", type=float, default=4.0047,  help="Gardner alpha")
    p.add_argument("--Dz", type=float, default=2.5)
    p.add_argument("--Dt", type=float, default=1.0 / 24.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print("PGML Training  --  PGML-MPC CENG0057")
    print("=" * 60)
    print(f"h5       : {args.h5}")
    print(f"out      : {args.out}")
    print(f"save     : {args.save}")
    print(f"width    : {args.width}")
    print(f"depth    : {args.depth}")
    print(f"epochs   : {args.epochs}")
    print(f"batch    : {args.batch_size}")
    print(f"lr       : {args.lr}")
    print(f"seed     : {args.seed if args.seed is not None else '(random)'}")
    print(f"patience : {args.patience}")
    print(f"clip     : {args.clip}")
    print(f"tag      : {args.tag if args.tag else '(none)'}")
    print(f"lambda   : {args.lam}")
    train(args)
