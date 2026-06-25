"""
train_pgml.py
=============
The single training script for BOTH surrogate models. The loss is always:

    total = data_loss + lambda * physics_loss

  data_loss    : MSE( predicted Se(t+1), true Se(t+1) ) against the VGM truth
  physics_loss : mean-squared Gardner residual on the prediction (physics_loss.py)
  lambda       : weight on the physics term (soft penalty)

lambda selects which model is trained, using identical machinery (same MLP,
clipping, scheduler, data) so the only difference is the physics term -- a fair
PGML-vs-baseline comparison:
    lambda  > 0  ->  PGML         (data + Gardner physics)   -> saves pgml_best.pt
    lambda == 0  ->  MLP baseline (data only, physics off)   -> saves mlp_best.pt

This supersedes the old standalone train_mlp.py (now redundant: --lam 0 here is
the fair MLP control, sharing this script's clipping/scheduler).

SPACE NOTE (important):
  In strategy1.h5 Se is already physical [0,1] (NOT scaled) -- only R and the
  soil params are min-max scaled. So:
    - data_loss runs directly on Se (no un-scaling),
    - physics_loss needs R in PHYSICAL units, so R is un-scaled before the
      residual sees it,
    - Se -> theta via theta = theta_r + Se*(theta_s - theta_r).

LAMBDA:
  Default lambda is set by SCALE-MATCHING on the first batch: measure data_loss
  and raw physics_loss, set lambda = data_loss / physics_loss so the two terms
  start comparable. Pass --lam to override (e.g. for a sweep). --lam 0 reduces
  this to the plain MLP baseline (useful as a control).

Usage (Windows; backslash paths, ^ line continuation):
    python train_pgml.py ^
        --h5   "Data generation\\Python dataset\\strategy1.h5" ^
        --out  "Leaching Codes\\03_Preprocessing" ^
        --save "Leaching Codes\\04_Training"

Origin: PGML variant of train_mlp.py
Status: active development
Depends on: preprocess.py (build_dataloaders, load_scalers), physics_loss.py, torch

Changelog
---------
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
import json

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "03_Preprocessing"))
from preprocess import build_dataloaders, load_scalers, INPUT_DIM, OUTPUT_DIM

from physics_loss import physics_loss


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

    Se_t = x_batch[:, :OUTPUT_DIM]                 # (B,221) physical Se(t)
    R_scaled = x_batch[:, OUTPUT_DIM:OUTPUT_DIM + 1]  # (B,1) scaled
    R_phys = unscale_R(R_scaled, scalers)          # (B,1) physical

    theta_t    = theta_r + Se_t   * dth            # Se -> theta
    theta_pred = theta_r + y_pred * dth

    return physics_loss(
        theta_t, theta_pred, R_phys,
        theta_s, theta_r,
        phys["Ks"], phys["m_g"], phys["alpha_g"],
        phys["Dz"], phys["Dt"],
    )


# =============================================================================
# TRAINING LOOP  (mirrors train_mlp.run_epoch, loss = data + lam*physics)
# =============================================================================
def run_epoch(model, loader, data_criterion, optimizer, device,
              scalers, phys, lam, clip, train=True):
    model.train(train)
    context = torch.enable_grad() if train else torch.no_grad()
    tot, tot_d, tot_p, n = 0.0, 0.0, 0.0, 0

    with context:
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(x_batch)
            data_l = data_criterion(y_pred, y_batch)
            phys_l = physics_term(x_batch, y_pred, scalers, phys, device)
            loss   = data_l + lam * phys_l

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

            tot   += loss.item()
            tot_d += data_l.item()
            tot_p += phys_l.item()
            n     += 1

    return tot / n, tot_d / n, tot_p / n


def measure_lambda(model, loader, data_criterion, device, scalers, phys):
    """Scale-match lambda on the first batch: lam = data_loss / physics_loss."""
    model.eval()
    with torch.no_grad():
        x_batch, y_batch = next(iter(loader))
        x_batch = x_batch.to(device); y_batch = y_batch.to(device)
        y_pred = model(x_batch)
        d = data_criterion(y_pred, y_batch).item()
        p = physics_term(x_batch, y_pred, scalers, phys, device).item()
    if p <= 0:
        return 1.0, d, p
    return d / p, d, p


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    # --- data ---
    print("\n" + "=" * 60)
    print("Building DataLoaders")
    print("=" * 60)
    train_loader, val_loader, _ = build_dataloaders(
        h5_path=args.h5, out_dir=args.out, batch_size=args.batch_size,
    )
    scalers = load_scalers(args.out)

    # --- physics constants (fixed for Strategy 1; per-run in Strategy 3) ---
    # Soil values are the physical numbers (scaler stores min==max==value).
    phys = {
        "theta_s": scalers["Ks"].get("theta_s", 0.33) if "theta_s" in scalers.get("Ks", {}) else 0.33,
        "theta_r": 0.0,
        "Ks":      scalers["Ks"]["max"],     # physical Ks (e.g. 170.0)
        "m_g":     args.m_g,                 # Gardner m  (Olivares fit)
        "alpha_g": args.alpha_g,             # Gardner alpha (Olivares fit)
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

    # --- lambda: scale-match unless given ---
    if args.lam is None:
        lam, d0, p0 = measure_lambda(model, train_loader, data_criterion, device, scalers, phys)
        print(f"\nLambda   : scale-matched = {lam:.3e}  (first-batch data={d0:.3e} physics={p0:.3e})")
    else:
        lam = args.lam
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
                                   device, scalers, phys, lam, args.clip, train=True)
        va, _, _ = run_epoch(model, val_loader, data_criterion, optimizer,
                             device, scalers, phys, lam, args.clip, train=False)
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
    p.add_argument("--batch_size", type=int,   default=256)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--patience",   type=int,   default=20)
    # physics-loss controls
    p.add_argument("--lam", type=float, default=None,
                   help="Physics weight. Default: scale-matched. 0 = MLP control.")
    p.add_argument("--clip", type=float, default=1.0,
                   help="Max gradient norm (clipping). Default 1.0; stabilises the physics term.")
    p.add_argument("--tag", type=str, default=None,
                   help="Optional checkpoint name tag for a lambda sweep "
                        "(e.g. --tag lam100 -> pgml_lam100.pt). "
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
    print(f"patience : {args.patience}")
    print(f"clip     : {args.clip}")
    print(f"tag      : {args.tag if args.tag else '(none)'}")
    print(f"lambda   : {'scale-matched' if args.lam is None else args.lam}")
    train(args)
