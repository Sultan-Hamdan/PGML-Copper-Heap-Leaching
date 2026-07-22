"""
train_mlp.py
============
Baseline MLP surrogate training script.
Trains a vanilla multi-layer perceptron on strategy1.h5 (or any compatible h5)
to predict Se(z, t+1) from Se(z, t), R(t), and soil parameters.

Architecture:
    Input  (226)  ->  4 x [Linear -> BatchNorm -> ReLU] (width 512)  ->  Output (221)
    No output activation. No skip connections.

Loss:
    MSE only.
    
Usage:
    python train_mlp.py ^
        --h5   "Data generation\Python dataset\strategy1.h5" ^
        --out  "Leaching Codes\03_Preprocessing" ^
        --save "Leaching Codes\04_Training"

All paths use Windows backslash convention. Caret (^) is Windows line continuation.

Origin: original
Status: active development
Depends on: preprocess.py (build_dataloaders), torch

Changelog
---------
2026.06.10  v1.0  Initial MLP baseline, Strategy 1, architecture 226->512x4->221
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

# preprocess.py lives one folder up in 03_Preprocessing/
# Add its directory to the path so we can import build_dataloaders.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "03_Preprocessing"))
from preprocess import build_dataloaders, INPUT_DIM, OUTPUT_DIM


# =============================================================================
# MODEL
# Four hidden layers, each: Linear -> BatchNorm -> ReLU.
# Output layer: Linear only, no activation.
# Width and depth are arguments so we can experiment without editing code.
# =============================================================================

class MLP(nn.Module):
    """
    Vanilla MLP surrogate.

    Parameters
    ----------
    input_dim  : number of input features (226)
    output_dim : number of output values (221)
    width      : neurons per hidden layer (default 512)
    depth      : number of hidden layers (default 4)
    """

    def __init__(
        self,
        input_dim:  int = INPUT_DIM,
        output_dim: int = OUTPUT_DIM,
        width:      int = 512,
        depth:      int = 4,
    ):
        super().__init__()

        layers = []

        # --- input -> first hidden layer ---
        layers += [
            nn.Linear(input_dim, width),
            nn.BatchNorm1d(width),
            nn.ReLU(),
        ]

        # --- hidden -> hidden layers (depth - 1 remaining) ---
        for _ in range(depth - 1):
            layers += [
                nn.Linear(width, width),
                nn.BatchNorm1d(width),
                nn.ReLU(),
            ]

        # --- last hidden -> output, no activation ---
        layers.append(nn.Linear(width, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =============================================================================
# TRAINING LOOP
# One epoch = one full pass over the training set.
# Val loss is computed after every epoch.
# Best model (lowest val loss) is saved as a checkpoint.
# Early stopping halts training if val loss has not improved for patience epochs.
# =============================================================================

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    """
    Run one epoch over loader.
    If train=True: compute gradients and update weights.
    If train=False: no gradients (eval mode).

    Returns mean loss over all batches.
    """
    model.train(train)
    context = torch.enable_grad() if train else torch.no_grad()
    total_loss = 0.0
    n_batches  = 0

    with context:
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(x_batch)
            loss   = criterion(y_pred, y_batch)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

    return total_loss / n_batches


def train(args):

    # --- device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    # --- data ---
    print("\n" + "=" * 60)
    print("Building DataLoaders")
    print("=" * 60)
    train_loader, val_loader, _ = build_dataloaders(
        h5_path    = args.h5,
        out_dir    = args.out,
        batch_size = args.batch_size,
    )

    # --- model ---
    model = MLP(
        input_dim  = INPUT_DIM,
        output_dim = OUTPUT_DIM,
        width      = args.width,
        depth      = args.depth,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel    : MLP  width={args.width}  depth={args.depth}")
    print(f"Params   : {n_params:,}")

    # --- loss, optimiser, scheduler ---
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode     = "min",   # minimise val loss
        factor   = 0.5,     # halve lr on plateau
        patience = 10,      # wait 10 epochs before reducing
    )

    # --- checkpoint path ---
    os.makedirs(args.save, exist_ok=True)
    ckpt_path = os.path.join(args.save, "mlp_best.pt")

    # --- training loop ---
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)
    print(f"{'Epoch':>6}  {'Train loss':>12}  {'Val loss':>12}  {'LR':>10}  {'Time':>7}")
    print("-" * 60)

    best_val_loss    = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss   = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed    = time.perf_counter() - t0

        print(f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}  "
              f"{current_lr:>10.2e}  {elapsed:>6.1f}s")

        # --- checkpoint: save if best val loss ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "val_loss":   best_val_loss,
                "args":       vars(args),
            }, ckpt_path)
            print(f"         checkpoint saved  (val_loss={best_val_loss:.6f})")
        else:
            epochs_no_improve += 1

        # --- early stopping ---
        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping: val loss did not improve for {args.patience} epochs.")
            break

    print("\n" + "=" * 60)
    print(f"Training complete.")
    print(f"Best val loss : {best_val_loss:.6f}")
    print(f"Checkpoint    : {ckpt_path}")
    print("=" * 60)


# =============================================================================
# ARGUMENT PARSER
# All paths and hyperparameters are CLI arguments.
# Defaults reproduce the agreed baseline configuration.
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="MLP baseline training — PGML-MPC")

    # paths
    p.add_argument("--h5",   required=True,
                   help="Path to strategy1.h5 (or strategy3.h5 when ready)")
    p.add_argument("--out",  required=True,
                   help="Path to 03_Preprocessing/ (scaler.json, split.json)")
    p.add_argument("--save", required=True,
                   help="Path to 04_Training/ (checkpoint written here)")

    # architecture
    p.add_argument("--width", type=int, default=512,
                   help="Neurons per hidden layer (default 512)")
    p.add_argument("--depth", type=int, default=4,
                   help="Number of hidden layers (default 4)")

    # training
    p.add_argument("--epochs",     type=int,   default=200)
    p.add_argument("--batch_size", type=int,   default=256)
    p.add_argument("--lr",         type=float, default=1e-3,
                   help="Initial learning rate (default 1e-3)")
    p.add_argument("--patience",   type=int,   default=20,
                   help="Early stopping patience in epochs (default 20)")

    return p.parse_args()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("MLP Baseline Training  --  PGML-MPC CENG0057")
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

    train(args)
