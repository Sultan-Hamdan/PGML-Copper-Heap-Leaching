"""
measure_ranges.py
=================
Provenance tool: regenerates the two normalisation constants (DATA_MAX,
PHYS_MAX) that are HARDWIRED in train_pgml.py. It reports the scale each loss
term starts at on the UNTRAINED network. Run this only when the dataset changes
(e.g. S3); copy the printed DATA_MAX/PHYS_MAX into train_pgml.py's constants.

What it produces
    DATA_MAX  : mean MSE over the training set, untrained network
    PHYS_MAX  : mean Gardner residual over the training set, untrained network
    (both mins are 0 by construction; a perfect model reaches 0 on each term)

Why untrained
    Normalising each term by its value on the fresh network is the GradNorm
    L_i(0) normaliser (Chen et al., ICML 2018): term_norm = term / term(0).
    Here term(0) is measured once, held fixed, and reused across the whole
    convex sweep. The weighting stays a fixed lambda you sweep and pick; only
    the per-term scale is borrowed from GradNorm.

What it does NOT do
    No training. No weight updates. It builds the model (random weights, seed
    fixed), runs the training set through ONCE under torch.no_grad(), and reads
    the two term values off that single forward pass. loss.backward() and
    optimizer.step() are simply never called, so the weights cannot move.

Determinism
    --seed fixes the weight init so the two maxes are reproducible. eval() mode
    is used (matching measure_lambda in train_pgml) so BatchNorm uses fixed
    stats and each prediction is independent of batch grouping; the maxes are
    therefore batch-size independent.

Physics wiring is imported from train_pgml (MLP, physics_term) so this script
measures the term the exact way the trainer computes it. No re-typed maths, no
drift.

Run from 04_Training (same working directory as train_pgml.py):
    python measure_ranges.py ^
        --h5   "Data generation\\Python dataset\\strategy2.h5" ^
        --out  "Leaching Codes\\03_Preprocessing" ^
        --save-json "04_Training\\ranges.json"   (optional record; not auto-read)

Origin: support script for the convex lambda sweep (Action 1)
Status: active development
Depends on: preprocess.py (build_dataloaders, load_scalers, INPUT_DIM,
            OUTPUT_DIM), train_pgml.py (MLP, physics_term), torch

Changelog
---------
2026.07.22  v1.0  Initial. Measures DATA_MAX and PHYS_MAX on the untrained
                  network over the full training set (seed fixed, eval mode, no
                  weight updates). Mins are 0 by construction. Optional
                  --save-json writes the four range numbers for the sweep to
                  read. Physics term imported from train_pgml to stay identical
                  to the trainer.
"""

import os
import sys
import json
import argparse

import torch
import torch.nn as nn
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "03_Preprocessing"))
from preprocess import build_dataloaders, load_scalers, INPUT_DIM, OUTPUT_DIM

# Import the model and the physics-term plumbing straight from the trainer so
# the measured physics residual is computed identically to training.
from train_pgml import MLP, physics_term


def build_phys(scalers, args):
    """
    Rebuild the same physics-constants dict the trainer uses (train_pgml.train).
    S2 fixed-soil case only, matching the trainer's own note.
    """
    return {
        "theta_s": 0.33,                  # Cariaga nominal (fixed in S2)
        "theta_r": 0.0,
        "Ks":      scalers["Ks"]["max"],  # file-level physical Ks (min == max in S2)
        "m_g":     args.m_g,              # Gardner m  (Olivares fit)
        "alpha_g": args.alpha_g,          # Gardner alpha (Olivares fit)
        "Dz":      args.Dz,
        "Dt":      args.Dt,
    }


def measure(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("measure_ranges  --  convex sweep term scales (Action 1)")
    print("=" * 60)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    # --- reproducibility: fix the init so the two maxes are repeatable ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    print(f"Seed  : {args.seed} (deterministic weight init)")

    # --- data: we only need the TRAIN loader ---
    train_loader, _, _ = build_dataloaders(
        h5_path=args.h5, out_dir=args.out, batch_size=args.batch_size,
    )
    scalers = load_scalers(args.out)
    phys = build_phys(scalers, args)
    print(f"\nPhysics  : Ks={phys['Ks']}  m_g={phys['m_g']}  "
          f"alpha_g={phys['alpha_g']}  Dz={phys['Dz']}  Dt={phys['Dt']:.5f}")

    # --- untrained model, eval mode (fixed BN stats, no dropout) ---
    model = MLP(INPUT_DIM, OUTPUT_DIM, width=args.width, depth=args.depth).to(device)
    model.eval()
    data_criterion = nn.MSELoss()

    # --- one forward pass over the whole training set, no grad, no updates ---
    # Sample-weighted mean so the number is the true mean over the training set
    # (a smaller final batch does not skew it).
    sum_d, sum_p, n_samples = 0.0, 0.0, 0
    with torch.no_grad():
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(x_batch)
            d = data_criterion(y_pred, y_batch).item()
            p = physics_term(x_batch, y_pred, scalers, phys, device).item()
            bs = x_batch.shape[0]
            sum_d += d * bs
            sum_p += p * bs
            n_samples += bs

    data_max = sum_d / n_samples
    phys_max = sum_p / n_samples

    # --- report ---
    print("\n" + "=" * 60)
    print("RANGES  (min = 0 for both, by construction)")
    print("=" * 60)
    print(f"  samples measured : {n_samples}")
    print(f"  DATA_MIN         : 0.0")
    print(f"  DATA_MAX         : {data_max:.6e}   (mean MSE, untrained)")
    print(f"  PHYS_MIN         : 0.0")
    print(f"  PHYS_MAX         : {phys_max:.6e}   (mean Gardner residual, untrained)")
    print("=" * 60)
    print("Use in the sweep:")
    print("  data_norm = data_l / DATA_MAX")
    print("  phys_norm = phys_l / PHYS_MAX")
    print("  loss      = (1 - lambda) * data_norm + lambda * phys_norm")

    # --- optional: write the four numbers for the sweep to read ---
    if args.save_json:
        out = {
            "data_min": 0.0,
            "data_max": data_max,
            "phys_min": 0.0,
            "phys_max": phys_max,
            "n_samples": n_samples,
            "seed": args.seed,
            "note": "untrained-network term scales; GradNorm L_i(0) normaliser; "
                    "mins are 0 by construction. Per-dataset -- redo for S3.",
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)), exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved ranges: {args.save_json}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Measure untrained term scales for the convex lambda sweep")
    # paths
    p.add_argument("--h5",   required=True)
    p.add_argument("--out",  required=True)
    p.add_argument("--save-json", type=str, default=None,
                   help="Optional path to write ranges.json for the sweep to read.")
    # architecture (must match the trainer's model)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--depth", type=int, default=4)
    # measurement
    p.add_argument("--batch_size", type=int, default=1024,
                   help="Batch size for the forward pass. The mean is batch-size "
                        "independent in eval mode; default matches training.")
    p.add_argument("--seed", type=int, default=42,
                   help="Fixes weight init so the maxes are reproducible.")
    # physics constants (must match the trainer)
    p.add_argument("--m_g",     type=float, default=16.4094, help="Gardner m")
    p.add_argument("--alpha_g", type=float, default=4.0047,  help="Gardner alpha")
    p.add_argument("--Dz", type=float, default=2.5)
    p.add_argument("--Dt", type=float, default=1.0 / 24.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    measure(args)
