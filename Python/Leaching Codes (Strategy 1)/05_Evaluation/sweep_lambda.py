"""
sweep_lambda.py
===============
Phase 4 - the LAMBDA SWEEP figure. Turns "lambda=9.2 because scale-matching"
into a justified choice: train PGML at several lambda, evaluate each on the
test set, and plot the trade-off so the defensible lambda is READ OFF the knee.

WORKFLOW (you drive the training; this script only evaluates + plots):
  1. Train each lambda point yourself with the patched train_pgml.py --tag:
        python train_pgml.py ... --lam 100   --tag lam100      -> pgml_lam100.pt
        python train_pgml.py ... --lam 1000  --tag lam1000     -> pgml_lam1000.pt
        python train_pgml.py ... --lam 10000 --tag lam10000    -> pgml_lam10000.pt
     (lambda=0 = your mlp_best.pt; lambda~9.2 = your pgml_best.pt -- reuse both.)
  2. Drop every checkpoint to include into ONE folder (the --sweep_dir).
  3. Run this script. It reads lambda FROM INSIDE each .pt (not the filename),
     evaluates each on the test set, and writes the sweep plot + JSON.

WHY READ LAMBDA FROM THE CHECKPOINT: train_pgml.py saves "lambda" in the .pt.
So filenames do not have to be perfect -- the script pulls the true lambda each
model trained with. mlp_best.pt carries lambda=0 automatically.

WHAT IT PLOTS (two curves vs lambda, log-x):
  - data error      vs lambda  -> rises as lambda grows (physics crowds out data)
  - physics residual vs lambda  -> falls as lambda grows (physics enforced harder)
  - data floor (flat line)      -> the per-dataset consistency level (from evaluate)
The defensible lambda is the KNEE: physics near its floor, data not yet blown up.

PER-STRATEGY: the knee (and the floor) depend on this h5's soil + Gardner fit.
NOT a universal constant -- recompute the sweep on Strategy 3 data.

Reuses evaluate.py verbatim (no physics or eval logic duplicated):
  load_ckpt, evaluate_model, phys_from_ckpt.

Usage (Windows; run from 05_Evaluation):
    python sweep_lambda.py ^
        --h5        "..\\..\\..\\Data generation\\Python dataset\\strategy1.h5" ^
        --out       "..\\03_Preprocessing" ^
        --sweep_dir ".\\sweep_ckpts"

Origin: Phase 4 lambda-sweep driver
Status: evaluate-only (training done by hand via train_pgml.py --tag)
Depends on: evaluate.py (load_ckpt, evaluate_model, phys_from_ckpt),
            preprocess.py (build_dataloaders, load_scalers),
            torch, numpy, matplotlib

Changelog
---------
2026.06.14  v1.0  Reads lambda from each checkpoint, evaluates every .pt in
                  --sweep_dir on the test set, plots data-error + physics-
                  residual vs lambda with the per-dataset data floor, saves
                  sweep_lambda.png + sweep_lambda.json. No physics duplicated.
"""

import os
import sys
import glob
import json
import argparse

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, "..", "03_Preprocessing"))
sys.path.append(os.path.join(_HERE, "..", "04_Training"))

from preprocess import build_dataloaders, load_scalers
# Reuse the validated evaluator pieces -- do NOT re-implement them here:
from evaluate import load_ckpt, evaluate_model, phys_from_ckpt


def main():
    p = argparse.ArgumentParser(description="PGML lambda sweep -- CENG0057")
    p.add_argument("--h5",        required=True)
    p.add_argument("--out",       required=True, help="03_Preprocessing (scaler/split)")
    p.add_argument("--sweep_dir", required=True, help="folder of .pt checkpoints to sweep")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--save", default=".", help="where to write the plot + JSON")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 64)
    print("PGML LAMBDA SWEEP  --  trade-off: data error vs physics residual")
    print("=" * 64)
    print(f"Device     : {device}")
    if device.type == "cuda":
        print(f"GPU        : {torch.cuda.get_device_name(0)}")

    # --- collect checkpoints ---
    ckpt_paths = sorted(glob.glob(os.path.join(args.sweep_dir, "*.pt")))
    if not ckpt_paths:
        print(f"\nNo .pt files found in {args.sweep_dir}. Nothing to sweep.")
        sys.exit(1)
    print(f"\nFound {len(ckpt_paths)} checkpoint(s) in {args.sweep_dir}:")
    for c in ckpt_paths:
        print(f"  {os.path.basename(c)}")

    # --- test set + scalers (once) ---
    _, _, test_loader = build_dataloaders(
        h5_path=args.h5, out_dir=args.out, batch_size=args.batch_size,
    )
    scalers = load_scalers(args.out)

    # --- evaluate each checkpoint; read lambda from inside the .pt ---
    points = []   # list of dicts: lambda, data_error, physics_residual, name
    phys_for_floor = None
    print("\nEvaluating ...")
    for path in ckpt_paths:
        model, ckpt = load_ckpt(path, device)
        lam = ckpt.get("lambda", 0.0)
        phys = phys_from_ckpt(ckpt, scalers)
        if phys_for_floor is None:
            phys_for_floor = phys          # Gardner fit is shared; use any model's
        d, pr, _ = evaluate_model(model, test_loader, scalers, phys, device)
        points.append({
            "name": os.path.basename(path),
            "lambda": float(lam),
            "data_error": float(d),
            "physics_residual": float(pr),
        })
        print(f"  {os.path.basename(path):<22} lam={lam:>10.4g}  "
              f"data={d:.4e}  physics={pr:.4e}")

    # --- data floor (truth graded), same metric, once ---
    print("Computing data's own floor (truth graded) ...")
    _, floor_p, _ = evaluate_model(
        None, test_loader, scalers, phys_for_floor, device, use_truth=True)
    print(f"  data floor (physics residual on truth): {floor_p:.4e}")

    # --- sort points by lambda for a clean curve ---
    points.sort(key=lambda r: r["lambda"])
    lams = np.array([r["lambda"] for r in points])
    derr = np.array([r["data_error"] for r in points])
    pres = np.array([r["physics_residual"] for r in points])

    # --- table ---
    print("\n" + "=" * 64)
    print("SWEEP RESULTS  (sorted by lambda)")
    print("=" * 64)
    print(f"{'lambda':>12}{'data error':>18}{'physics residual':>20}")
    print("-" * 64)
    for r in points:
        print(f"{r['lambda']:>12.4g}{r['data_error']:>18.4e}{r['physics_residual']:>20.4e}")
    print("-" * 64)
    print(f"{'data floor':>12}{'(truth)':>18}{floor_p:>20.4e}")
    print("\nDefensible lambda = the KNEE: physics near the floor, data not yet")
    print("blown up. Read it off the plot. (Per-dataset -- redo for Strategy 3.)")

    # --- plot: two y-curves vs lambda (log-x), data floor as a flat line ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # lambda=0 cannot sit on a log axis; place it at a small sentinel and
        # label it honestly so the curve still starts from the MLP control.
        lam_plot = lams.copy()
        zero_mask = lam_plot == 0
        if zero_mask.any():
            nonzero = lam_plot[lam_plot > 0]
            sentinel = (nonzero.min() / 10.0) if nonzero.size else 1.0
            lam_plot[zero_mask] = sentinel

        fig, ax1 = plt.subplots(figsize=(7, 5))
        ax1.set_xscale("log")
        ax1.set_xlabel("lambda  (physics weight)  [0 shown at left sentinel]")

        c_data = "#5b8dee"
        c_phys = "#f0a255"
        ax1.set_ylabel("data error (MSE)", color=c_data)
        ax1.plot(lam_plot, derr, "o-", color=c_data, label="data error")
        ax1.tick_params(axis="y", labelcolor=c_data)
        ax1.set_yscale("log")

        ax2 = ax1.twinx()
        ax2.set_ylabel("physics residual", color=c_phys)
        ax2.plot(lam_plot, pres, "s-", color=c_phys, label="physics residual")
        ax2.axhline(floor_p, color="#3ecf8e", ls="--", lw=1.0,
                    label="data floor")
        ax2.tick_params(axis="y", labelcolor=c_phys)
        ax2.set_yscale("log")

        # annotate the lambda=0 point
        if zero_mask.any():
            ax1.annotate("lambda=0\n(MLP)", xy=(sentinel, derr[zero_mask][0]),
                         fontsize=8, color=c_data,
                         xytext=(0, 8), textcoords="offset points", ha="center")

        lines1, labs1 = ax1.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="best")
        ax1.set_title("Lambda sweep -- physics weight vs data/physics trade-off (Strategy 1)")
        ax1.grid(True, which="both", alpha=0.2)

        png = os.path.join(args.save, "sweep_lambda.png")
        fig.tight_layout()
        fig.savefig(png, dpi=150)
        print(f"\nSaved sweep plot : {png}")
    except ImportError:
        png = None
        print("\n[matplotlib not available - skipped plot; JSON still written]")

    # --- JSON ---
    out = {
        "test_set": {"h5": args.h5, "split_dir": args.out},
        "sweep_dir": args.sweep_dir,
        "data_floor": {"physics_residual": floor_p,
                       "note": "mean(res^2) on VGM truth; per-dataset, not universal"},
        "points": points,
        "gardner_params": {k: phys_for_floor[k] for k in ("m_g", "alpha_g", "Dz", "Dt")},
    }
    json_path = os.path.join(args.save, "sweep_lambda.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved raw numbers: {json_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
