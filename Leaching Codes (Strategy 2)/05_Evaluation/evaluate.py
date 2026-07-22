"""
evaluate.py
===========
Phase 4 - EVALUATION (05_Evaluation). The real thesis evidence.

Runs BOTH trained surrogates (mlp_best.pt, pgml_best.pt) on the held-out TEST set and reports TWO axes,
the comparison val loss alone cannot show:
    data error      = MSE( y_pred, VGM truth )        <- what val loss sees
    physics residual = mean( Gardner residual^2 )      <- what val loss misses

Why two axes (the whole point): the MLP fits data tighter, but it is never constrained to obey the physics.
The physics residual is the axis where PGML is expected to win.
The 2x2 (model x axis) is the thesis claim made measurable.

THREE OUTPUTS (one computation, saved at three levels of detail):
  1. 2x2 TABLE  -> printed to terminal (the headline; loud and first)
  2. PER-DEPTH PROFILE -> mean squared residual per node (221), saved as a PNG
        (shows WHERE each model violates physics - expected worst at the top surface-flux node and across a sharp wetting front)
  3. JSON  -> all raw numbers + metadata, for the lambda-sweep to append to and
        for reproducible figures without re-running the models.

GROUNDING (nothing physical re-derived here):
  - physics_loss / physics_residual_gardner_torch : VERIFIED (matches the NumPy reference to ~5e-17;
    the NumPy reference was verified on real run_0000 data).
  - test_loader : the 3rd return of build_dataloaders (training discards it);
    same run-level split, same scaling, same (x,y) layout the physics term uses.
  - Gardner params (m_g, alpha_g, Dz, Dt) are read from EACH checkpoint's saved
    'args' so every model is scored with the EXACT physics it trained with.

FAIR-COMPARISON NOTE: both models are scored with the identical physics_lossare 
call. The MLP saw no physics in training; we still measure its Gardner residual
the same way. That is the apples-to-apples comparison.

READING THE NUMBERS (the data's own floor, computed here, not hardcoded):
  The script grades the VGM GROUND-TRUTH next-step profile with the SAME
  mean(res^2) metric. That gives the data's own Gardner-residual floor -- the
  consistency limit set by (a) the simulator's 20-iteration non-convergence and
  (b) the VGM<->Gardner structural mismatch. A model at/below this floor is at
  the data's noise level; the meaningful comparison is the gap BETWEEN models.
  IMPORTANT: this floor is PER-DATASET / PER-STRATEGY (depends on the soil in
  this h5 and the Gardner fit). It is NOT a universal constant -- it is
  recomputed from whatever --h5 is passed, so Strategy 3 gets its own floor.

Usage (Windows; run from 05_Evaluation, or pass --models to point elsewhere):
    python evaluate.py ^
        --h5     "..\\..\\Data generation\\Python dataset\\strategy1.h5" ^
        --out    "..\\03_Preprocessing" ^
        --models "..\\04_Training"

Origin: Phase 4 evaluator (2x2 + per-depth + JSON)
Status: 2x2 stage (lambda sweep is a separate driver, built after sign-off)
Depends on: preprocess.py (build_dataloaders, load_scalers, INPUT_DIM, OUTPUT_DIM),
            physics_loss.py (physics_loss, physics_residual_gardner_torch),
            train_pgml.py (MLP class, unscale_R - imported, not duplicated),
            torch, numpy, matplotlib

Changelog
---------
2026.06.15  v1.2  Added --mlp and --pgml args so checkpoints are named
                  explicitly (no hardcoded filenames). Confirmed pair:
                  the checkpoints in --models (e.g. 06_Control/pgml_lam0p4.pt).
2026.06.14  v1.1  Removed misleading hardcoded 4.2e-3 line. Replaced with
                  data's own Gardner-residual floor: same mean(res^2) metric
                  run on VGM truth (use_truth flag). Floor is per-dataset,
                  plotted as a profile and stored in JSON.
2026.06.14  v1.0  2x2 table + per-depth residual profile (PNG) + JSON. Reuses
                  verified physics_loss verbatim; Gardner params read per-model
                  from each checkpoint's saved args. test_loader from
                  build_dataloaders. No physics re-derived.
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn as nn

# --- locate sibling phase folders so imports resolve from 05_Evaluation -------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, "..", "03_Preprocessing"))
sys.path.append(os.path.join(_HERE, "..", "04_Training"))

from preprocess import build_dataloaders, load_scalers, INPUT_DIM, OUTPUT_DIM
from physics_loss import physics_loss, physics_residual_gardner_torch
# Reuse the EXACT model + helpers PGML trained with (do not re-define them):
from train_pgml import MLP, unscale_R


# =============================================================================
# Build the physics-constants dict for ONE model, from its saved checkpoint args
# (mirrors train_pgml.physics_term's 'phys', but per-checkpoint so each model is
#  scored with the physics it actually trained with).
# =============================================================================
def phys_from_ckpt(ckpt, scalers):
    a = ckpt["args"]
    return {
        "theta_s": 0.33,            # Strategy 1 fixed soil (theta_r = 0)
        "theta_r": 0.0,
        "Ks":      scalers["Ks"]["max"],     # physical Ks (e.g. 170.0)
        "m_g":     a["m_g"],
        "alpha_g": a["alpha_g"],
        "Dz":      a["Dz"],
        "Dt":      a["Dt"],
    }


# =============================================================================
# Run one model over the test set. Returns:
#   data_mse       : scalar  (mean over all samples + nodes)
#   phys_scalar    : scalar  (mean squared Gardner residual)
#   phys_profile   : (221,)  mean squared residual PER NODE (the depth profile)
# One pass; the scalar is just the profile averaged over nodes - same compute.
#
# use_truth=True  -> grade the GROUND-TRUTH next-step profile (y_batch) instead
#   of the model's prediction. Same metric, same axis: this is the DATA's OWN
#   Gardner-residual floor. It is non-zero for two stacked reasons -- (1) the
#   simulator runs 20 fixed-point iterations, not to convergence (~1e-5 data
#   floor), and (2) VGM truth graded by a Gardner residual carries the
#   structural VGM<->Gardner mismatch. A model dropping BELOW this floor is
#   fitting discretisation noise, not "more physical".
#   NOTE: this floor is PER-DATASET / PER-STRATEGY (depends on this h5's soil +
#   the Gardner fit). It is NOT a universal constant -- recompute on Strategy 3.
# =============================================================================
@torch.no_grad()
def evaluate_model(model, loader, scalers, phys, device, use_truth=False):
    theta_s, theta_r = phys["theta_s"], phys["theta_r"]
    dth = theta_s - theta_r

    data_se = 0.0            # sum of squared data errors
    data_n = 0              # count of data elements
    res_sq_sum = None       # (221,) running sum of squared residual per node
    res_n = 0              # count of samples (rows)

    if model is not None:
        model.eval()
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        if use_truth:
            y_pred = y_batch            # grade the truth -> data's own floor
        else:
            y_pred = model(x_batch)

        # --- data error (same space as training: Se is physical) ---
        # (zero by construction when use_truth; kept for a single code path)
        diff = y_pred - y_batch
        data_se += torch.sum(diff ** 2).item()
        data_n += diff.numel()

        # --- physics residual (reuse the verified machinery) ---
        Se_t = x_batch[:, :OUTPUT_DIM]
        R_scaled = x_batch[:, OUTPUT_DIM:OUTPUT_DIM + 1]
        R_phys = unscale_R(R_scaled, scalers)

        theta_t = theta_r + Se_t * dth
        theta_pred = theta_r + y_pred * dth

        res = physics_residual_gardner_torch(
            theta_t, theta_pred, R_phys,
            theta_s, theta_r,
            phys["Ks"], phys["m_g"], phys["alpha_g"],
            phys["Dz"], phys["Dt"],
        )                                   # (B, 221) per-node residual
        rsq = res ** 2
        batch_sum = torch.sum(rsq, dim=0)   # (221,) sum over samples in batch
        res_sq_sum = batch_sum if res_sq_sum is None else res_sq_sum + batch_sum
        res_n += rsq.shape[0]

    data_mse = data_se / data_n
    phys_profile = (res_sq_sum / res_n).cpu().numpy()    # (221,) MSR per node
    phys_scalar = float(phys_profile.mean())
    return data_mse, phys_scalar, phys_profile


def load_ckpt(path, device):
    # weights_only=False: the checkpoint stores the 'args' dict, not just tensors.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MLP(INPUT_DIM, OUTPUT_DIM,
                width=ckpt["args"]["width"], depth=ckpt["args"]["depth"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt


def main():
    p = argparse.ArgumentParser(description="PGML evaluation 2x2 -- CENG0057")
    p.add_argument("--h5",     required=True)
    p.add_argument("--out",    required=True, help="03_Preprocessing (scaler/split)")
    p.add_argument("--mlp",    required=True, help="path to MLP checkpoint (.pt)")
    p.add_argument("--pgml",   required=True, help="path to PGML checkpoint (.pt)")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--save", default=".", help="where to write profile PNG + JSON")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 64)
    print("PGML EVALUATION  --  2x2  (data error  x  physics residual)")
    print("=" * 64)
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # --- test set (the 3rd loader; training discards it) ---
    _, _, test_loader = build_dataloaders(
        h5_path=args.h5, out_dir=args.out, batch_size=args.batch_size,
    )
    scalers = load_scalers(args.out)

    # --- load both models ---
    mlp_path  = args.mlp
    pgml_path = args.pgml
    mlp,  mlp_ck  = load_ckpt(mlp_path,  device)
    pgml, pgml_ck = load_ckpt(pgml_path, device)

    # each scored with the physics IT trained with
    mlp_phys  = phys_from_ckpt(mlp_ck,  scalers)
    pgml_phys = phys_from_ckpt(pgml_ck, scalers)
    print(f"\nGardner physics (from checkpoints): "
          f"m_g={pgml_phys['m_g']}  alpha_g={pgml_phys['alpha_g']}  "
          f"Dz={pgml_phys['Dz']}  Dt={pgml_phys['Dt']:.5f}")

    # --- evaluate ---
    print("\nRunning test set ...")
    mlp_d,  mlp_p,  mlp_prof  = evaluate_model(mlp,  test_loader, scalers, mlp_phys,  device)
    pgml_d, pgml_p, pgml_prof = evaluate_model(pgml, test_loader, scalers, pgml_phys, device)

    # --- DATA'S OWN floor: same metric on the ground-truth next-step profile ---
    # Uses the Gardner fit (shared); model arg is None (use_truth ignores it).
    # This floor is PER-DATASET (this h5's soil + this Gardner fit) -- NOT a
    # universal constant. Recompute it whenever the dataset/strategy changes.
    print("Computing data's own Gardner-residual floor (truth graded) ...")
    _, floor_p, floor_prof = evaluate_model(
        None, test_loader, scalers, pgml_phys, device, use_truth=True)

    # ------------------------------------------------------------------ TABLE
    print("\n" + "=" * 64)
    print("2x2 RESULTS  (test set)")
    print("=" * 64)
    print(f"{'Model':<16}{'data error (MSE)':>22}{'physics residual':>22}")
    print("-" * 64)
    print(f"{'MLP (lam=0)':<16}{mlp_d:>22.6e}{mlp_p:>22.6e}")
    print(f"{'PGML':<16}{pgml_d:>22.6e}{pgml_p:>22.6e}")
    print("-" * 64)
    print(f"{'DATA floor':<16}{'(truth)':>22}{floor_p:>22.6e}")
    print("-" * 64)
    print("Floor = the same mean(res^2) metric on the VGM TRUTH next-step")
    print("profile. It is the data's own consistency limit (sim non-convergence")
    print("+ VGM<->Gardner mismatch). PER-DATASET, not universal -- recompute")
    print("for Strategy 3. A residual at/below the floor is at the noise level.")
    print("Read: MLP lower data error; PGML lower physics residual (claim a).")

    # quick verdict (descriptive, not a pass/fail assertion)
    if pgml_p < mlp_p:
        print(f"\n-> PGML physics residual is LOWER than MLP "
              f"({pgml_p:.3e} < {mlp_p:.3e}): physics-consistency claim supported.")
    else:
        print(f"\n-> NOTE: PGML physics residual not lower than MLP "
              f"({pgml_p:.3e} vs {mlp_p:.3e}). Investigate before reporting.")

    # honest read of where each model sits relative to the data's own floor
    print(f"-> vs DATA floor ({floor_p:.3e}):  "
          f"MLP {'below' if mlp_p < floor_p else 'above'}, "
          f"PGML {'below' if pgml_p < floor_p else 'above'}.")
    if pgml_p < floor_p or mlp_p < floor_p:
        print("   (A model BELOW the data floor is at/under the data's own")
        print("    consistency level -- 'more physical than the truth' is not")
        print("    meaningful; the gap between models is still the comparison.)")

    # ------------------------------------------------------ PER-DEPTH PROFILE
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        depth_idx = np.arange(OUTPUT_DIM)
        fig, ax = plt.subplots(figsize=(6, 7))
        ax.plot(mlp_prof,   depth_idx, label="MLP (lam=0)", color="#5b8dee", lw=1.5)
        ax.plot(pgml_prof,  depth_idx, label="PGML",        color="#f0a255", lw=1.5)
        ax.plot(floor_prof, depth_idx, label="data floor (truth graded)",
                color="#3ecf8e", lw=1.2, ls="--")
        ax.set_xscale("log")
        ax.invert_yaxis()                       # node 0 (surface) at top
        ax.set_xlabel("mean squared Gardner residual (per node)")
        ax.set_ylabel("depth node  (0 = surface, 220 = bottom)")
        ax.set_title("Where each model violates physics (Strategy 1)")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.2)
        prof_png = os.path.join(args.save, "physics_residual_profile.png")
        fig.tight_layout()
        fig.savefig(prof_png, dpi=150)
        print(f"\nSaved per-depth profile : {prof_png}")
        print("  (node 0 = surface-flux BC; green dashed = data's own floor.)")
    except ImportError:
        prof_png = None
        print("\n[matplotlib not available - skipped profile PNG; JSON still has the arrays]")

    # ----------------------------------------------------------------- JSON
    out = {
        "test_set": {"h5": args.h5, "split_dir": args.out},
        "models": {
            "MLP":  {"checkpoint": mlp_path,  "lambda": mlp_ck.get("lambda", 0.0),
                     "val_loss": mlp_ck.get("val_loss"),  "best_epoch": mlp_ck.get("epoch")},
            "PGML": {"checkpoint": pgml_path, "lambda": pgml_ck.get("lambda"),
                     "val_loss": pgml_ck.get("val_loss"), "best_epoch": pgml_ck.get("epoch")},
        },
        "results_2x2": {
            "MLP":  {"data_error": mlp_d,  "physics_residual": mlp_p},
            "PGML": {"data_error": pgml_d, "physics_residual": pgml_p},
        },
        "data_floor": {
            "physics_residual": floor_p,
            "note": "same mean(res^2) metric on VGM truth; PER-DATASET, not universal",
        },
        "per_depth_profile": {
            "MLP":   mlp_prof.tolist(),
            "PGML":  pgml_prof.tolist(),
            "floor": floor_prof.tolist(),
        },
        "gardner_params": {"m_g": pgml_phys["m_g"], "alpha_g": pgml_phys["alpha_g"],
                           "Dz": pgml_phys["Dz"], "Dt": pgml_phys["Dt"]},
    }
    json_path = os.path.join(args.save, "eval_2x2.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved raw numbers       : {json_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
