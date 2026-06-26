"""
benchmark_rollout.py
====================
Phase 4 - EVALUATION (05_Evaluation). Speed claim, part 2: full rollout.

Chains 600 forward steps (25 days x 24 hours) two ways, from the same
initial state and the same constant irrigation schedule:

    PGML      : 600 successive neural-net passes (float32, PyTorch)
                output of each step fed back as input to the next
    simulator : one call to run_batch for the full 25-day run (float64, sim_gpu)

Both sides run on the GPU. Batch sizes swept: 1, 10, 100, 1000, 2000.

Two results are reported:

    Speed : total wall time for the full 600-step chain, mean over reps,
            plus speedup factor (sim time / PGML time).

    Drift : after both rollouts, mean absolute error between PGML Se and
            sim Se at each of the 600 timesteps. Reports max drift over
            the run and the timestep at which it peaks. Drift is computed
            for B=1 only (representative; heap-scale drift is the same
            per column).

INPUT VECTOR (226):
    Se[:, t]   221  current moisture profile (already in [0,1], no scaling)
    R[t]         1  irrigation rate (min-max scaled via scaler.json)
    soil vec     4  Ks, alpha, n, theta_i (scaled; all zero for Strategy 1
                    because min==max for fixed soil -- see preprocess.py)

IRRIGATION SCHEDULE:
    Constant Q = 24 L/h for all 25 days (option 1: synthetic, self-contained).
    Converted to R (cm/h) as: R = (Q / Area) * 100.

Does NOT modify sim_gpu.py or preprocess.py.

Usage (from 05_Evaluation):
    python benchmark_rollout.py
Optional:
    --ckpt PATH   --batches 1,10,100,1000,2000   --reps 5   --warmup 1

Origin: Phase 4 speed benchmark (rollout level, A2)
Status: rollout stage
Depends on: sim_gpu.py (run_batch, xp, ON_GPU, _sync),
            preprocess.py (INPUT_DIM, OUTPUT_DIM, load_scalers, apply_scaler),
            train_pgml.py (MLP), torch, numpy

Changelog
---------
2026.06.14  v1.0  Full 25-day rollout benchmark: PGML chain vs sim_gpu
                  run_batch. Speed table (mean, median, min) + drift over
                  600 timesteps for B=1. Constant Q schedule (synthetic).
"""

import os
import sys
import time
import argparse
import json

import numpy as np

# --- resolve sibling folders the same way benchmark_forward_step.py does -----
_HERE = os.path.dirname(os.path.abspath(__file__))
for _folder in ("01_Simulator", "03_Preprocessing", "04_Training"):
    sys.path.append(os.path.join(_HERE, "..", _folder))

import torch

from sim_gpu import run_batch, xp, ON_GPU, _sync
from preprocess import INPUT_DIM, OUTPUT_DIM, load_scalers, apply_scaler
from train_pgml import MLP


# =============================================================================
# Fixed simulation parameters (Strategy 1, matches sim_gpu defaults)
# =============================================================================
SIM = dict(
    theta_s=0.33, theta_r=0.0, theta_i=0.14,
    Ks=170.0, alpha=0.035, n=2.267,
    H=5.50, Area=308.0, Dz=2.5,
    Dt=1.0 / 24.0, D_H=24, tf=25, n_iter=20,
)
NZ1      = int(SIM["H"] * 100 / SIM["Dz"]) + 1   # 221
N_STEPS  = SIM["tf"] * SIM["D_H"]                 # 600
Q_CONST  = 24.0                                    # L/h constant for all days
R_CONST  = (Q_CONST / SIM["Area"]) * 100.0        # cm/h


# =============================================================================
# PGML loader (mirrors evaluate.load_ckpt)
# =============================================================================
def load_pgml(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MLP(INPUT_DIM, OUTPUT_DIM,
                width=ckpt["args"]["width"],
                depth=ckpt["args"]["depth"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# =============================================================================
# Build scaled inputs for PGML rollout
# R_const_scaled : float -- the single scaled R value (constant schedule)
# soil_vec       : (4,) float32 -- scaled soil params (all 0.0 for Strategy 1)
# =============================================================================
def build_pgml_inputs(scalers):
    R_scaled = float(apply_scaler(np.array([R_CONST]), "R_out", scalers)[0])
    soil_vec = np.array([
        apply_scaler(np.array([SIM["Ks"]]),     "Ks",      scalers)[0],
        apply_scaler(np.array([SIM["alpha"]]),  "alpha",   scalers)[0],
        apply_scaler(np.array([SIM["n"]]),      "n_vgm",   scalers)[0],
        apply_scaler(np.array([SIM["theta_i"]]),"theta_i", scalers)[0],
    ], dtype=np.float32)
    return R_scaled, soil_vec


# =============================================================================
# PGML rollout: chain 600 forward passes on the GPU.
# Se_init : (B, 221) float32 tensor on device -- starting profile
# Returns Se_traj : (B, 221, N_STEPS) float32 numpy on CPU
# =============================================================================
def pgml_rollout(model, Se_init, R_scaled, soil_vec, device):
    B = Se_init.shape[0]

    # fixed suffix: R + soil, same every step, shape (B, 5)
    suffix = torch.zeros(B, 5, device=device, dtype=torch.float32)
    suffix[:, 0] = R_scaled
    suffix[:, 1:] = torch.tensor(soil_vec, device=device, dtype=torch.float32)

    Se_cur = Se_init.clone()                           # (B, 221)
    traj = torch.empty(B, NZ1, N_STEPS,
                       device=device, dtype=torch.float32)

    with torch.no_grad():
        for t in range(N_STEPS):
            x = torch.cat([Se_cur, suffix], dim=1)    # (B, 226)
            Se_cur = model(x)                          # (B, 221)
            traj[:, :, t] = Se_cur

    return traj.cpu().numpy()


# =============================================================================
# Simulator rollout: one run_batch call for the full 25-day run.
# Returns Se_traj : (B, 221, N_STEPS) float64 numpy on CPU
# =============================================================================
def sim_rollout(B):
    Q = np.full((B, SIM["tf"]), Q_CONST, dtype=np.float64)
    Se = run_batch(Q,
                   Ks=SIM["Ks"], theta_s=SIM["theta_s"], theta_r=SIM["theta_r"],
                   theta_i=SIM["theta_i"], alpha=SIM["alpha"], n=SIM["n"],
                   H=SIM["H"], Area=SIM["Area"], tf=SIM["tf"],
                   Dz=SIM["Dz"], Dt=SIM["Dt"], D_H=SIM["D_H"],
                   n_iter=SIM["n_iter"])
    _sync()
    return (Se.get() if ON_GPU else Se)   # bring to CPU numpy


# =============================================================================
# Timing helper: warm up, then time fn() over reps calls.
# Returns (mean, median, std, min) in seconds.
# =============================================================================
def time_calls(fn, sync_fn, reps, warmup):
    for _ in range(warmup):
        fn()
    sync_fn()
    times = np.empty(reps, dtype=np.float64)
    for i in range(reps):
        t0 = time.perf_counter()
        fn()
        sync_fn()
        times[i] = time.perf_counter() - t0
    return float(times.mean()), float(np.median(times)), \
           float(times.std()),  float(times.min())


# =============================================================================
# MAIN
# =============================================================================
def main():
    p = argparse.ArgumentParser(
        description="Rollout speed + drift: PGML chain vs sim_gpu -- CENG0057")
    p.add_argument("--ckpt",
                   default=os.path.join(_HERE, "sweep_ckpts", "pgml_lam100.pt"))
    p.add_argument("--scaler",
                   default=os.path.join(_HERE, "..", "03_Preprocessing", "scaler.json"))
    p.add_argument("--batches", default="1,10,100,1000,2000")
    p.add_argument("--reps",   type=int, default=5,
                   help="rollout reps (default 5; each rep is 600 steps)")
    p.add_argument("--warmup", type=int, default=1)
    args = p.parse_args()

    batches = [int(b) for b in args.batches.split(",")]

    tdev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_cuda = (tdev.type == "cuda")

    def torch_sync():
        if torch_cuda:
            torch.cuda.synchronize()

    # --- header ---
    print("=" * 100)
    print("ROLLOUT SPEED + DRIFT  --  PGML 600-step chain  vs  sim_gpu run_batch")
    print("=" * 100)
    print(f"PGML device      : {tdev}" +
          (f"  ({torch.cuda.get_device_name(0)})" if torch_cuda else ""))
    print(f"Simulator backend: {'GPU (cupy)' if ON_GPU else 'CPU (numpy fallback)'}")
    if not ON_GPU or not torch_cuda:
        print("WARNING: not both on GPU. Numbers are not a GPU-vs-GPU comparison.")
    print(f"Steps per rollout: {N_STEPS}  ({SIM['tf']} days x {SIM['D_H']} h)")
    print(f"Q schedule       : constant {Q_CONST:.1f} L/h all days  "
          f"-> R = {R_CONST:.6f} cm/h")
    print(f"Reps             : {args.reps}   Warmup: {args.warmup}")

    # --- load checkpoint ---
    print(f"\nLoading PGML     : {args.ckpt}")
    if not os.path.isfile(args.ckpt):
        print("ERROR: checkpoint not found. Pass correct path with --ckpt. Aborting.")
        return
    model = load_pgml(args.ckpt, tdev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  loaded ({n_params:,} params)")

    # --- load scalers ---
    print(f"Loading scalers  : {args.scaler}")
    if not os.path.isfile(args.scaler):
        print("ERROR: scaler.json not found. Pass correct path with --scaler. Aborting.")
        return
    scalers = load_scalers(os.path.dirname(args.scaler))
    R_scaled, soil_vec = build_pgml_inputs(scalers)
    print(f"  R_const scaled : {R_scaled:.6f}")
    print(f"  soil vec       : {soil_vec}  (all zeros expected for Strategy 1)")

    # --- speed table ---
    print("\n" + "-" * 100)
    hdr = (f"{'B':>6}{'sim mean s':>13}{'sim med s':>11}{'sim min s':>11}"
           f"{'PGML mean s':>13}{'PGML med s':>12}{'PGML min s':>12}"
           f"{'PGML faster':>13}{'sim ms/col':>12}{'PGML ms/col':>12}")
    print(hdr)
    print("-" * 100)

    drift_Se_sim  = None   # stored from B=1 for drift comparison
    drift_Se_pgml = None

    for B in batches:
        # initial state: uniform theta_i, converted to Se
        theta_i = SIM["theta_i"]
        theta_s = SIM["theta_s"]
        theta_r = SIM["theta_r"]
        Se0_val = (theta_i - theta_r) / (theta_s - theta_r)   # scalar ~0.424

        # --- simulator ---
        sim_m, sim_med, sim_s, sim_min = time_calls(
            lambda: sim_rollout(B), _sync, args.reps, args.warmup)

        # --- PGML ---
        Se_init = torch.full((B, NZ1), Se0_val,
                             device=tdev, dtype=torch.float32)

        pgml_m, pgml_med, pgml_s, pgml_min = time_calls(
            lambda: pgml_rollout(model, Se_init, R_scaled, soil_vec, tdev),
            torch_sync, args.reps, args.warmup)

        factor     = sim_m / pgml_m if pgml_m > 0 else float("nan")
        sim_mscol  = (sim_m  / B) * 1e3
        pgml_mscol = (pgml_m / B) * 1e3

        print(f"{B:>6}"
              f"{sim_m:>13.3f}{sim_med:>11.3f}{sim_min:>11.3f}"
              f"{pgml_m:>13.3f}{pgml_med:>11.3f}{pgml_min:>11.3f}"
              f"{factor:>11.1f}x{sim_mscol:>12.3f}{pgml_mscol:>12.3f}")

        # store B=1 trajectories for drift
        if B == 1:
            drift_Se_sim  = sim_rollout(1)            # (1, 221, 600) float64
            drift_Se_pgml = pgml_rollout(
                model, Se_init, R_scaled, soil_vec, tdev)  # (1, 221, 600) float32

    print("-" * 100)
    print("Read: times in seconds per rollout (600 steps). ms/col = per column.")
    print("med=median, min=latency floor. Precision: sim float64, PGML float32.")
    print("=" * 100)

    # --- drift ---
    if drift_Se_sim is not None and drift_Se_pgml is not None:
        print("\n" + "-" * 60)
        print("DRIFT  --  B=1, MAE per timestep (PGML vs sim, float64 ref)")
        print("-" * 60)
        # mae_t: mean over 221 nodes, shape (600,)
        mae_t = np.mean(np.abs(drift_Se_pgml[0].astype(np.float64)
                               - drift_Se_sim[0]), axis=0)
        max_mae   = float(mae_t.max())
        peak_step = int(mae_t.argmax())
        final_mae = float(mae_t[-1])
        print(f"Max MAE over run : {max_mae:.6e}  (at step {peak_step} / {N_STEPS})")
        print(f"Final MAE (t=600): {final_mae:.6e}")
        print(f"Mean MAE over run: {float(mae_t.mean()):.6e}")
        print("\nPer-day MAE summary (mean of 24 hourly steps):")
        print(f"  {'Day':>4}  {'MAE':>12}")
        for d in range(SIM["tf"]):
            day_mae = float(mae_t[d * SIM["D_H"] : (d + 1) * SIM["D_H"]].mean())
            print(f"  {d+1:>4}  {day_mae:>12.6e}")
        print("-" * 60)


if __name__ == "__main__":
    main()
