"""
benchmark_b.py
==============
Phase 4 - EVALUATION (05_Evaluation). Speed claim, Benchmark B: light
R-optimiser decision.

Times ONE full PGML control decision at heap scale: sweep 100 R candidates
through PGML for a batch of B columns, score each predicted profile against
a uniform setpoint, pick the best R per column. That whole round is the unit
being timed.

This is a SPEED benchmark only. It does not test control quality or
generalisation. Those are separate axes.

Objective (grounded in Olivares et al. 2025, Eq. 32, one-step version):
    For each candidate R, predict Se one step ahead with PGML, then score:

        J(R) = mean over 221 nodes of ( Se_next_p(R) - Se_SP )^2

    The R minimising J(R) is the controller output for that column.
    Se_SP = 0.7 (physical Se, matches the paper's target in Fig. 7).
    Se is already physical [0,1] in the model output -- no unscaling needed.
    R candidates in cm/day are passed through the same min-max scaler
    fitted on the training set (key "R_out" in scaler.json).

Batch construction:
    For B columns and 100 candidates, expand to (B*100, 226) and run ONE
    forward pass. Reshape output to (B, 100, 221), score, argmin.
    No Python loop over candidates.

Sweep: B = 1, 10, 100, 1000, 2000 (matches A1 and A2).
Reps: 300 per B (matches A1). Warmup: 20 calls excluded.

Reference comparison:
    Olivares Job 3 (optimisation step): 0.06 to 0.10 s per step, CPU
    (Ryzen 7 3750H). Different hardware, cited only. Not a same-machine
    comparison.

Usage (Windows; plain PowerShell terminal):
    From Leaching Codes folder:
        python "05_Evaluation\benchmark_b.py"
    Or full path from anywhere:
        python "C:\...\Leaching Codes\05_Evaluation\benchmark_b.py"

    Optional arguments:
        --ckpt   PATH   path to pgml_lam100.pt
        --scaler PATH   path to directory containing scaler.json and split.json
        --h5     PATH   path to strategy1.h5
        --reps   INT    timing reps per B (default 300)
        --warmup INT    warmup calls excluded from timing (default 20)
        --save          if passed, save ASCII results table to 05_Evaluation

Origin: Phase 4 speed benchmark (B -- light R-optimiser decision level)
Status: initial
Depends on: preprocess.py (load_scalers, apply_scaler, INPUT_DIM, OUTPUT_DIM),
            train_pgml.py (MLP), sim_gpu.py (_sync, ON_GPU),
            torch, numpy, h5py, json

Changelog
---------
2026.06.15  v1.0  Initial implementation. Benchmark B: PGML one-step
                  R-optimiser timing vs Olivares Job 3 reference.
                  Sweep B = 1, 10, 100, 1000, 2000. 100 R candidates on
                  uniform grid 0.0 to 15.584 cm/day. Real Se snapshot from
                  strategy1.h5 test split. Objective: mean squared deviation
                  from Se_SP = 0.7 across all 221 nodes, per Olivares Eq. 32.
                  One forward pass of size (B*100, 226) per decision.
                  ASCII-only output. No sealed files edited.
"""

import os
import sys
import time
import json
import argparse

import numpy as np
import h5py

# ---------------------------------------------------------------------------
# Resolve sibling phase folders so imports work from any working directory.
# benchmark_b.py lives in 05_Evaluation. Siblings are at the same level.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
for _folder in ("01_Simulator", "03_Preprocessing", "04_Training"):
    sys.path.insert(0, os.path.join(_HERE, "..", _folder))

import torch
from sim_gpu import _sync, ON_GPU
from preprocess import load_scalers, apply_scaler, INPUT_DIM, OUTPUT_DIM
from train_pgml import MLP


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NZ          = OUTPUT_DIM          # 221 depth nodes
N_CAND      = 100                 # number of R candidates
R_MIN       = 0.0                 # cm/day, lower bound (matches sim_gpu Q=0)
R_MAX       = 15.584              # cm/day, upper bound (matches Q=48, Area=308)
SE_SP       = 0.7                 # setpoint physical Se, per Olivares Fig. 7
BATCHES     = [1, 10, 100, 1000, 2000]

# Olivares Job 3 published reference (CPU, Ryzen 7 3750H -- different hardware)
OLIVARES_LO = 0.06               # seconds
OLIVARES_HI = 0.10               # seconds

# Strategy 1 soil (fixed, used to build the suffix vector)
SOIL = dict(Ks=170.0, alpha=0.035, n_vgm=2.267, theta_i=0.14)


# ---------------------------------------------------------------------------
# Default paths (resolved relative to this file so working directory does
# not matter)
# ---------------------------------------------------------------------------
_DEFAULT_CKPT   = os.path.join(_HERE, "sweep_ckpts", "pgml_lam100.pt")
_DEFAULT_SCALER = os.path.join(_HERE, "..", "03_Preprocessing")
_DEFAULT_H5     = os.path.join(
    _HERE, "..", "..", "..", "Data generation", "Python dataset", "strategy1.h5"
)


# ---------------------------------------------------------------------------
# PGML loader (mirrors benchmark_forward_step and benchmark_rollout)
# ---------------------------------------------------------------------------
def load_pgml(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MLP(
        INPUT_DIM, OUTPUT_DIM,
        width=ckpt["args"]["width"],
        depth=ckpt["args"]["depth"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Build the 100 R candidates, scaled, as a (N_CAND,) float32 tensor.
# Candidates stay within [R_MIN, R_MAX] -- clamping lives here, not in PGML.
# ---------------------------------------------------------------------------
def build_r_candidates(scalers, device):
    r_phys = np.linspace(R_MIN, R_MAX, N_CAND, dtype=np.float32)  # (100,)
    r_scaled = apply_scaler(r_phys, "R_out", scalers)              # (100,) float32
    return torch.from_numpy(r_scaled).to(device)                   # (100,)


# ---------------------------------------------------------------------------
# Build the 4-element soil suffix vector (scaled).
# Strategy 1: all soil params are fixed so min==max -> apply_scaler returns 0.
# ---------------------------------------------------------------------------
def build_soil_vec(scalers, device):
    soil = np.array([
        apply_scaler(np.array([SOIL["Ks"]]),      "Ks",      scalers)[0],
        apply_scaler(np.array([SOIL["alpha"]]),   "alpha",   scalers)[0],
        apply_scaler(np.array([SOIL["n_vgm"]]),   "n_vgm",   scalers)[0],
        apply_scaler(np.array([SOIL["theta_i"]]), "theta_i", scalers)[0],
    ], dtype=np.float32)
    return torch.from_numpy(soil).to(device)   # (4,)


# ---------------------------------------------------------------------------
# Pull a real Se snapshot from the test split of strategy1.h5.
# Returns a (221,) float32 numpy array (one column, one timestep).
# Timestep 300 (middle of the 600-hour run) is representative.
# ---------------------------------------------------------------------------
def load_se_snapshot(h5_path, scaler_dir):
    split_path = os.path.join(scaler_dir, "split.json")
    with open(split_path, "r") as f:
        split = json.load(f)
    test_ids = split["test"]
    if not test_ids:
        raise RuntimeError("split.json has no test IDs -- rerun preprocess.py")

    # Use the first test run, timestep 300 (mid-run, representative)
    run_key = test_ids[0]
    t_snap  = 300

    with h5py.File(h5_path, "r") as f:
        # Se_out shape: (221, 600) -- nodes x timesteps
        se = f[run_key]["Se_out"][:, t_snap].astype(np.float32)   # (221,)

    print(f"[Snapshot] run={run_key}  t={t_snap}  "
          f"Se min={se.min():.4f}  max={se.max():.4f}  mean={se.mean():.4f}")
    return se


# ---------------------------------------------------------------------------
# Build the (B*N_CAND, 226) input tensor for one decision.
#
# Layout per sample (same as training, per preprocess.py HeapDataset):
#     Se[:,t]     221  physical Se -- no scaling, already [0,1]
#     R           1    scaled R candidate
#     soil        4    scaled soil params (all 0.0 for Strategy 1)
#
# Construction:
#   1. se_snap (221,) -> tile to (B, 221) -> repeat each row N_CAND times
#      -> (B*N_CAND, 221)
#   2. r_cands (N_CAND,) -> repeat for each column
#      -> (B*N_CAND, 1)
#   3. soil_vec (4,) -> broadcast
#      -> (B*N_CAND, 4)
#   4. cat -> (B*N_CAND, 226)
# ---------------------------------------------------------------------------
def build_input(se_snap, r_cands, soil_vec, B, device):
    # se_snap: (221,) numpy -> (1, 221) -> (B, 221) -> (B, N_CAND, 221)
    # then reshape to (B*N_CAND, 221)
    se_t = torch.from_numpy(se_snap).to(device)            # (221,)
    se_t = se_t.unsqueeze(0).expand(B, -1)                 # (B, 221)
    se_t = se_t.unsqueeze(1).expand(-1, N_CAND, -1)        # (B, N_CAND, 221)
    se_t = se_t.reshape(B * N_CAND, NZ)                    # (B*N_CAND, 221)

    # r_cands: (N_CAND,) -> (1, N_CAND) -> (B, N_CAND) -> (B*N_CAND, 1)
    r_t = r_cands.unsqueeze(0).expand(B, -1)               # (B, N_CAND)
    r_t = r_t.reshape(B * N_CAND, 1)                       # (B*N_CAND, 1)

    # soil_vec: (4,) -> (B*N_CAND, 4)
    soil_t = soil_vec.unsqueeze(0).expand(B * N_CAND, -1)  # (B*N_CAND, 4)

    return torch.cat([se_t, r_t, soil_t], dim=1)           # (B*N_CAND, 226)


# ---------------------------------------------------------------------------
# ONE full control decision for B columns:
#   1. Build (B*N_CAND, 226) input
#   2. Forward pass -> (B*N_CAND, 221)
#   3. Reshape to (B, N_CAND, 221)
#   4. Score: mean squared deviation from Se_SP across 221 nodes
#      -> score shape (B, N_CAND)
#   5. Argmin over N_CAND -> best candidate index per column (B,)
# Returns best_idx (B,) int tensor. Timing does not depend on returning more.
# ---------------------------------------------------------------------------
def one_decision(model, x):
    with torch.no_grad():
        Se_next = model(x)                                  # (B*N_CAND, 221)
    B = x.shape[0] // N_CAND
    Se_next = Se_next.view(B, N_CAND, NZ)                  # (B, N_CAND, 221)
    score   = ((Se_next - SE_SP) ** 2).mean(dim=2)         # (B, N_CAND)
    best    = score.argmin(dim=1)                           # (B,)
    return best


# ---------------------------------------------------------------------------
# Timing harness (mirrors benchmark_rollout.time_calls exactly).
# Warm up, then time fn() over reps calls with GPU sync around each.
# Returns (mean, median, std, min) in seconds.
# ---------------------------------------------------------------------------
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
    return (float(times.mean()), float(np.median(times)),
            float(times.std()),  float(times.min()))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Benchmark B: PGML R-optimiser speed vs Olivares Job 3 -- CENG0057"
    )
    p.add_argument("--ckpt",   default=_DEFAULT_CKPT,
                   help="path to pgml_lam100.pt")
    p.add_argument("--scaler", default=_DEFAULT_SCALER,
                   help="directory containing scaler.json and split.json")
    p.add_argument("--h5",     default=_DEFAULT_H5,
                   help="path to strategy1.h5")
    p.add_argument("--reps",   type=int, default=300,
                   help="timing reps per B (default 300; reduce if B=2000 is slow)")
    p.add_argument("--warmup", type=int, default=20,
                   help="warmup calls excluded from timing (default 20)")
    p.add_argument("--save",   action="store_true",
                   help="save ASCII results table to 05_Evaluation/benchmark_b_results.txt")
    args = p.parse_args()

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_cuda = (device.type == "cuda")

    def torch_sync():
        if torch_cuda:
            torch.cuda.synchronize()

    # --- header ---
    print("=" * 80)
    print("BENCHMARK B  --  PGML LIGHT R-OPTIMISER  vs  OLIVARES JOB 3 REFERENCE")
    print("=" * 80)
    print(f"Device          : {device}" +
          (f"  ({torch.cuda.get_device_name(0)})" if torch_cuda else ""))
    print(f"Simulator back  : {'GPU (cupy)' if ON_GPU else 'CPU (numpy)'}")
    print(f"Candidates      : {N_CAND} R values  "
          f"[{R_MIN:.3f}, {R_MAX:.3f}] cm/day  step ~{R_MAX/(N_CAND-1):.4f}")
    print(f"Setpoint Se_SP  : {SE_SP}")
    print(f"Score           : mean( (Se_next - Se_SP)^2 ) over {NZ} nodes  -> argmin")
    print(f"Input per rep   : (B*{N_CAND}, {INPUT_DIM}) -- one forward pass")
    print(f"Reps            : {args.reps}   Warmup: {args.warmup}")

    # --- checkpoint ---
    print(f"\nLoading PGML    : {args.ckpt}")
    if not os.path.isfile(args.ckpt):
        print("ERROR: checkpoint not found.")
        print("Pass the correct path with --ckpt, or place pgml_lam100.pt in")
        print("05_Evaluation\\sweep_ckpts\\  Aborting.")
        return
    model  = load_pgml(args.ckpt, device)
    n_par  = sum(p.numel() for p in model.parameters())
    print(f"  loaded  ({n_par:,} params)")

    # --- scalers ---
    print(f"Loading scalers : {args.scaler}")
    if not os.path.isfile(os.path.join(args.scaler, "scaler.json")):
        print("ERROR: scaler.json not found in that directory. Aborting.")
        return
    scalers = load_scalers(args.scaler)

    # --- snapshot ---
    print(f"Loading H5      : {args.h5}")
    if not os.path.isfile(args.h5):
        print("ERROR: strategy1.h5 not found at that path. Aborting.")
        return
    se_snap = load_se_snapshot(args.h5, args.scaler)   # (221,) float32

    # --- build fixed tensors ---
    r_cands  = build_r_candidates(scalers, device)     # (100,)  scaled
    soil_vec = build_soil_vec(scalers, device)          # (4,)    scaled (all 0 S1)

    print(f"\n[Soil vec] all zeros = {(soil_vec == 0).all().item()} "
          f"(expected True for Strategy 1 fixed soil)")
    print(f"[R cands]  scaled min={r_cands.min():.4f}  max={r_cands.max():.4f}  "
          f"(physical {R_MIN} to {R_MAX} cm/day)")

    # --- sweep ---
    print("\n" + "-" * 80)
    hdr = (f"{'B':>6}  {'input shape':>14}  "
           f"{'mean ms':>10}  {'median ms':>10}  {'min ms':>8}  "
           f"{'us/col':>10}")
    print(hdr)
    print("-" * 80)

    rows = []
    for B in BATCHES:
        x = build_input(se_snap, r_cands, soil_vec, B, device)  # (B*100, 226)

        def decision():
            one_decision(model, x)

        mean_s, med_s, std_s, min_s = time_calls(
            decision, torch_sync, args.reps, args.warmup
        )

        mean_ms  = mean_s  * 1e3
        med_ms   = med_s   * 1e3
        min_ms   = min_s   * 1e3
        us_col   = (mean_s / B) * 1e6
        shape_str = f"({B * N_CAND}, {INPUT_DIM})"

        print(f"{B:>6}  {shape_str:>14}  "
              f"{mean_ms:>10.3f}  {med_ms:>10.3f}  {min_ms:>8.3f}  "
              f"{us_col:>10.2f}")
        rows.append((B, shape_str, mean_ms, med_ms, min_ms, us_col, mean_s))

    print("-" * 80)
    print(f"Columns: B = columns per decision | input shape = (B*{N_CAND}, {INPUT_DIM})")
    print(f"         mean/median/min in ms per decision | us/col = mean / B in microseconds")

    # --- reference comparison ---
    # Use B=2000 (heap scale) as the primary comparison point.
    # Olivares Job 3 is a single-column optimisation on CPU.
    # We compare at B=1 (single column) to the Olivares single-column number,
    # and separately state the B=2000 heap-scale number.
    mean_b1_s    = rows[0][6]   # B=1 mean in seconds
    mean_b2000_s = rows[-1][6]  # B=2000 mean in seconds

    olivares_mid = (OLIVARES_LO + OLIVARES_HI) / 2.0   # 0.08 s midpoint

    speedup_lo_b1    = OLIVARES_LO    / mean_b1_s    if mean_b1_s > 0 else float("nan")
    speedup_hi_b1    = OLIVARES_HI    / mean_b1_s    if mean_b1_s > 0 else float("nan")
    speedup_mid_b1   = olivares_mid   / mean_b1_s    if mean_b1_s > 0 else float("nan")

    # Per-column comparison at B=2000: PGML cost per column vs Olivares per column
    pgml_per_col_s   = mean_b2000_s / 2000.0
    speedup_col_lo   = OLIVARES_LO  / pgml_per_col_s if pgml_per_col_s > 0 else float("nan")
    speedup_col_hi   = OLIVARES_HI  / pgml_per_col_s if pgml_per_col_s > 0 else float("nan")

    print("\n" + "=" * 80)
    print("REFERENCE COMPARISON  --  Olivares et al. (2025) Job 3")
    print("=" * 80)
    print(f"Olivares Job 3   : {OLIVARES_LO:.2f} to {OLIVARES_HI:.2f} s per step")
    print(f"                   CPU (Ryzen 7 3750H) -- DIFFERENT hardware, cited only")
    print(f"                   Single-column NLP solve (interior point, CasADi)")
    print("")
    print(f"PGML B=1 (single column decision):")
    print(f"  mean            : {mean_b1_s*1e3:.3f} ms  ({mean_b1_s:.6f} s)")
    print(f"  difference      : {olivares_mid - mean_b1_s:.4f} s faster (vs midpoint)")
    print(f"  speedup         : {speedup_lo_b1:.0f}x to {speedup_hi_b1:.0f}x "
          f"(vs {OLIVARES_LO:.2f} to {OLIVARES_HI:.2f} s range)")
    print("")
    print(f"PGML B=2000 (heap scale, 2000 columns in parallel):")
    print(f"  total mean      : {mean_b2000_s*1e3:.3f} ms  ({mean_b2000_s:.6f} s)")
    print(f"  per-column mean : {pgml_per_col_s*1e6:.3f} us  ({pgml_per_col_s:.8f} s)")
    print(f"  per-col speedup : {speedup_col_lo:.0f}x to {speedup_col_hi:.0f}x "
          f"(vs Olivares single-column)")
    print("")
    print("NOTE: speedup numbers cross hardware boundaries (PGML on RTX 5060 GPU,")
    print("      Olivares on Ryzen 7 3750H CPU). They illustrate the latency gap;")
    print("      they are not a controlled same-hardware comparison.")
    print("=" * 80)

    # --- optional save ---
    if args.save:
        out_path = os.path.join(_HERE, "benchmark_b_results.txt")
        with open(out_path, "w") as f:
            f.write("BENCHMARK B  --  PGML LIGHT R-OPTIMISER TIMING\n")
            f.write(f"Run date (approx): 2026.06.15\n")
            f.write(f"Checkpoint: pgml_lam100.pt  lambda=100  1,021,661 params\n")
            f.write(f"Candidates: {N_CAND}  R in [{R_MIN}, {R_MAX}] cm/day\n")
            f.write(f"Setpoint Se_SP: {SE_SP}\n")
            f.write(f"Reps: {args.reps}  Warmup: {args.warmup}\n\n")
            f.write(f"{'B':>6}  {'input shape':>14}  "
                    f"{'mean ms':>10}  {'median ms':>10}  {'min ms':>8}  "
                    f"{'us/col':>10}\n")
            f.write("-" * 68 + "\n")
            for row in rows:
                B, shape_str, mean_ms, med_ms, min_ms, us_col, _ = row
                f.write(f"{B:>6}  {shape_str:>14}  "
                        f"{mean_ms:>10.3f}  {med_ms:>10.3f}  {min_ms:>8.3f}  "
                        f"{us_col:>10.2f}\n")
            f.write("-" * 68 + "\n\n")
            f.write(f"Olivares Job 3 reference: {OLIVARES_LO:.2f} to {OLIVARES_HI:.2f} s  "
                    f"CPU Ryzen 7 3750H  different hardware\n")
            f.write(f"PGML B=1 mean: {mean_b1_s*1e3:.3f} ms\n")
            f.write(f"PGML B=1 speedup: {speedup_lo_b1:.0f}x to {speedup_hi_b1:.0f}x\n")
            f.write(f"PGML B=2000 per-col: {pgml_per_col_s*1e6:.3f} us\n")
            f.write(f"PGML B=2000 per-col speedup: "
                    f"{speedup_col_lo:.0f}x to {speedup_col_hi:.0f}x\n")
        print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()
