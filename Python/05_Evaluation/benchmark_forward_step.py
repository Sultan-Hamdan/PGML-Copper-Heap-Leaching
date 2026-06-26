"""
benchmark_forward_step.py
=========================
Phase 4 - EVALUATION (05_Evaluation). Speed claim, part 1: the forward step.

Times ONE forward step two ways, on the same GPU, from the same start state and
the same irrigation rate, each producing Se(z, t+1h) for one column:

    PGML      : one neural-net forward pass            (float32, PyTorch)
    simulator : one hour-step of the VGM solver        (float64, sim_gpu)
                = build RHS once + 20 fixed-point Thomas iterations

This is the like-for-like forward-model comparison: PGML was trained to
reproduce exactly this hour-step. Accuracy (does PGML match the VGM Se) is
settled elsewhere (evaluate.py); this script measures SPEED only.

Reported in seconds first, then as an x-times factor, single column and batched
across many columns (the heap-scale view). Precision is native on each side
(simulator float64, PGML float32) because that is how each is actually run; the
gap is stated, not hidden.

DOES NOT EDIT sim_gpu.py. It imports sim_gpu's existing functions and calls them
in the same order the hourly step already uses. The data-generation code is
untouched.

Fairness controls: per-framework warm-up (exclude cold start and kernel
compile), explicit GPU synchronize around every timed call (GPU calls are
async, so naive timing reads wrong), many repeats, report mean / median / std / min.

Usage (Windows; from 05_Evaluation, or just press Run in VS Code):
    python benchmark_forward_step.py
  (default checkpoint: 05_Evaluation\\sweep_ckpts\\pgml_lam100.pt)
Optional:
    --ckpt PATH   --batches 1,10,100,1000,2000   --reps 100   --warmup 20

Origin: Phase 4 speed benchmark (forward-step level)
Status: forward-step stage (decision-level / add-on is a later, separate step)
Depends on: sim_gpu.py (_build_b, _build_diags_c, _solve, xp, ON_GPU, _sync),
            thomas_kernel.py (via sim_gpu), preprocess.py (INPUT_DIM, OUTPUT_DIM),
            train_pgml.py (MLP), torch, numpy

Changelog
---------
2026.06.14  v1.2  Added MEDIAN and MIN columns to the results table. time_calls
                  now returns (mean, median, std, min). Table widened to fit.
                  Bump --reps to 300+ for cleaner figures.
2026.06.14  v1.1  Folder paths corrected to the real layout (sim_gpu and
                  thomas_kernel in 01_Simulator). Default --ckpt points to
                  05_Evaluation\\sweep_ckpts\\pgml_lam100.pt, resolved relative
                  to this file so the VS Code working directory does not matter.
                  Added a checkpoint-not-found check.
2026.06.14  v1.0  Forward-step timing harness: PGML pass vs sim_gpu hour-step,
                  single + batched sweep. Native precision (sim f64, PGML f32).
                  Imports sim_gpu functions; does not modify sim_gpu.
"""

import os
import sys
import time
import argparse

import numpy as np

# --- locate sibling phase folders so imports resolve from 05_Evaluation -------
# sim_gpu.py and thomas_kernel.py live in 01_Simulator; preprocess in
# 03_Preprocessing; train_pgml (and physics_loss) in 04_Training. Paths are
# relative to THIS file, so the working directory does not matter (VS Code may
# run from the workspace root, not 05_Evaluation).
_HERE = os.path.dirname(os.path.abspath(__file__))
for _folder in ("01_Simulator", "03_Preprocessing", "04_Training"):
    sys.path.append(os.path.join(_HERE, "..", _folder))

import torch

# Simulator building blocks (used as-is; sim_gpu is NOT modified):
from sim_gpu import _build_b, _build_diags_c, _solve, xp, ON_GPU, _sync
# Dimensions and the trained model class (single source of truth):
from preprocess import INPUT_DIM, OUTPUT_DIM
from train_pgml import MLP


# =============================================================================
# Simulator parameters (Strategy 1 fixed soil) taken verbatim from sim_gpu's
# run_batch defaults. Fixed-soil params are passed as scalars, exactly as
# run_batch does for the scalar case.
# =============================================================================
SIM = dict(theta_s=0.33, theta_r=0.0, theta_i=0.14, Ks=170.0,
           alpha=0.035, n=2.267, H=5.50, Area=308.0,
           Dz=2.5, Dt=1.0 / 24.0, n_iter=20)
NZ1 = int(SIM["H"] * 100 / SIM["Dz"]) + 1          # 221 nodes, matches OUTPUT_DIM


# =============================================================================
# ONE simulator hour-step. This is the inner block of sim_gpu.run_batch's
# hour loop, reproduced call-for-call:
#     b = build RHS once
#     repeat n_iter: build diagonals -> Thomas solve -> clip
# Returns theta after one hour. No physics added here.
# =============================================================================
def sim_step_hour(theta, R):
    b = _build_b(theta, R, SIM["theta_s"], SIM["theta_r"], SIM["Ks"],
                 SIM["alpha"], SIM["n"], SIM["Dz"], SIM["Dt"])
    theta_v = theta
    for _ in range(SIM["n_iter"]):
        lo, ma, up, c = _build_diags_c(theta_v, R, SIM["theta_s"], SIM["theta_r"],
                                       SIM["Ks"], SIM["alpha"], SIM["n"],
                                       SIM["Dz"], SIM["Dt"])
        theta_v = _solve(lo, ma, up, b + c)
        theta_v = xp.clip(theta_v, SIM["theta_r"], SIM["theta_s"])
    return theta_v


def make_sim_state(B):
    """Initial state for B columns, identical to run_batch's init (uniform
    theta_i), plus a representative irrigation rate. Timing is insensitive to
    the exact values; the operation count per step is the same."""
    theta = xp.ones((B, NZ1), dtype=xp.float64) * SIM["theta_i"]
    R = xp.full((B,), (24.0 / SIM["Area"]) * 100.0, dtype=xp.float64)
    return theta, R


# =============================================================================
# PGML loader. Mirrors evaluate.load_ckpt (proven) so the model is built and
# loaded the exact same way it is scored.
# =============================================================================
def load_pgml(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MLP(INPUT_DIM, OUTPUT_DIM,
                width=ckpt["args"]["width"], depth=ckpt["args"]["depth"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# =============================================================================
# Timing: warm up, then time each call with a device-synchronize around it so
# the wall-clock reflects work actually finished (GPU calls are async).
# Returns (mean, median, std, min) in seconds.
# =============================================================================
def time_calls(fn, sync, reps, warmup):
    for _ in range(warmup):
        fn()
    sync()
    times = np.empty(reps, dtype=np.float64)
    for i in range(reps):
        t0 = time.perf_counter()
        fn()
        sync()
        times[i] = time.perf_counter() - t0
    return float(times.mean()), float(np.median(times)), float(times.std()), float(times.min())


def main():
    p = argparse.ArgumentParser(description="Forward-step speed: PGML vs sim_gpu -- CENG0057")
    p.add_argument("--ckpt",
                   default=os.path.join(_HERE, "sweep_ckpts", "pgml_lam100.pt"),
                   help="path to the PGML checkpoint "
                        "(default: 05_Evaluation\\sweep_ckpts\\pgml_lam100.pt)")
    p.add_argument("--batches", default="1,10,100,1000,2000",
                   help="comma-separated batch sizes (columns per call)")
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    args = p.parse_args()

    batches = [int(b) for b in args.batches.split(",")]

    tdev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_cuda = (tdev.type == "cuda")

    def torch_sync():
        if torch_cuda:
            torch.cuda.synchronize()

    print("=" * 100)
    print("FORWARD-STEP SPEED  --  PGML pass  vs  sim_gpu hour-step")
    print("=" * 100)
    print(f"PGML device      : {tdev}" + (f"  ({torch.cuda.get_device_name(0)})" if torch_cuda else ""))
    print(f"Simulator backend: {'GPU (cupy)' if ON_GPU else 'CPU (numpy fallback)'}")
    if not ON_GPU or not torch_cuda:
        print("WARNING: not both on GPU. Numbers are not a GPU-vs-GPU comparison.")
    print(f"Precision        : simulator float64 (native), PGML float32 (native)")
    print(f"Unit of work     : Se(z, t+1h), one column. Sim step = {SIM['n_iter']} Thomas iters.")
    print(f"Nodes            : {NZ1}  (must equal OUTPUT_DIM={OUTPUT_DIM})")
    print(f"Reps             : {args.reps}   Warmup: {args.warmup}")

    if NZ1 != OUTPUT_DIM:
        print(f"WARNING: node count {NZ1} != OUTPUT_DIM {OUTPUT_DIM}. Check H/Dz.")

    print(f"\nLoading PGML     : {args.ckpt}")
    if not os.path.isfile(args.ckpt):
        print("ERROR: checkpoint not found at that path.")
        print("Pass the right path with --ckpt, or place pgml_lam100.pt in")
        print("05_Evaluation\\sweep_ckpts. Aborting.")
        return
    model = load_pgml(args.ckpt, tdev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  loaded ({n_params:,} params)")

    print("\n" + "-" * 100)
    hdr = (f"{'B':>6}{'sim mean ms':>14}{'sim med ms':>12}{'sim min ms':>12}"
           f"{'PGML mean ms':>14}{'PGML med ms':>13}{'PGML min ms':>13}"
           f"{'PGML faster':>13}{'sim us/col':>12}{'PGML us/col':>12}")
    print(hdr)
    print("-" * 100)

    max_sim_std = 0.0
    max_pgml_std = 0.0
    for B in batches:
        # --- simulator side ---
        theta, R = make_sim_state(B)
        sim_m, sim_med, sim_s, sim_min = time_calls(lambda: sim_step_hour(theta, R),
                                           _sync, args.reps, args.warmup)

        # --- PGML side (values irrelevant to timing; eval-mode BatchNorm uses
        #     fixed running stats, so cost depends on shape, not values) ---
        x = torch.rand(B, INPUT_DIM, device=tdev, dtype=torch.float32)

        def pgml_fn():
            with torch.no_grad():
                model(x)

        pgml_m, pgml_med, pgml_s, pgml_min = time_calls(pgml_fn, torch_sync, args.reps, args.warmup)

        factor = sim_m / pgml_m if pgml_m > 0 else float("nan")
        sim_uscol = (sim_m / B) * 1e6
        pgml_uscol = (pgml_m / B) * 1e6
        max_sim_std = max(max_sim_std, sim_s)
        max_pgml_std = max(max_pgml_std, pgml_s)

        print(f"{B:>6}"
              f"{sim_m * 1e3:>14.3f}{sim_med * 1e3:>12.3f}{sim_min * 1e3:>12.3f}"
              f"{pgml_m * 1e3:>14.3f}{pgml_med * 1e3:>13.3f}{pgml_min * 1e3:>13.3f}"
              f"{factor:>11.1f}x{sim_uscol:>12.2f}{pgml_uscol:>12.3f}")

    print("-" * 100)
    print(f"max spread (std) over reps:  sim {max_sim_std * 1e3:.3f} ms   "
          f"PGML {max_pgml_std * 1e3:.3f} ms")
    print("\nRead: 'PGML faster' = sim mean / PGML mean per call. 'us/col' = per")
    print("column (mean time / B). med=median, min=latency floor (best-case).")
    print("Precision differs by design: sim float64, PGML float32 (see header).")
    print("=" * 100)


if __name__ == "__main__":
    main()
