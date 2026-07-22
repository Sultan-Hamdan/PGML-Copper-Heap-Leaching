"""
check_steadystate_coverage.py
=============================
Phase 4 - EVALUATION (05_Evaluation). Coverage check tied to the rollout
drift diagnosis.

The rollout benchmark drives the column to the steady state of a constant
Q=24 L/h schedule held for 25 days. This script asks one question:

    Does the training data (strategy1.h5) ever contain Se profiles close
    to that constant-Q steady state?

If not, the steady-state region is outside the training distribution, which
would explain why PGML mispredicts there during a long free run.

What it does:
  1. Regenerates the constant Q=24, 25-day simulator run and takes its
     final-hour Se profile as the target steady state.
  2. Loads all 201 training Se profiles (201 runs x 600 hourly steps).
  3. Finds the single training profile closest to the target (min MAE),
     and reports that distance plus which run and step it came from.
  4. Reports a few Q-schedule context numbers (how close any training
     schedule gets to constant 24).

Read of the result:
  small min distance (say < 0.02) -> steady state IS covered; a coverage
      gap is not the story, drift is self-feeding error during the free run.
  large min distance (say > 0.10) -> steady state is NOT covered; coverage
      gap is real and explains the late-run error.

Does NOT modify any other file.

Usage (from 05_Evaluation, VS Code terminal / PowerShell):
    python check_steadystate_coverage.py --h5 "PATH/TO/strategy1.h5"

Origin: drift diagnosis (14 Jun 2026), companion to diagnose_rollout_drift.py
Status: diagnostic, throwaway
Depends on: sim_gpu.py (run_batch, ON_GPU), h5py, numpy

Changelog
---------
2026.06.14  v1.0  Nearest-training-profile distance to the constant-Q
                  steady state, plus Q-schedule context.
"""

import os
import sys
import argparse
import numpy as np
import h5py

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, "..", "01_Simulator"))

from sim_gpu import run_batch, ON_GPU

try:
    import cupy as cp
except Exception:
    cp = None

# Strategy 1 fixed soil (matches generate_dataset.py and sim_gpu defaults)
PARS = dict(Ks=170.0, theta_s=0.33, theta_r=0.0, theta_i=0.14,
            alpha=0.035, n=2.267, H=5.50, Area=308.0, tf=25,
            Dz=2.5, Dt=1.0 / 24.0, D_H=24, n_iter=20)
Q_CONST = 24.0


def main():
    p = argparse.ArgumentParser(
        description="Does training data cover the constant-Q steady state?")
    p.add_argument("--h5", required=True, help="path to strategy1.h5")
    args = p.parse_args()

    if not os.path.isfile(args.h5):
        print(f"ERROR: h5 not found: {args.h5}")
        return

    print("=" * 70)
    print("STEADY-STATE COVERAGE CHECK  --  constant Q=24 target vs training")
    print("=" * 70)
    print(f"Backend : {'GPU' if ON_GPU else 'CPU'}")

    # 1. target steady state: constant Q=24 for 25 days, final-hour profile
    Q = np.full((1, PARS["tf"]), Q_CONST, dtype=np.float64)
    Se = run_batch(Q, Ks=PARS["Ks"], theta_s=PARS["theta_s"],
                   theta_r=PARS["theta_r"], theta_i=PARS["theta_i"],
                   alpha=PARS["alpha"], n=PARS["n"], H=PARS["H"],
                   Area=PARS["Area"], tf=PARS["tf"], Dz=PARS["Dz"],
                   Dt=PARS["Dt"], D_H=PARS["D_H"], n_iter=PARS["n_iter"])
    Se = cp.asnumpy(Se) if ON_GPU else Se
    target = Se[0][:, -1]                     # (221,) float64
    print(f"target steady profile : Se in [{target.min():.4f}, {target.max():.4f}], "
          f"mean {target.mean():.4f}")

    # 2. load all training Se profiles
    with h5py.File(args.h5, "r") as f:
        names = sorted([k for k in f.keys()])
        n_runs = len(names)
        nz = f[names[0]]["Se_out"].shape[0]
        nt = f[names[0]]["Se_out"].shape[1]
        Se_all = np.empty((n_runs, nz, nt), dtype=np.float64)
        Q_all = np.empty((n_runs, PARS["tf"]), dtype=np.float64)
        for i, nm in enumerate(names):
            Se_all[i] = f[nm]["Se_out"][:]
            Q_all[i] = f[nm]["Q_schedule"][:]
    print(f"loaded {n_runs} runs, Se_out {nz} x {nt}")

    # 3. nearest training profile to the target
    flat = Se_all.transpose(0, 2, 1).reshape(-1, nz)     # (n_runs*nt, 221)
    d = np.mean(np.abs(flat - target[None, :]), axis=1)  # (n_runs*nt,)
    imin = int(d.argmin())
    run_i = imin // nt
    step_i = imin % nt
    print("\n" + "-" * 70)
    print("NEAREST training profile to the constant-Q steady state")
    print("-" * 70)
    print(f"min MAE distance : {d[imin]:.6f}")
    print(f"from run         : {names[run_i]}  step {step_i} (hour {step_i + 1})")
    print(f"that run Q sched : {Q_all[run_i]}")
    if d[imin] < 0.02:
        print("VERDICT: steady state IS covered. Coverage gap is not the cause.")
    elif d[imin] > 0.10:
        print("VERDICT: steady state NOT covered. Coverage gap is real.")
    else:
        print("VERDICT: borderline. Report the number, judge in context.")

    # 4. Q-schedule context
    qmean = np.mean(np.abs(Q_all - Q_CONST), axis=1)
    cq = int(qmean.argmin())
    print("\n" + "-" * 70)
    print("Q-schedule context (target is constant 24 all 25 days)")
    print("-" * 70)
    print(f"closest schedule : {names[cq]}  mean|Q-24| = {qmean[cq]:.3f}")
    print(f"its Q schedule   : {Q_all[cq]}")
    print(f"global Q range   : [{Q_all.min():.1f}, {Q_all.max():.1f}]")
    print("=" * 70)


if __name__ == "__main__":
    main()
