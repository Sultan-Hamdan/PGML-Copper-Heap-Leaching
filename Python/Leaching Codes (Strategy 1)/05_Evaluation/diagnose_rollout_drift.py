"""
diagnose_rollout_drift.py
=========================
Phase 4 - EVALUATION (05_Evaluation). Diagnostic for the PGML rollout drift
seen in benchmark_rollout.py (MAE rises to ~0.26 by day 10, then flatlines).

Runs four checks for B=1, constant Q=24 L/h, 25 days, reusing the same
loaders and simulator as benchmark_rollout.py:

  CHECK 1 - output range (the cheap clipping test)
      Free-runs the PGML chain and records the min and max of Se across the
      221 nodes at every one of the 600 steps. If Se stays inside [0,1], the
      missing clamp in the rollout loop cannot be the cause of the drift.

  CHECK 2 - one-step error on TRUE inputs (the key test)
      At each step, feeds PGML the simulator's true Se from the previous step
      and asks for one step ahead, then compares to the simulator's true Se.
      This is the error PGML makes when its input is correct. If it stays
      tiny all the way to day 25, the big free-run drift is only PGML feeding
      on its own errors. If it grows late, PGML never learned that region.

  CHECK 3 - settled or not
      Reports how much each trajectory still changes per step over the last
      day. Small means it has reached a fixed profile (a steady state).

  CHECK 4 - where the gap is
      At day 25, compares the PGML free-run profile to the simulator profile
      by depth: node of largest gap, and mean gap in the top, middle, and
      bottom third of the column (node 0 = surface).

Also reproduces the free-run drift curve as an anchor against the benchmark.

Does NOT modify any other file. Does NOT retrain.

Usage (from 05_Evaluation, VS Code terminal / PowerShell):
    python diagnose_rollout_drift.py
Optional:
    --ckpt PATH   --scaler PATH

Origin: drift diagnosis (14 Jun 2026), follow-up to benchmark_rollout.py A2
Status: diagnostic, throwaway
Depends on: benchmark_rollout.py (SIM, NZ1, N_STEPS, load_pgml,
            build_pgml_inputs, sim_rollout), preprocess.py (load_scalers),
            torch, numpy

Changelog
---------
2026.06.14  v1.0  Four-check rollout drift diagnostic: output range,
                  one-step error on true inputs, settled check, gap by depth.
"""

import os
import argparse
import numpy as np
import torch

# importing benchmark_rollout also sets sys.path for the sibling folders
from benchmark_rollout import (SIM, NZ1, N_STEPS,
                               load_pgml, build_pgml_inputs, sim_rollout)
from preprocess import load_scalers


_HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser(
        description="PGML rollout drift diagnostic, B=1 -- CENG0057")
    p.add_argument("--ckpt",
                   default=os.path.join(_HERE, "sweep_ckpts", "pgml_lam100.pt"))
    p.add_argument("--scaler",
                   default=os.path.join(_HERE, "..", "03_Preprocessing", "scaler.json"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("PGML ROLLOUT DRIFT DIAGNOSTIC  --  B=1, constant Q=24, 25 days")
    print("=" * 70)
    print(f"Device : {device}")

    if not os.path.isfile(args.ckpt):
        print(f"ERROR: checkpoint not found: {args.ckpt}")
        return
    if not os.path.isfile(args.scaler):
        print(f"ERROR: scaler.json not found: {args.scaler}")
        return

    model = load_pgml(args.ckpt, device)
    scalers = load_scalers(os.path.dirname(args.scaler))
    R_scaled, soil_vec = build_pgml_inputs(scalers)
    print(f"R scaled : {R_scaled:.6f}")
    print(f"soil vec : {soil_vec}  (zeros expected for Strategy 1)")

    # fixed suffix (R + soil), same every step
    suffix = torch.zeros(1, 5, device=device, dtype=torch.float32)
    suffix[0, 0] = R_scaled
    suffix[0, 1:] = torch.tensor(soil_vec, device=device, dtype=torch.float32)

    # initial state: uniform theta_i -> Se
    Se0 = (SIM["theta_i"] - SIM["theta_r"]) / (SIM["theta_s"] - SIM["theta_r"])
    Se_init = torch.full((1, NZ1), Se0, device=device, dtype=torch.float32)
    print(f"Se start : {Se0:.6f} (uniform)")

    # simulator truth for this exact schedule
    Se_sim = sim_rollout(1)[0]           # (221, 600) float64
    print(f"Sim done : Se_sim shape {Se_sim.shape}")

    # ---------------- free-running PGML, record min/max -----------------
    traj = np.empty((NZ1, N_STEPS), dtype=np.float64)
    minv = np.empty(N_STEPS)
    maxv = np.empty(N_STEPS)
    Se_cur = Se_init.clone()
    with torch.no_grad():
        for t in range(N_STEPS):
            x = torch.cat([Se_cur, suffix], dim=1)
            Se_cur = model(x)
            minv[t] = float(Se_cur.min())
            maxv[t] = float(Se_cur.max())
            traj[:, t] = Se_cur[0].cpu().numpy().astype(np.float64)

    # ---------- one-step error on TRUE inputs (the key test) ------------
    tf_mae = np.empty(N_STEPS, dtype=np.float64)
    with torch.no_grad():
        for t in range(N_STEPS):
            if t == 0:
                inp = Se_init
            else:
                prev = Se_sim[:, t - 1].astype(np.float32)[None, :]
                inp = torch.tensor(prev, device=device, dtype=torch.float32)
            x = torch.cat([inp, suffix], dim=1)
            pred = model(x)[0].cpu().numpy().astype(np.float64)
            tf_mae[t] = np.mean(np.abs(pred - Se_sim[:, t]))

    # ---------------- free-run drift curve (anchor) ---------------------
    free_mae = np.mean(np.abs(traj - Se_sim), axis=0)   # (600,)

    DH = SIM["D_H"]
    def day_mean(arr, d):
        return float(arr[d * DH:(d + 1) * DH].mean())

    # ============================ REPORT ===============================
    print("\n" + "-" * 70)
    print("CHECK 1 - PGML output range over the free run (clipping test)")
    print("-" * 70)
    print(f"min Se over run : {minv.min():.6f}")
    print(f"max Se over run : {maxv.max():.6f}")
    out_lo = np.where(minv < -1e-3)[0]
    out_hi = np.where(maxv > 1.0 + 1e-3)[0]
    first_lo = int(out_lo[0]) if out_lo.size else -1
    first_hi = int(out_hi[0]) if out_hi.size else -1
    print(f"first step below -1e-3  : {first_lo}  (-1 = never)")
    print(f"first step above 1+1e-3 : {first_hi}  (-1 = never)")
    if first_lo == -1 and first_hi == -1:
        print("VERDICT: Se stays in [0,1]. Missing clamp does not explain drift.")
    else:
        print("VERDICT: Se leaves [0,1]. The clamp test is worth running next.")

    print("\n" + "-" * 70)
    print("CHECK 2 - one-step error on TRUE inputs (key test)")
    print("-" * 70)
    print(f"  {'Day':>4}  {'one-step MAE':>14}  {'free-run MAE':>14}")
    for d in (0, 4, 8, 9, 14, 19, 24):
        print(f"  {d+1:>4}  {day_mean(tf_mae, d):>14.6e}  {day_mean(free_mae, d):>14.6e}")
    print("Read: one-step MAE small everywhere -> drift is self-feeding only.")
    print("      one-step MAE grows late       -> that region was never learned.")

    print("\n" + "-" * 70)
    print("CHECK 3 - settled or not (max per-step change over last day)")
    print("-" * 70)
    sim_chg = float(np.abs(np.diff(Se_sim[:, -DH:], axis=1)).max())
    pgml_chg = float(np.abs(np.diff(traj[:, -DH:], axis=1)).max())
    print(f"sim  max change : {sim_chg:.6e}")
    print(f"PGML max change : {pgml_chg:.6e}")
    print("Read: small means the profile has reached a fixed (steady) state.")

    print("\n" + "-" * 70)
    print("CHECK 4 - where the day-25 gap sits (node 0 = surface)")
    print("-" * 70)
    gap = np.abs(traj[:, -1] - Se_sim[:, -1])     # (221,)
    n = gap.shape[0]
    a = n // 3
    b = 2 * n // 3
    print(f"node of largest gap : {int(gap.argmax())} / {n - 1}")
    print(f"largest gap value   : {gap.max():.6f}")
    print(f"mean gap top third  : {gap[:a].mean():.6f}")
    print(f"mean gap mid third  : {gap[a:b].mean():.6f}")
    print(f"mean gap bot third  : {gap[b:].mean():.6f}")

    print("\n" + "-" * 70)
    print("ANCHOR - free-run drift (should match the benchmark)")
    print("-" * 70)
    print(f"max free-run MAE  : {free_mae.max():.6e}  at step {int(free_mae.argmax())}")
    print(f"final free-run MAE: {free_mae[-1]:.6e}")
    print("=" * 70)


if __name__ == "__main__":
    main()
