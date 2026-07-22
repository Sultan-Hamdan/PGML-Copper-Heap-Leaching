"""
pgml_closed_loop.py
===================
PGML closed-loop deliverable (S2), SELF-CONTAINED. Runs the PGML closed loop:
the surrogate is the internal model, drives irrigation R to the setpoint Se_SP
over tf days, and records the saturation-vs-depth-vs-time surface, the R
trajectory, and the realised Eq. 32 tracking cost per day. The surface is the
analogue of Olivares Fig 7.

Depends ONLY on:
    ..\\01_Simulator\\sim_gpu.py    (run_batch with theta0 + the plant)
    ..\\03_Preprocessing\\scaler.json + preprocess.py (scaler helpers, dims)
    ..\\04_Training\\train_pgml.py   (MLP, optional; rebuilt locally if import fails)
    torch, numpy, scipy
No dependency on the parked Gardner baseline (driver/controller/casadi).

Outputs (next to this script):
    closed_loop_surface.mat   Se_full_mat (221 x tf*24), Z_vect (cm), t_days (days),
                              R_cmday, eq32, Se_SP  -- variable names match the
                              MATLAB Sim_column.m
    closed_loop_surface.csv   long format depth_cm,hour,day_frac,Se
    closed_loop_traj.csv      day,R_cmday,eq32,mean_Se
    plot_closed_loop.m        MATLAB surf of the Se(z,t) surface

Run from 06_Control/:
    python pgml_closed_loop.py
    python pgml_closed_loop.py --tf 25 --candidates 200 --se_sp 0.70 --theta_i 0.14

Changelog
---------
2026.07.22  v0.3  Renamed run_clite.py -> pgml_closed_loop.py and removed the
                  "clite" naming throughout. Outputs are now
                  closed_loop_surface.mat/.csv, closed_loop_traj.csv, and
                  plot_closed_loop.m. Default --ckpt is now pgml_lam0p4.pt (the
                  shipped model). Run-location note updated to 06_Control. No
                  change to the control logic.
2026.07.19  v0.2  Made self-contained (removed driver/controller/casadi imports
                  that broke the PGML-only run). Hourly surface export.
2026.07.19  v0.1  Initial closed-loop runner.
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio

_HERE = Path(__file__).resolve().parent
for _sib in ("01_Simulator", "03_Preprocessing", "04_Training"):
    sys.path.insert(0, str(_HERE.parent / _sib))

import torch
from preprocess import INPUT_DIM, OUTPUT_DIM, load_scalers, apply_scaler
from sim_gpu import run_batch


def default_phys():
    return dict(theta_s=0.33, theta_r=0.0, theta_i=0.14, Ks=170.0, alpha=0.035,
                n=2.267, H=5.50, Area=308.0, Dz=2.5, Dt=1.0 / 24.0, D_H=24,
                n_iter=20, tf=25, Se_SP=0.70)


def build_candidates(n_cand=200, r_min=0.0, r_max=15.584):
    return np.linspace(r_min, r_max, n_cand)


def score_eq32(Se_pred, Se_SP):
    return np.mean((np.asarray(Se_pred) - Se_SP) ** 2, axis=-1)


def plant_advance_one_day(theta0, R_cmday, phys):
    Q = (R_cmday / 100.0) * phys["Area"]              # cm/day -> m3/day
    Se = run_batch(np.array([[Q]]), Ks=phys["Ks"], theta_s=phys["theta_s"],
                   theta_r=phys["theta_r"], theta_i=phys["theta_i"],
                   alpha=phys["alpha"], n=phys["n"], H=phys["H"],
                   Area=phys["Area"], tf=1, Dz=phys["Dz"], Dt=phys["Dt"],
                   D_H=phys["D_H"], n_iter=phys["n_iter"],
                   theta0=theta0[None, :])[0]          # (221, 24)
    Se = np.asarray(getattr(Se, "get", lambda: Se)())  # CuPy -> NumPy if on GPU
    theta_end = Se[:, -1] * (phys["theta_s"] - phys["theta_r"]) + phys["theta_r"]
    return Se[:, -1], theta_end, Se


def decide_pgml(Se_now, model, candidates, scaler, soil_vec, Se_SP, D_H, device):
    R_scaled = apply_scaler(np.asarray(candidates, dtype=np.float64), "R_out", scaler)
    n_cand = len(candidates)
    Se = torch.tensor(np.tile(Se_now, (n_cand, 1)), dtype=torch.float32, device=device)
    suffix = torch.zeros(n_cand, 5, dtype=torch.float32, device=device)
    suffix[:, 0] = torch.tensor(np.asarray(R_scaled), dtype=torch.float32, device=device)
    suffix[:, 1:] = torch.tensor(soil_vec, dtype=torch.float32, device=device)
    with torch.no_grad():
        for _h in range(D_H):
            Se = model(torch.cat([Se, suffix], dim=1)).clamp(0.0, 1.0)
    scores = score_eq32(Se.cpu().numpy(), Se_SP)
    return float(candidates[int(np.argmin(scores))])


def run_pgml(model, scaler, soil_vec, device, phys, candidates):
    Nz1 = int(phys["H"] * 100 / phys["Dz"]) + 1
    Se_now = np.full(Nz1, (phys["theta_i"] - phys["theta_r"]) /
                     (phys["theta_s"] - phys["theta_r"]))
    theta = np.full(Nz1, phys["theta_i"])
    Se_mat = np.zeros((Nz1, phys["tf"])); R_hist = []
    Se_hourly = np.zeros((Nz1, phys["tf"] * phys["D_H"]))
    for day in range(phys["tf"]):
        R = decide_pgml(Se_now, model, candidates, scaler, soil_vec,
                        phys["Se_SP"], phys["D_H"], device)
        Se_end, theta, Se_day = plant_advance_one_day(theta, R, phys)
        Se_mat[:, day] = Se_end; R_hist.append(R)
        Se_hourly[:, day * phys["D_H"]:(day + 1) * phys["D_H"]] = Se_day
        Se_now = Se_end
    return Se_mat, np.array(R_hist), Se_hourly


def load_mlp(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    w, d = ck["args"]["width"], ck["args"]["depth"]
    try:
        from train_pgml import MLP
        model = MLP(INPUT_DIM, OUTPUT_DIM, width=w, depth=d)
    except Exception:
        import torch.nn as nn

        class MLP(nn.Module):
            def __init__(self, i, o, width, depth):
                super().__init__()
                L = [nn.Linear(i, width), nn.BatchNorm1d(width), nn.ReLU()]
                for _ in range(depth - 1):
                    L += [nn.Linear(width, width), nn.BatchNorm1d(width), nn.ReLU()]
                L.append(nn.Linear(width, o))
                self.net = nn.Sequential(*L)

            def forward(self, x):
                return self.net(x)

        model = MLP(INPUT_DIM, OUTPUT_DIM, w, d)
    model.load_state_dict(ck["model_state"])
    return model.to(device).eval(), ck["args"].get("lam", None)


def main():
    p = argparse.ArgumentParser(description="PGML closed loop -- CENG0057")
    p.add_argument("--ckpt", default=str(_HERE / "pgml_lam0p4.pt"))
    p.add_argument("--scaler", default=str(_HERE.parent / "03_Preprocessing"))
    p.add_argument("--tf", type=int, default=25)
    p.add_argument("--candidates", type=int, default=200)
    p.add_argument("--se_sp", type=float, default=0.70)
    p.add_argument("--theta_i", type=float, default=0.14)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("PGML CLOSED LOOP")
    print("=" * 70)
    print(f"device      : {device}")
    if not os.path.isfile(args.ckpt):
        print(f"ERROR: checkpoint not found: {args.ckpt}. Pass --ckpt. Aborting."); return
    model, lam = load_mlp(args.ckpt, device)
    scaler = load_scalers(args.scaler)
    phys = default_phys(); phys["tf"] = args.tf; phys["theta_i"] = args.theta_i
    phys["Se_SP"] = args.se_sp
    candidates = build_candidates(args.candidates)

    soil = np.array([
        apply_scaler(np.array([phys["Ks"]]),      "Ks",      scaler)[0],
        apply_scaler(np.array([phys["alpha"]]),   "alpha",   scaler)[0],
        apply_scaler(np.array([phys["n"]]),       "n_vgm",   scaler)[0],
        apply_scaler(np.array([phys["theta_i"]]), "theta_i", scaler)[0],
    ], dtype=np.float32)
    print(f"lambda={lam}  setpoint Se_SP={args.se_sp}  theta_i={args.theta_i}  "
          f"candidates={args.candidates}  days={args.tf}")
    print(f"soil vec: {np.round(soil, 4)}  (Ks/alpha/n fixed -> 0; theta_i scaled)\n")

    Se, R, Se_hourly = run_pgml(model, scaler, soil, device, phys, candidates)

    eq32 = np.array([score_eq32(Se[:, d], args.se_sp) for d in range(args.tf)])
    depth_cm = np.arange(0.0, phys["H"] * 100.0 + 1e-9, phys["Dz"])
    days = np.arange(1, args.tf + 1)
    n_hours = Se_hourly.shape[1]
    t_days_hourly = np.arange(1, n_hours + 1) / phys["D_H"]

    print(f"{'day':>4}{'R cm/d':>9}{'mean Se':>9}{'Eq32':>11}")
    for d in range(args.tf):
        print(f"{d+1:>4}{R[d]:>9.2f}{Se[:, d].mean():>9.4f}{eq32[d]:>11.3e}")
    print(f"\nfinal mean Se = {Se[:, -1].mean():.4f}   final Eq32 = {eq32[-1]:.3e}")
    print(f"cumulative Eq32 = {eq32.sum():.4e}")

    sio.savemat(str(_HERE / "closed_loop_surface.mat"),
                {"Se_full_mat": Se_hourly, "Z_vect": depth_cm.reshape(-1, 1),
                 "t_days": t_days_hourly.reshape(1, -1), "Se_daily": Se,
                 "day": days, "R_cmday": np.asarray(R), "eq32": eq32,
                 "Se_SP": float(args.se_sp), "theta_r": float(phys["theta_r"]),
                 "theta_s": float(phys["theta_s"])}, format="5")
    with open(_HERE / "closed_loop_surface.csv", "w") as fh:
        fh.write("depth_cm,hour,day_frac,Se\n")
        for h in range(n_hours):
            for i in range(Se_hourly.shape[0]):
                fh.write(f"{depth_cm[i]:.2f},{h+1},{t_days_hourly[h]:.5f},{Se_hourly[i, h]:.6f}\n")
    with open(_HERE / "closed_loop_traj.csv", "w") as fh:
        fh.write("day,R_cmday,eq32,mean_Se\n")
        for d in range(args.tf):
            fh.write(f"{d+1},{R[d]:.6f},{eq32[d]:.6e},{Se[:, d].mean():.6f}\n")
    with open(_HERE / "plot_closed_loop.m", "w") as fh:
        fh.write(
            "% plot_closed_loop.m  -- PGML closed-loop Se(z,t) surface\n"
            "% Mirrors the Figure 1 block of the MATLAB Sim_column.m.\n"
            "load('closed_loop_surface.mat');\n"
            "[T_full, Z_full] = meshgrid(t_days, Z_vect);\n"
            "figure\n"
            "surf(T_full, Z_full, Se_full_mat, 'EdgeColor', 'none');\n"
            "xlabel('Time (days)')\n"
            "ylabel('Axial position z (cm)')\n"
            "zlabel('Effective saturation S_e (-)')\n"
            "title('S_e(z,t) -- PGML closed loop')\n"
            "grid on\n"
            "colorbar\n")

    print("\nWrote closed_loop_surface.mat, closed_loop_surface.csv, "
          "closed_loop_traj.csv, plot_closed_loop.m")
    print("In MATLAB (same folder): run  plot_closed_loop")


if __name__ == "__main__":
    main()
