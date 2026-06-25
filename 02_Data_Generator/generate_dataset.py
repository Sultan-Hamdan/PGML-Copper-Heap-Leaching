"""
generate_dataset.py
===================
Pipeline driver for Strategy 1 dataset generation. Reads 201 Q-schedules
from existing MATLAB .mat files, runs all 201 simulations through the GPU
batch runner in one pass, and saves the results to a single HDF5 file.

Steps:
    1. Read Q-schedules from run_XXXX.mat files (via --mat_dir)
    2. Run all schedules through sim_gpu.run_batch() in one GPU batch
    3. Save all runs into strategy1.h5 (one group per run)
    4. Print a timing summary and sanity check on first and last run

Usage:
    python generate_dataset.py \
        --mat_dir  "C:/path/to/Matlab dataset" \
        --out_dir  "C:/path/to/Python dataset"

Output file: strategy1.h5
    ├-- run_0000/
    │   ├-- Se_out       (221 x 600)
    │   ├-- R_out        (600,)
    │   ├-- q_out        (600,)
    │   └-- Q_schedule   (25,)
    ├-- run_0001/ ...
    └-- run_0200/
    File-level attrs: strategy, Ks, alpha, n, theta_i, n_runs

Origin: original (not a MATLAB port)
Status: active development
Depends on: sim_gpu.py

Changelog
---------
2026.06.09  v1.0  Initial Strategy 1 pipeline, 201 runs, verified vs MATLAB to ~1e-13
"""

import sys
import os
import glob
import time
import argparse
import numpy as np
import h5py
from scipy.io import loadmat

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "01_Simulator"))

from sim_gpu import run_batch, _sync, ON_GPU

try:
    import cupy as cp
except ImportError:
    cp = None

# -- Fixed Cariaga parameters (Strategy 1) -------------------------------------
KS      = 170.0        # cm/day
ALPHA   = 0.035        # 1/cm
N_VGM   = 2.267
THETA_I = 0.14
THETA_S = 0.33
THETA_R = 0.0
AREA    = 308.0        # m²
TF      = 25           # days
D_H     = 24           # hours per day


def load_q_schedules(mat_dir):
    """Read Q-schedules from all run_XXXX.mat files."""
    files = sorted(glob.glob(os.path.join(mat_dir, "run_*.mat")))
    if not files:
        raise FileNotFoundError(f"No run_*.mat files found in {mat_dir}")
    schedules = []
    for f in files:
        m = loadmat(f)
        qkey = next((k for k in ("Q_cariaga", "Q") if k in m), None)
        if qkey is None:
            raise KeyError(f"No Q key in {os.path.basename(f)}")
        Q = np.asarray(m[qkey], dtype=float).ravel()[:TF]
        schedules.append((os.path.basename(f).replace(".mat", ""), Q))
    print(f"Loaded {len(schedules)} Q-schedules from {mat_dir}")
    return schedules


def compute_r_out(Q_schedule):
    """Daily Q (m3/day) -> hourly R (cm/day), repeated 24x per day."""
    return np.repeat((Q_schedule / AREA) * 100.0, D_H)


def compute_q_out(Se_run):
    """Bottom node outflow proxy from Se surface."""
    m_vgm = 1.0 - 2.0 / N_VGM
    delta = 3.0 + 2.0 / (m_vgm * N_VGM)
    return KS * Se_run[-1, :] ** delta * (AREA / 100.0)


def save_all_h5(out_dir, names, Se_all, Q_batch, strategy="1"):
    """Save all runs into one HDF5 file, one group per run."""
    path = os.path.join(out_dir, f"strategy{strategy}.h5")
    with h5py.File(path, "w") as f:
        f.attrs["strategy"] = strategy
        f.attrs["Ks"]       = KS
        f.attrs["alpha"]    = ALPHA
        f.attrs["n"]        = N_VGM
        f.attrs["theta_i"]  = THETA_I
        f.attrs["n_runs"]   = len(names)
        for i, name in enumerate(names):
            grp = f.create_group(name)
            grp.create_dataset("Se_out",     data=Se_all[i], compression="gzip")
            grp.create_dataset("R_out",      data=compute_r_out(Q_batch[i]))
            grp.create_dataset("q_out",      data=compute_q_out(Se_all[i]))
            grp.create_dataset("Q_schedule", data=Q_batch[i])
    size_mb = os.path.getsize(path) / 1e6
    print(f"Saved {len(names)} runs -> {path}  ({size_mb:.1f} MB)")
    return path


def sanity_check(h5_path, first_name, last_name):
    """Load first and last run back and confirm shapes + ranges."""
    with h5py.File(h5_path, "r") as f:
        n_runs = f.attrs["n_runs"]
        strat  = f.attrs["strategy"]
        for name in (first_name, last_name):
            grp = f[name]
            Se  = grp["Se_out"][:]
            R   = grp["R_out"][:]
            q   = grp["q_out"][:]
            Q   = grp["Q_schedule"][:]
            print(f"\n-- {name} --")
            print(f"  Se_out     {Se.shape}   [{Se.min():.4f}, {Se.max():.4f}]")
            print(f"  R_out      {R.shape}   [{R.min():.3f}, {R.max():.3f}]")
            print(f"  q_out      {q.shape}   [{q.min():.3f}, {q.max():.3f}]")
            print(f"  Q_schedule {Q.shape}   [{Q.min():.1f}, {Q.max():.1f}]")
    print(f"\n  strategy={strat}  n_runs={n_runs}")


def main(mat_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output folder : {out_dir}")
    print(f"Backend       : {'GPU' if ON_GPU else 'CPU'}\n")

    # Load Q-schedules from .mat files
    schedules = load_q_schedules(mat_dir)
    names     = [s[0] for s in schedules]
    Q_batch   = np.array([s[1] for s in schedules], dtype=float)

    # GPU batch run
    print(f"Running {len(names)} simulations...")
    t_start = time.perf_counter()
    Se_gpu = run_batch(Q_batch, heartbeat=True)
    _sync()
    t_run = time.perf_counter() - t_start
    print(f"\nGPU batch complete: {t_run:.2f} s  ({t_run/len(names)*1000:.1f} ms/run)")

    # Back to CPU
    Se_all = cp.asnumpy(Se_gpu) if ON_GPU else Se_gpu

    # Save to single HDF5
    print(f"\nSaving to single HDF5 file...")
    t_save = time.perf_counter()
    h5_path = save_all_h5(out_dir, names, Se_all, Q_batch, strategy="1")
    t_save = time.perf_counter() - t_save

    # Summary
    t_total = t_run + t_save
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Runs generated : {len(names)}")
    print(f"GPU batch time : {t_run:.2f} s  ({t_run/len(names)*1000:.1f} ms/run)")
    print(f"Save time      : {t_save:.1f} s")
    print(f"Total time     : {t_total:.2f} s")
    print(f"Output file    : {h5_path}")

    # Sanity check
    print(f"\nSanity check:")
    sanity_check(h5_path, names[0], names[-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PGML dataset as HDF5")
    parser.add_argument("--mat_dir", required=True,
                        help="Folder containing run_XXXX.mat files")
    parser.add_argument("--out_dir", required=True,
                        help="Output folder for strategy1.h5")
    args = parser.parse_args()
    main(args.mat_dir, args.out_dir)
