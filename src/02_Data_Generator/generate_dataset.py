"""
generate_dataset.py
===================
Pipeline driver for dataset generation. Draws 4096 designs from a scrambled
Sobol sequence over the joint feasible box, builds an irrigation schedule for
each one under the Olivares 2025 multiplicative rate band with a per-soil upper
limit, runs all simulations through the GPU batch runner in VRAM-safe tiles, and
saves the results to a single HDF5 file.

Soil varies per run as well as the schedule, which is what makes this a
generalisation dataset. The box was fixed by a nested-inflation study and is not
a command line option:

    Ks     [130, 232]   cm/day
    n      [2.2, 3.0]   van Genuchten
    alpha  [0.025, 0.042] 1/cm
    dtheta [0.27, 0.388] theta_s, with theta_r = 0
    Se_i   [0.40, 0.63]  initial effective saturation

Every point in the box, corners included, integrates at the operational cap
R/Ks = 0.30, so a plain Sobol draw inside it is safe and no in-loop rejection
is needed. The finite guard is kept as insurance.

DESIGN: 7 Sobol dimensions
    dim 0 Ks       dim 1 n        dim 2 alpha    dim 3 dtheta
    dim 4 Se_i     dim 5 qf       dim 6 sigma
  theta_i is NOT a dimension: theta_i = Se_i * dtheta per run.
  theta_r is NOT a dimension: it is the zero datum.
  R/Ks is NOT a dimension: it is the cap. Qbar is drawn as a fraction qf
    of that soil's own ceiling, Qbar = qf * Q_ub(Ks), so R/Ks <= 0.30 holds by
    construction for every soil.

Steps:
    1. Draw 4096 Sobol points in 7 dimensions, scale to the box
    2. theta_i = Se_i * dtheta; Q_ub = 0.30 * Ks * Area / 100 per run
    3. Build a 25-day Q-schedule per run inside the per-soil multiplicative band
    4. Run all schedules through column_solver.run_batch() in VRAM-safe tiles
    5. Abort if any run is non-finite (nothing written)
    6. Save all runs into dataset.h5 (one group per run, per-run soil stored)
    7. Print a timing summary and sanity check on first and last run

Usage, from 02_Data_Generator:
    python generate_dataset.py

--out_dir defaults to the data/ folder in the repository root. Give it
explicitly to write elsewhere:
    python generate_dataset.py --out_dir "D:/somewhere else"

N_RUNS and SEED are fixed design values, not command line options. A run count
that is a power of two is required for Sobol's stratification guarantee.

Output file: dataset.h5
    |-- run_0000/
    |   |-- Se_out       (221 x 600)
    |   |-- R_out        (600,)
    |   |-- q_out        (600,)
    |   |-- Q_schedule   (25,)
    |   attrs: Ks, n_vgm, alpha, theta_s, theta_r, theta_i, Se_i, Qbar, sigma
    |-- run_0001/ ... run_4095/
    File-level attrs: strategy, n_runs, seed, tf, area, box bounds

Status: stable
Depends on: column_solver.py

Changelog
---------
2026.09.07  v2.3  Repository release. Import updated to column_solver, output
                  renamed to dataset.h5, --out_dir now defaults to data/.
2026.07.28  v3.0  Soil now varies per run over the fixed box. Sobol 3 dims ->
                  7 dims (added Ks, n, alpha, dtheta; Se_i replaces raw
                  theta_i). Qbar drawn as a per-soil fraction qf of
                  Q_ub(Ks) = 0.30*Ks*Area/100, so the rate band upper limit is
                  per-run and R/Ks <= 0.30 holds by construction.
                  compute_q_out takes per-run Ks and n, previously global
                  constants. Runs go through a VRAM-safe tile loop, since 4096
                  will not fit one batch on 8.55 GB. Per-run soil written to
                  each group; there are no file-level soil attrs.
2026.07.17  v2.1  assert_all_finite(). A saturating run produces NaN and was
                  previously written silently. Now reports every dead run with
                  its design point and aborts before saving.
2026.07.17  v2.0  Replaced the file reader with a scrambled Sobol sampler
                  (512 runs, 3 dims) and the Olivares 2025 multiplicative band.
                  theta_i per-run. Renamed file attr "n" to "n_vgm".
2026.06.09  v1.0  Initial pipeline, 201 runs, verified against a single-column
                  reference to ~1e-13.
"""

import sys
import os
import time
import argparse
import numpy as np
import h5py
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths
paths.add_stage_paths()

from column_solver import run_batch, _sync, ON_GPU

try:
    import cupy as cp
except ImportError:
    cp = None

# -- Fixed geometry and grid (frozen, benchmark compatibility) -----------------
THETA_R = 0.0          # zero datum
AREA    = 308.0        # m2
TF      = 25           # days
D_H     = 24           # hours per day
H       = 5.50         # m
DZ      = 2.5          # cm
DT      = 1.0 / 24.0   # days
N_ITER  = 20

# -- Feasible box. Order: Ks, n, alpha, dtheta, Se_i --------------------------
KS_LB,     KS_UB     = 130.0,  232.0
N_LB,      N_UB      = 2.20,   3.00
ALPHA_LB,  ALPHA_UB  = 0.025,  0.042
DTHETA_LB, DTHETA_UB = 0.270,  0.388
SE_I_LB,   SE_I_UB   = 0.40,   0.63

# -- Schedule design values ----------------------------------------------------
# qf: Qbar as a fraction of the per-soil ceiling Q_ub(Ks). qf in [0.10, 1.0]
#     gives R/Ks in [0.03, 0.30], covering the operational range (industrial
#     0.07 to 0.28) with margin, without over-flooring at Q_LB.
# Q_LB = 0.5: resolves the (1 - w) * 0 = 0 degeneracy in Olivares' rule.
# OMEGA = 1.0: the loosest w the baseline band permits (Olivares give w in (0,1],
#     never a value); matches the FORM of their rule.
# RKS_CEIL = 0.30: operational cap; Q_ub(Ks) = 0.30 * Ks * Area / 100.
N_RUNS     = 4096
N_DIM      = 7
SEED       = 42
QF_LB      = 0.10
QF_UB      = 1.0
SIGMA_LB   = 0.0
SIGMA_UB   = 1.0
Q_LB       = 0.5       # m3/day
OMEGA      = 1.0
RKS_CEIL   = 0.30

VRAM_GB       = 8.55
VRAM_HEADROOM = 0.40   # conservative: a one-shot generation must not OOM late


def q_ub_of_ks(Ks):
    """Per-soil irrigation ceiling from the R/Ks cap, m3/day."""
    return RKS_CEIL * Ks * AREA / 100.0


def sample_design(n_runs=N_RUNS, seed=SEED):
    """
    Draw the design and build one Q-schedule per run.

    Sobol, scrambled, 7 continuous dimensions scaled to the box:
        0 Ks [130,232]  1 n [2.2,3.0]  2 alpha [0.025,0.042]
        3 dtheta [0.27,0.388]  4 Se_i [0.40,0.63]
        5 qf [0.10,1.0]  6 sigma [0,1]

    Derived per run: theta_s = dtheta, theta_i = Se_i * dtheta (theta_r = 0),
    Q_ub = 0.30 * Ks * Area / 100, Qbar = qf * Q_ub.

    Schedule (Olivares multiplicative band, per-soil upper limit):
        Q[0] = Q_lb
        lo   = max(Q_lb, (1 - w) * Q[d-1])
        hi   = min(Q_ub, (1 + w) * Q[d-1])          # Q_ub is PER RUN
        Q[d] = clip(Qbar * exp(sigma * u), lo, hi),  u ~ U(-1, 1)

    To grow the design to 8192 later, draw 8192 with the same seed and take
    rows 4096 onward. A new generator restarts at index 0 and silently
    regenerates the same 4096 runs:
        pts = qmc.Sobol(N_DIM, scramble=True, seed=SEED).random(8192)
        new = pts[4096:]
    """
    pts = qmc.Sobol(N_DIM, scramble=True, seed=seed).random(n_runs)
    lo = [KS_LB, N_LB, ALPHA_LB, DTHETA_LB, SE_I_LB, QF_LB, SIGMA_LB]
    hi = [KS_UB, N_UB, ALPHA_UB, DTHETA_UB, SE_I_UB, QF_UB, SIGMA_UB]
    s = qmc.scale(pts, lo, hi)

    Ks     = s[:, 0]
    n_vgm  = s[:, 1]
    alpha  = s[:, 2]
    dtheta = s[:, 3]
    Se_i   = s[:, 4]
    qf     = s[:, 5]
    sigma  = s[:, 6]

    theta_s = dtheta
    theta_i = Se_i * (theta_s - THETA_R) + THETA_R   # = Se_i * dtheta
    Q_ub    = q_ub_of_ks(Ks)
    Qbar    = qf * Q_ub

    rng = np.random.default_rng(seed)
    Q_batch = np.empty((n_runs, TF), dtype=float)
    Q_batch[:, 0] = Q_LB
    for d in range(1, TF):
        u  = rng.uniform(-1.0, 1.0, size=n_runs)
        lo_d = np.maximum(Q_LB, (1.0 - OMEGA) * Q_batch[:, d - 1])
        hi_d = np.minimum(Q_ub, (1.0 + OMEGA) * Q_batch[:, d - 1])
        Q_batch[:, d] = np.clip(Qbar * np.exp(sigma * u), lo_d, hi_d)

    # R/Ks coverage, for the audit (does forcing span the operational range?)
    rks_peak = (Q_batch / AREA * 100.0).max(axis=1) / Ks

    names = [f"run_{i:04d}" for i in range(n_runs)]
    print(f"Sampled {n_runs} Sobol designs in {N_DIM}D (seed {seed})")
    print(f"  Ks      [{Ks.min():.2f}, {Ks.max():.2f}]")
    print(f"  n       [{n_vgm.min():.4f}, {n_vgm.max():.4f}]")
    print(f"  alpha   [{alpha.min():.5f}, {alpha.max():.5f}]")
    print(f"  dtheta  [{dtheta.min():.4f}, {dtheta.max():.4f}]")
    print(f"  Se_i    [{Se_i.min():.4f}, {Se_i.max():.4f}]")
    print(f"  sigma   [{sigma.min():.4f}, {sigma.max():.4f}]")
    print(f"  Q       [{Q_batch.min():.2f}, {Q_batch.max():.2f}]  m3/day")
    print(f"  R/Ks pk [{rks_peak.min():.3f}, {rks_peak.max():.3f}]  (cap {RKS_CEIL})")

    soil = dict(Ks=Ks, n_vgm=n_vgm, alpha=alpha, theta_s=theta_s,
                theta_i=theta_i, Se_i=Se_i, Qbar=Qbar, sigma=sigma)
    return names, soil, Q_batch


def measure_tile(nz_plus1, hours, dtype_bytes=8):
    per_run = nz_plus1 * hours * dtype_bytes
    budget = VRAM_GB * (1024 ** 3) * VRAM_HEADROOM
    return max(1, int(budget // per_run))


def run_all_tiled(names, soil, Q_batch):
    """Run every design through run_batch in VRAM-safe tiles, one batched call
    per tile. Returns Se_all (n_runs, Nz+1, hours) on CPU."""
    n_runs = len(names)
    Nz = int(H * 100 / DZ)
    hours = TF * D_H
    tile = measure_tile(Nz + 1, hours)
    n_tiles = int(np.ceil(n_runs / tile))
    print(f"\nRunning {n_runs} simulations in {n_tiles} VRAM tile(s) "
          f"(<= {tile} runs/tile)...")

    Se_all = np.empty((n_runs, Nz + 1, hours), dtype=float)
    t0 = time.perf_counter()
    for start in range(0, n_runs, tile):
        sl = slice(start, min(start + tile, n_runs))
        Se = run_batch(
            Q_batch[sl],
            Ks=soil["Ks"][sl], theta_s=soil["theta_s"][sl],
            theta_r=np.full(sl.stop - sl.start, THETA_R),
            theta_i=soil["theta_i"][sl], alpha=soil["alpha"][sl],
            n=soil["n_vgm"][sl], H=H, Area=AREA, tf=TF, Dz=DZ, Dt=DT,
            D_H=D_H, n_iter=N_ITER,
        )
        Se_all[sl] = cp.asnumpy(Se) if ON_GPU else Se
        print(f"  tile {start}-{sl.stop} / {n_runs} done  "
              f"{time.perf_counter() - t0:.1f}s", flush=True)
    _sync()
    return Se_all


def assert_all_finite(names, Se_all, soil):
    """Abort if any run is non-finite. Nothing is written. Every point in the
    box integrates at the cap, so zero dead runs is expected and this is
    insurance. A dead run means the box or the grid assumption is wrong, not
    that one design was unlucky."""
    bad = ~np.isfinite(Se_all).all(axis=(1, 2))
    if not bad.any():
        print(f"Finite check  : all {len(names)} runs finite")
        return
    idx = np.flatnonzero(bad)
    print(f"\n{'=' * 62}")
    print(f"ABORT: {len(idx)} of {len(names)} runs are non-finite")
    print(f"{'=' * 62}")
    print(f"  {'run':10s} {'Ks':>7s} {'n':>6s} {'alpha':>7s} {'dtheta':>7s} {'Se_i':>6s}")
    for i in idx:
        print(f"  {names[i]:10s} {soil['Ks'][i]:7.2f} {soil['n_vgm'][i]:6.3f} "
              f"{soil['alpha'][i]:7.4f} {soil['theta_s'][i]:7.4f} {soil['Se_i'][i]:6.3f}")
    print("\n  Every point in the box was verified to integrate at the cap.")
    print("  A dead run falsifies that for this grid. Investigate the design")
    print("  point, do not drop it.")
    print("  Nothing was saved.")
    raise RuntimeError(f"{len(idx)} non-finite runs. See the design points above.")


def compute_r_out(Q_schedule):
    """Daily Q (m3/day) -> hourly R (cm/day), repeated 24x per day."""
    return np.repeat((Q_schedule / AREA) * 100.0, D_H)


def compute_q_out(Se_run, Ks, n):
    """Bottom-node outflow proxy from Se surface, PER-RUN Ks and n."""
    m_vgm = 1.0 - 2.0 / n
    delta = 3.0 + 2.0 / (m_vgm * n)
    return Ks * Se_run[-1, :] ** delta * (AREA / 100.0)


def save_all_h5(out_dir, names, Se_all, Q_batch, soil, strategy="3"):
    """Save all runs into one HDF5 file, one group per run, soil PER RUN.
    No file-level soil attrs on purpose."""
    path = os.path.join(out_dir, "dataset.h5")
    with h5py.File(path, "w") as f:
        f.attrs["strategy"] = strategy
        f.attrs["n_runs"]   = len(names)
        f.attrs["seed"]     = SEED
        f.attrs["tf"]       = TF
        f.attrs["area"]     = AREA
        f.attrs["theta_r"]  = THETA_R
        f.attrs["box_Ks"]     = [KS_LB, KS_UB]
        f.attrs["box_n"]      = [N_LB, N_UB]
        f.attrs["box_alpha"]  = [ALPHA_LB, ALPHA_UB]
        f.attrs["box_dtheta"] = [DTHETA_LB, DTHETA_UB]
        f.attrs["box_Se_i"]   = [SE_I_LB, SE_I_UB]
        for i, name in enumerate(names):
            grp = f.create_group(name)
            grp.create_dataset("Se_out", data=Se_all[i], compression="gzip")
            grp.create_dataset("R_out", data=compute_r_out(Q_batch[i]))
            grp.create_dataset("q_out",
                               data=compute_q_out(Se_all[i], soil["Ks"][i],
                                                  soil["n_vgm"][i]))
            grp.create_dataset("Q_schedule", data=Q_batch[i])
            grp.attrs["Ks"]      = float(soil["Ks"][i])
            grp.attrs["n_vgm"]   = float(soil["n_vgm"][i])
            grp.attrs["alpha"]   = float(soil["alpha"][i])
            grp.attrs["theta_s"] = float(soil["theta_s"][i])
            grp.attrs["theta_r"] = THETA_R
            grp.attrs["theta_i"] = float(soil["theta_i"][i])
            grp.attrs["Se_i"]    = float(soil["Se_i"][i])
            grp.attrs["Qbar"]    = float(soil["Qbar"][i])
            grp.attrs["sigma"]   = float(soil["sigma"][i])
    size_mb = os.path.getsize(path) / 1e6
    print(f"Saved {len(names)} runs -> {path}  ({size_mb:.1f} MB)")
    return path


def sanity_check(h5_path, first_name, last_name):
    """Load first and last run back and confirm shapes, ranges, per-run soil."""
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
            print(f"  soil       Ks {grp.attrs['Ks']:.2f}  n {grp.attrs['n_vgm']:.4f}"
                  f"  alpha {grp.attrs['alpha']:.5f}  dtheta {grp.attrs['theta_s']:.4f}"
                  f"  Se_i {grp.attrs['Se_i']:.4f}")
            print(f"  sched      Qbar {grp.attrs['Qbar']:.2f}  sigma {grp.attrs['sigma']:.4f}")
            print(f"  Se_out     {Se.shape}   [{Se.min():.4f}, {Se.max():.4f}]")
            print(f"  R_out      {R.shape}   [{R.min():.3f}, {R.max():.3f}]")
            print(f"  q_out      {q.shape}   [{q.min():.3f}, {q.max():.3f}]")
            print(f"  Q_schedule {Q.shape}   [{Q.min():.1f}, {Q.max():.1f}]")
    print(f"\n  strategy={strat}  n_runs={n_runs}")


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output folder : {out_dir}")
    print(f"Backend       : {'GPU' if ON_GPU else 'CPU'}\n")

    names, soil, Q_batch = sample_design()

    t_start = time.perf_counter()
    Se_all = run_all_tiled(names, soil, Q_batch)
    t_run = time.perf_counter() - t_start
    print(f"\nAll tiles complete: {t_run:.2f} s  ({t_run/len(names)*1000:.1f} ms/run)")

    assert_all_finite(names, Se_all, soil)

    print(f"\nSaving to single HDF5 file...")
    t_save = time.perf_counter()
    h5_path = save_all_h5(out_dir, names, Se_all, Q_batch, soil, strategy="3")
    t_save = time.perf_counter() - t_save

    t_total = t_run + t_save
    print(f"\n{'=' * 50}")
    print("SUMMARY")
    print(f"{'=' * 50}")
    print(f"Runs generated : {len(names)}")
    print(f"Sim time       : {t_run:.2f} s  ({t_run/len(names)*1000:.1f} ms/run)")
    print(f"Save time      : {t_save:.1f} s")
    print(f"Total time     : {t_total:.2f} s")
    print(f"Output file    : {h5_path}")

    print(f"\nSanity check:")
    sanity_check(h5_path, names[0], names[-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the PGML heap leaching dataset as HDF5")
    parser.add_argument("--out_dir", default=None,
                        help="Output folder for dataset.h5. Default: data/")
    args = parser.parse_args()

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(paths.REPO_ROOT, "data")
        os.makedirs(out_dir, exist_ok=True)

    main(out_dir)
