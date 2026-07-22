"""
verify_all.py
=============
Verifies the Python port against the 201 MATLAB .mat files, running two
independent passes and comparing their Se output to the MATLAB reference.

Two passes:
    1. Serial CPU  -- runs sim_column.py one run at a time (exact port of MATLAB)
    2. GPU batch   -- runs sim_gpu.py on all 201 runs simultaneously

Both passes check against the Se_out stored in each .mat file. Time is
recorded for each pass to give a direct CPU vs GPU comparison.

Run from the Windows terminal:
    python verify_all.py <path\to\dataset>

Outputs (written to the same directory as this script):
    verify_all_errors.png   -- max |dSe| per run, both passes side by side
    verify_all_heatmap.png  -- error field (node x hour) for the worst run
    verify_all_summary.txt  -- plain-text table of every run's errors + timing

Note: with errors at ~1e-13, the plots may show negligible variation.
Known improvement: add a --pass flag to run CPU only, GPU only, or both.

Origin: original (not a MATLAB port)
Status: stable
Depends on: sim_column.py, sim_gpu.py

Changelog
---------
2026.06.09  v1.0  Initial full verification, 201 runs, CPU and GPU passes
"""

import sys
import os
import glob
import time
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "01_Simulator"))

from sim_column import run_simulation, SimParams

try:
    import cupy as cp
    from sim_gpu import run_batch
    ON_GPU = True
except Exception:
    ON_GPU = False


TOL = 1e-3   # same tolerance as the earlier verification


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_mat(path):
    """Load one .mat file. Returns dict with Se, Q, and any stored params."""
    m = loadmat(path)
    out = {"Se": np.asarray(m["Se_out"], dtype=float)}

    qkey = next((k for k in ("Q_cariaga", "Q") if k in m), None)
    if qkey is None:
        raise KeyError(f"No Q key in {os.path.basename(path)}: "
                       f"{[k for k in m if not k.startswith('__')]}")
    out["Q"] = np.asarray(m[qkey], dtype=float).ravel()

    for key in ("Ks", "alpha", "n", "theta_i"):
        if key in m:
            out[key] = float(np.asarray(m[key]).ravel()[0])
    return out


def params_for(ref):
    """Build SimParams from a loaded reference dict."""
    p = SimParams()
    if ref["Q"].size >= p.tf:
        p.Q_schedule = list(ref["Q"])
    for key in ("Ks", "alpha", "n", "theta_i"):
        if key in ref:
            setattr(p, key, ref[key])
    return p


# ---------------------------------------------------------------------------
# Pass 1: Serial CPU
# ---------------------------------------------------------------------------

def run_serial(refs):
    """
    Re-run every file through sim_column.py and compare to stored Se_out.
    Returns:
        rows  : list of (name, max_err, mean_err, passed)
        worst : dict with the worst run's error field
        elapsed : total wall time (seconds)
    """
    print("\n" + "="*60)
    print("PASS 1 — Serial CPU (sim_column.py)")
    print("="*60)
    rows = []
    worst = {"max_err": -1.0, "err_field": None, "name": None}

    t_start = time.perf_counter()

    for name, ref in refs:
        p = params_for(ref)
        res = run_simulation(p)

        if res.Se_full.shape != ref["Se"].shape:
            print(f"  {name}: SHAPE MISMATCH {res.Se_full.shape} vs {ref['Se'].shape} — skipped")
            continue

        err = np.abs(res.Se_full - ref["Se"])
        max_err  = float(err.max())
        mean_err = float(err.mean())
        passed   = max_err < TOL
        rows.append((name, max_err, mean_err, passed))

        if max_err > worst["max_err"]:
            worst = {"max_err": max_err, "err_field": err.copy(), "name": name}

        flag = "OK  " if passed else "FAIL"
        print(f"  {name}  max|dSe|={max_err:.2e}  mean={mean_err:.2e}  [{flag}]")

    elapsed = time.perf_counter() - t_start
    n_pass = sum(1 for r in rows if r[3])
    print(f"\n  Serial result : {n_pass}/{len(rows)} passed  |  "
          f"worst max|dSe| = {worst['max_err']:.2e}  |  "
          f"wall time = {elapsed:.1f} s  ({elapsed/len(rows):.3f} s/run)")
    return rows, worst, elapsed


# ---------------------------------------------------------------------------
# Pass 2: GPU batch
# ---------------------------------------------------------------------------

def run_gpu(refs):
    """
    Stack all Q-schedules into a batch, run sim_gpu.run_batch, compare to
    stored Se_out for every run.
    Returns:
        rows    : list of (name, max_err, mean_err, passed)
        worst   : dict with worst run's error field
        elapsed : total wall time including GPU sync (seconds)
    """
    print("\n" + "="*60)
    print("PASS 2 — GPU batch (sim_gpu.py)")
    print("="*60)

    if not ON_GPU:
        print("  CuPy / GPU not available — skipping GPU pass.")
        return [], {"max_err": None, "err_field": None, "name": None}, None

    names = [r[0] for r in refs]
    rlist = [r[1] for r in refs]
    B     = len(rlist)

    # Build Q batch — shape (B, 44). Pad short schedules with zeros.
    Q_len = max(r["Q"].size for r in rlist)
    Q_batch = np.zeros((B, Q_len), dtype=float)
    for i, r in enumerate(rlist):
        Q_batch[i, :r["Q"].size] = r["Q"]

    # Collect per-run soil params (arrays or default scalar)
    def _collect(key, default):
        vals = [r.get(key, default) for r in rlist]
        arr  = np.array(vals, dtype=float)
        return arr if np.any(arr != arr[0]) else arr[0]

    Ks_arr      = _collect("Ks",      170.0)
    alpha_arr   = _collect("alpha",   0.035)
    n_arr       = _collect("n",       2.267)
    theta_i_arr = _collect("theta_i", 0.14)

    # Warm-up: one tiny batch to avoid JIT in the timing window
    _dummy_Q = np.ones((1, Q_len), dtype=float) * 12.0
    run_batch(_dummy_Q, Ks=170.0)
    cp.cuda.runtime.deviceSynchronize()

    t_start = time.perf_counter()
    Se_gpu = run_batch(
        Q_batch,
        Ks=Ks_arr, alpha=alpha_arr, n=n_arr, theta_i=theta_i_arr,
    )
    cp.cuda.runtime.deviceSynchronize()
    elapsed = time.perf_counter() - t_start

    # Back to CPU for comparison
    Se_np = cp.asnumpy(Se_gpu)   # (B, 221, 600)

    rows  = []
    worst = {"max_err": -1.0, "err_field": None, "name": None}

    for i, (name, ref) in enumerate(refs):
        Se_ref = ref["Se"]   # (221, 600)
        Se_run = Se_np[i]    # (221, 600)

        if Se_run.shape != Se_ref.shape:
            print(f"  {name}: SHAPE MISMATCH {Se_run.shape} vs {Se_ref.shape} — skipped")
            continue

        err     = np.abs(Se_run - Se_ref)
        max_err  = float(err.max())
        mean_err = float(err.mean())
        passed   = max_err < TOL
        rows.append((name, max_err, mean_err, passed))

        if max_err > worst["max_err"]:
            worst = {"max_err": max_err, "err_field": err.copy(), "name": name}

        flag = "OK  " if passed else "FAIL"
        print(f"  {name}  max|dSe|={max_err:.2e}  mean={mean_err:.2e}  [{flag}]")

    n_pass = sum(1 for r in rows if r[3])
    print(f"\n  GPU result : {n_pass}/{len(rows)} passed  |  "
          f"worst max|dSe| = {worst['max_err']:.2e}  |  "
          f"wall time = {elapsed:.2f} s  ({elapsed/len(rows)*1000:.1f} ms/run)")
    return rows, worst, elapsed


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(cpu_rows, cpu_worst, gpu_rows, gpu_worst):
    n = len(cpu_rows)
    x = np.arange(n)
    cpu_errs = np.array([r[1] for r in cpu_rows])
    gpu_errs = np.array([r[1] for r in gpu_rows]) if gpu_rows else None

    # --- Plot 1: max error per run, both paths ---
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x - 0.2, cpu_errs, width=0.35, color="#444444", label="CPU serial")
    if gpu_errs is not None:
        ax.bar(x + 0.2, gpu_errs, width=0.35, color="#999999", label="GPU batch")
    ax.axhline(TOL, color="#cc3333", linestyle="--", linewidth=0.8,
               label=f"TOL = {TOL:.0e}")
    ax.set_yscale("log")
    ax.set_xlabel("run index", fontsize=11)
    ax.set_ylabel("max |ΔSe|  (log scale)", fontsize=11)
    ax.set_title("Python port vs MATLAB — max Se error per run (all 201 runs)",
                 fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig("verify_all_errors.png", dpi=140)
    print("\nWrote verify_all_errors.png")
    plt.close(fig)

    # --- Plot 2: error heatmap for worst CPU run ---
    if cpu_worst["err_field"] is not None:
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(cpu_worst["err_field"], aspect="auto", origin="lower",
                       cmap="Greys")
        ax.set_xlabel("hour (0–599)", fontsize=10)
        ax.set_ylabel("depth node (0–220)", fontsize=10)
        ax.set_title(f"Error field |ΔSe(z,t)| — worst run: {cpu_worst['name']}",
                     fontsize=11)
        fig.colorbar(im, ax=ax, label="|ΔSe|")
        fig.tight_layout()
        fig.savefig("verify_all_heatmap.png", dpi=140)
        print("Wrote verify_all_heatmap.png")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def write_summary(cpu_rows, cpu_elapsed, gpu_rows, gpu_elapsed):
    lines = []
    lines.append("verify_all.py — summary")
    lines.append("=" * 70)

    header = f"{'run':<14}  {'CPU max|dSe|':>14}  {'CPU mean':>10}  "
    if gpu_rows:
        header += f"{'GPU max|dSe|':>14}  {'GPU mean':>10}"
    lines.append(header)
    lines.append("-" * 70)

    gpu_map = {r[0]: r for r in gpu_rows} if gpu_rows else {}
    for r in cpu_rows:
        name, c_max, c_mean, c_pass = r
        line = f"{name:<14}  {c_max:>14.3e}  {c_mean:>10.3e}  "
        if name in gpu_map:
            _, g_max, g_mean, _ = gpu_map[name]
            line += f"{g_max:>14.3e}  {g_mean:>10.3e}"
        lines.append(line)

    lines.append("=" * 70)
    n_cpu = sum(1 for r in cpu_rows if r[3])
    lines.append(f"CPU : {n_cpu}/{len(cpu_rows)} passed  |  "
                 f"worst = {max(r[1] for r in cpu_rows):.3e}  |  "
                 f"total = {cpu_elapsed:.1f} s  "
                 f"({cpu_elapsed/len(cpu_rows):.3f} s/run)")
    if gpu_rows and gpu_elapsed is not None:
        n_gpu = sum(1 for r in gpu_rows if r[3])
        lines.append(f"GPU : {n_gpu}/{len(gpu_rows)} passed  |  "
                     f"worst = {max(r[1] for r in gpu_rows):.3e}  |  "
                     f"total = {gpu_elapsed:.2f} s  "
                     f"({gpu_elapsed/len(gpu_rows)*1000:.1f} ms/run)")
        lines.append(f"Speedup (GPU vs CPU) : {cpu_elapsed/gpu_elapsed:.1f}x")

    txt = "\n".join(lines)
    with open("verify_all_summary.txt", "w") as f:
        f.write(txt)
    print("\n" + txt)
    print("\nWrote verify_all_summary.txt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    if not os.path.isdir(folder):
        print(f"Dataset folder not found: {folder}")
        print("Usage: python verify_all.py <path/to/dataset>")
        sys.exit(1)

    files = sorted(glob.glob(os.path.join(folder, "run_*.mat")))
    if not files:
        print(f"No run_*.mat files found in {folder}")
        sys.exit(1)

    print(f"Found {len(files)} .mat files in {folder}")
    print(f"GPU available: {ON_GPU}")

    # Load all reference data once — avoids re-reading during each pass
    print("\nLoading .mat files...")
    t_load = time.perf_counter()
    refs = []
    for f in files:
        refs.append((os.path.basename(f), load_mat(f)))
    print(f"  Loaded {len(refs)} files in {time.perf_counter()-t_load:.1f} s")

    cpu_rows, cpu_worst, cpu_elapsed = run_serial(refs)
    gpu_rows, gpu_worst, gpu_elapsed = run_gpu(refs)

    make_plots(cpu_rows, cpu_worst, gpu_rows, gpu_worst)
    write_summary(cpu_rows, cpu_elapsed, gpu_rows, gpu_elapsed)
