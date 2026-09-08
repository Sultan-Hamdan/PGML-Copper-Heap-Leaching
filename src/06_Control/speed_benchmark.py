r"""
speed_benchmark.py
==================
Times the numerical solver on the CPU, for comparison against the GPU figures.

The two solver modules pick their backend by trying to import cupy, so the CPU
path is selected by blocking that import through sys.meta_path before either
module loads. The script aborts if either is still on the GPU path.

FLAGS
  --counts       column counts, comma separated. Default "1,10". Ratios are
                 printed only for counts in GPU_PER_COLUMN_MS below. The CPU
                 path is sequential in depth, so 100 columns is 20,000 one-day
                 solves and will take a long time.
  --candidates   irrigation rates per decision. Default 200.
  --repeats      timed repetitions. Default 3, after 1 warmup.
  --timeout_s    abandon a count that exceeds this. Default 1800.

Writes speed_r4.json and a table to the terminal.

Usage, from 06_Control:
    python -u speed_benchmark.py --counts "1,10"

--root defaults through paths.py.

Status: stable
Depends on: column_solver.py, tridiagonal_solve.py, closed_loop.py,
            preprocess.py, paths.py, numpy

Changelog
---------
2026.09.07  v1.2  Repository release. Renamed from probe_speed_r4.py. Imports
                  updated to column_solver, tridiagonal_solve and closed_loop.
                  --root defaults through paths.py.
2026.08.26  v1.1  Patching the module attribute alone failed, since the two
                  modules import cupy independently and a numpy array could not
                  be written into a cupy buffer. Now blocks the cupy import
                  through sys.meta_path before either module loads, and aborts
                  if either is still on the GPU path.
2026.08.26  v1.0  Initial. CPU numerical solver.
"""

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths

DEF_ROOT = paths.REPO_ROOT

CARIAGA = {"Ks": 170.0, "alpha": 0.035, "n_vgm": 2.267,
           "theta_s": 0.33, "theta_r": 0.0, "Se_i": 0.14 / 0.33}

# GPU reference. Same soil, same counts, same code path.
GPU_PER_COLUMN_MS = {1: 1238.0, 10: 193.86, 100: 145.498}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=DEF_ROOT)
    p.add_argument("--counts", default="1,10",
                   help="must match the counts the GPU side was timed at")
    p.add_argument("--candidates", type=int, default=200)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--timeout_s", type=float, default=1800.0,
                   help="abandon a count that exceeds this, and say so")
    p.add_argument("--Dz", type=float, default=2.5)
    p.add_argument("--Dt", type=float, default=1.0 / 24.0)
    p.add_argument("--D_H", type=int, default=24)
    p.add_argument("--H", type=float, default=5.50)
    p.add_argument("--n_iter", type=int, default=20)
    p.add_argument("--area", type=float, default=308.0)
    p.add_argument("--interval_s", type=float, default=86400.0)
    a = p.parse_args()

    prep_dir = paths.REFERENCE
    paths.add_stage_paths()

    # ---- force the CPU backend BEFORE anything imports cupy ------------
    # Patching a module attribute is not enough: the two solver modules import
    # cupy independently, so solve_thomas_pure allocated a cupy array and then
    # failed filling it from numpy. Blocking the import makes every module take
    # its own except branch, whatever its internal names are.
    import importlib.abc

    for _m in [k for k in sys.modules if k == "cupy" or
               k.startswith("cupy.")]:
        del sys.modules[_m]

    class _BlockCupy(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "cupy" or fullname.startswith("cupy."):
                raise ImportError("cupy blocked for CPU timing")
            return None

    sys.meta_path.insert(0, _BlockCupy())

    import column_solver
    import tridiagonal_solve
    from closed_loop import plant_day
    from preprocess import load_scalers

    scal = load_scalers(prep_dir)

    print("=" * 78)
    print("numerical solver on the processor")
    print("=" * 78)
    print("cupy import : BLOCKED before the solver modules loaded")
    print(f"column_solver      ON_GPU={column_solver.ON_GPU}  "
          f"xp is numpy: {column_solver.xp is np}")
    tk_xp = getattr(tridiagonal_solve, "xp", None)
    tk_gpu = getattr(tridiagonal_solve, "ON_GPU", None)
    print(f"tridiagonal_solve  ON_GPU={tk_gpu}  xp is numpy: {tk_xp is np}")
    bad = column_solver.ON_GPU or column_solver.xp is not np or \
        (tk_xp is not None and tk_xp is not np) or tk_gpu is True
    if bad:
        print("\nABORT: a module is still on the GPU path. Every number this "
              "would print\nwould be mislabelled.")
        return
    print(f"candidates : {a.candidates}   D_H: {a.D_H}   n_iter: {a.n_iter}")
    print(f"repeats    : {a.repeats} timed, {a.warmup} warmup")
    print("statistic  : mean and sample SD, Olivares Table 2 convention")
    print("soil       : Cariaga 2015, replicated across columns")
    print(f"timeout    : {a.timeout_s:.0f} s per count")

    counts = sorted({int(t) for t in a.counts.split(",") if t.strip()})
    unknown = [c for c in counts if c not in GPU_PER_COLUMN_MS]
    if unknown:
        print(f"WARNING: no GPU figure at {unknown}. Those rows cannot be "
              f"matched and will print without a ratio.")

    r_min = float(scal["R_out"]["min"])
    r_cap = float(scal["R_out"]["max"])
    Nz1 = int(a.H * 100 / a.Dz) + 1

    print("\n" + "-" * 78)
    print(f"{'columns':>9}{'solves':>10}{'mean s':>12}{'SD s':>10}"
          f"{'per col ms':>13}{'GPU ms':>10}{'CPU/GPU':>9}")
    print("-" * 78)

    results = []
    for S in counts:
        soil = {k: np.full(S, v, dtype=np.float64) for k, v in CARIAGA.items()}
        r_max = np.minimum(0.30 * soil["Ks"], r_cap)
        w = np.linspace(0.0, 1.0, a.candidates)[None, :]
        cand_R = r_min + (r_max[:, None] - r_min) * w

        Se0 = np.repeat(soil["Se_i"][:, None], Nz1, axis=1)
        theta = Se0 * (soil["theta_s"] - soil["theta_r"])[:, None] \
            + soil["theta_r"][:, None]

        rep = np.repeat(np.arange(S), a.candidates)
        soil_rep = {k: soil[k][rep] for k in soil}
        theta_rep = theta[rep]
        R_rep = cand_R.reshape(-1)

        def f():
            plant_day(theta_rep, R_rep, soil_rep, a.area, a.Dz, a.Dt,
                      a.D_H, a.n_iter, a.H, diag=False)

        t_start = time.perf_counter()
        try:
            for _ in range(a.warmup):
                f()
                if time.perf_counter() - t_start > a.timeout_s:
                    raise TimeoutError
            ts = []
            for _ in range(a.repeats):
                t0 = time.perf_counter()
                f()
                ts.append(time.perf_counter() - t0)
                if time.perf_counter() - t_start > a.timeout_s:
                    break
            mean = statistics.fmean(ts)
            sd = statistics.stdev(ts) if len(ts) > 1 else 0.0
            per = 1e3 * mean / S
            g = GPU_PER_COLUMN_MS.get(S)
            row = {"columns": S, "solves": S * a.candidates, "status": "ok",
                   "mean_s": mean, "sd_s": sd, "n": len(ts),
                   "per_column_ms": per, "gpu_per_column_ms": g,
                   "cpu_over_gpu": (per / g) if g else None}
            print(f"{S:>9}{S*a.candidates:>10}{mean:>12.2f}{sd:>10.2f}"
                  f"{per:>13.1f}"
                  f"{(f'{g:.1f}' if g else '--'):>10}"
                  f"{(f'{per/g:.1f}x' if g else '--'):>9}")
        except (TimeoutError, KeyboardInterrupt):
            row = {"columns": S, "solves": S * a.candidates,
                   "status": "timeout",
                   "wall_s": time.perf_counter() - t_start}
            print(f"{S:>9}{S*a.candidates:>10}   TIMEOUT after "
                  f"{time.perf_counter()-t_start:.0f} s")
        results.append(row)

    ok = [r for r in results if r["status"] == "ok"]
    print("\n" + "-" * 78)
    print("NUMERICAL SOLVER, CPU")
    print("-" * 78)
    if ok:
        big = max(ok, key=lambda r: r["columns"])
        print(f"per column      : {big['per_column_ms']:.1f} ms at "
              f"{big['columns']} columns")
        pcs = [r["per_column_ms"] for r in ok]
        print(f"range explored  : {min(pcs):.1f} to {max(pcs):.1f} ms")
        if big.get("cpu_over_gpu"):
            print(f"CPU / GPU       : {big['cpu_over_gpu']:.1f}x "
                  f"at matched column count")
        cols = 1e4 * 100.0 / a.area
        pct = 100.0 * cols * big["per_column_ms"] / 1e3 / a.interval_s
        print(f"100 ha heap     : {cols:,.0f} columns, "
              f"{pct:.3f} % of the {a.interval_s:.0f} s step")
        print("\nQuote it with its column count attached. The per-column cost "
              "was still falling\nwith batch size on the GPU side and there "
              "is no reason to assume it has settled here.")
    else:
        print("no count completed inside the timeout")

    out = {"backend_forced": "cpu",
           "method": "cupy import blocked via sys.meta_path before the "
                     "solver modules were loaded",
           "counts": counts, "candidates": a.candidates,
           "repeats": a.repeats, "warmup": a.warmup, "n_iter": a.n_iter,
           "D_H": a.D_H, "Dz": a.Dz, "Dt": a.Dt,
           "interval_s": a.interval_s, "area_per_column_m2": a.area,
           "statistic": "mean and sample standard deviation",
           "cariaga_soil": CARIAGA,
           "gpu_reference_per_column_ms": GPU_PER_COLUMN_MS,
           "results": results}
    with open("speed_r4.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote speed_r4.json")


if __name__ == "__main__":
    main()
