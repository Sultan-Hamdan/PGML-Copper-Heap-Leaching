"""
sample_strategy3.py
===================
Parameter sampler for Strategy 3 dataset generation. Uses Sobol sampling to
generate combinations of soil parameters (Ks, alpha, n, theta_i) and
irrigation schedules (Qbar, sigma) to be fed into the GPU batch runner.

This file is not part of the current active pipeline. Strategy 1 is complete
and does not use this file. This file will become the first step of the
Strategy 3 pipeline once parameter ranges are confirmed with the supervisor
and the full Strategy 3 dataset generation is ready to be built.

Current parameter windows (tight, centred on Cariaga nominal values):
    Ks      [1.2,  2.2]   m/day    (nominal 1.70)
    alpha   [0.025, 0.045] 1/cm   (nominal 0.035)
    n       [2.6,  3.0]   -        (nominal 2.267, kept high for stability)
    theta_i [0.12, 0.18]  cm3/cm3  (nominal 0.14)
    Qbar    [15,   30]    m3/day
    sigma   [2,    6]     m3/day

Origin: original (not a MATLAB port)
Status: pending -- not active until Strategy 3 pipeline is established
Depends on: nothing (no project imports)

Changelog
---------
2026.06.09  v1.0  Initial proof-of-concept sampler, narrow windows for stability
"""

import numpy as np
from scipy.stats import qmc

BOUNDS_LOW  = np.array([1.2, 0.025, 2.6, 0.12, 15.0, 2.0])
BOUNDS_HIGH = np.array([2.2, 0.045, 3.0, 0.18, 30.0, 6.0])

N_DAYS = 25
Q_MIN, Q_MAX = 0.0, 48.0


def _walk(Qbar, sigma, rng):
    Q = np.empty(N_DAYS)
    Q[0] = np.clip(Qbar, Q_MIN, Q_MAX)
    for d in range(1, N_DAYS):
        Q[d] = np.clip(Q[d-1] + (rng.random()*2-1)*sigma, Q_MIN, Q_MAX)
    return np.round(Q)


def make_poc(m=6, seed=42):
    """2^m runs (m=6 -> 64). Returns Ks, alpha, n, theta_i, Q, table."""
    u = qmc.Sobol(d=6, scramble=True, seed=seed).random_base2(m=m)
    s = qmc.scale(u, BOUNDS_LOW, BOUNDS_HIGH)
    rng = np.random.default_rng(seed)
    Q = np.array([_walk(qb, sg, rng) for qb, sg in zip(s[:, 4], s[:, 5])])
    return s[:, 0], s[:, 1], s[:, 2], s[:, 3], Q, s


if __name__ == "__main__":
    Ks, al, n, ti, Q, _ = make_poc()
    print(f"POC sample: {len(Ks)} runs")
    print(f"  Ks      [{Ks.min():.2f}, {Ks.max():.2f}]")
    print(f"  alpha   [{al.min():.3f}, {al.max():.3f}]")
    print(f"  n       [{n.min():.2f}, {n.max():.2f}]")
    print(f"  theta_i [{ti.min():.2f}, {ti.max():.2f}]")
    print(f"  Q range [{Q.min():.0f}, {Q.max():.0f}]")
