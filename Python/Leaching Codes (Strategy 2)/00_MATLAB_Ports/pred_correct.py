"""
pred_correct.py
===============
Port of the predictor-corrector core:
    Pred_correct_b.m  -> build_b()       (the RHS vector, built once per hour)
    Pred_correct.m    -> pred_correct()  (one fixed-point iteration: assemble
                                          A and c, solve A*theta = b + c, clamp)

The matrix and boundary condition assembly is mirrored line-for-line against the MATLAB,
with the only change being the unavoidable 1-based to 0-based index shift:
    MATLAB node i = 1        (top)      -> Python index 0
    MATLAB node i = 2..Nz    (interior) -> Python index 1..Nz-1
    MATLAB node i = Nz+1     (bottom)   -> Python index Nz   (the (Nz+1)-th node)

Two solvers are provided for verification purposes:
    solve_full(A, B)    -- builds the dense (Nz+1)x(Nz+1) matrix and solves with
                           numpy.linalg.solve. This is the exact analogue of
                           MATLAB's A\\B (blunt, treats all entries as possibly
                           non-zero). Used as the verification reference.
    solve_tridiag(...)  -- exploits the tridiagonal structure via
                           scipy.linalg.solve_banded. Mathematically identical,
                           much faster. Used in sim_column.py and sim_gpu.py.

build_A_c() returns the three diagonals (lower, main, upper) plus c, so the same assembly feeds both
solvers with no risk of them diverging because of differing assembly code.

Origin: port of Pred_correct_b.m, Pred_correct.m
Status: stable
Depends on: hydraulics.py, theta_m1_theta.py

Changelog
---------
2026.06.09  v1.0  Initial port of Pred_correct_b.m and Pred_correct.m from MATLAB
"""

import numpy as np
from scipy.linalg import solve_banded

from hydraulics import S_theta, h_theta, K_theta, D_theta, dtheta_dh_theta
from theta_m1_theta import theta_m1_theta


# =============================================================================
# build_b : RHS vector, evaluated at the start-of-hour state theta_v
#           Mirrors Pred_correct_b.m
# =============================================================================
def build_b(theta_v, R, theta_s, theta_r, Ks, alpha, n, method, Dz, Dt):
    Nz = len(theta_v) - 1   # so that index Nz is the (Nz+1)-th / bottom node

    # Hydraulic properties at the current state
    S = S_theta(theta_v, theta_r, theta_s)
    h = h_theta(S, n, alpha, method)
    K = K_theta(S, Ks, n, method)
    dtheta_dh = dtheta_dh_theta(h, theta_s, theta_r, n, alpha, method)
    D = D_theta(K, dtheta_dh)

    # Ghost-node quantities at the top boundary (node -1)
    D0 = D[0]
    K0 = K[0]
    theta_0 = theta_v[0]
    theta_m1, D_m1, K_m1 = theta_m1_theta(
        theta_0, D0, K0, Dz, R, n, alpha, Ks, theta_r, theta_s, method
    )

    w1 = Dt / (2.0 * Dz ** 2)
    w2 = Dt / (4.0 * Dz)

    b = np.zeros(Nz + 1)

    # --- Top node (MATLAB i = 1 -> Python 0): uses ghost node ---
    i = 0
    Df = (D[i + 1] + D[i]) / 2.0
    Db = (D[i] + D_m1) / 2.0
    b[i] = (theta_v[i]
            + w1 * (Df * (theta_v[i + 1] - theta_v[i]) - Db * (theta_v[i] - theta_m1))
            - w2 * (K[i + 1] - K_m1))

    # --- Interior nodes (MATLAB i = 2..Nz -> Python 1..Nz-1) ---
    for i in range(1, Nz):
        Df = (D[i + 1] + D[i]) / 2.0
        Db = (D[i] + D[i - 1]) / 2.0
        b[i] = (theta_v[i]
                + w1 * (Df * (theta_v[i + 1] - theta_v[i]) - Db * (theta_v[i] - theta_v[i - 1]))
                - w2 * (K[i + 1] - K[i - 1]))

    # --- Bottom node (MATLAB i = Nz+1 -> Python Nz): zero-gradient BC ---
    i = Nz
    Df = D[i]
    Db = (D[i] + D[i - 1]) / 2.0
    b[i] = (theta_v[i]
            + w1 * (Df * 0.0 - Db * (theta_v[i] - theta_v[i - 1]))
            - w2 * (K[i] - K[i - 1]))

    return b


# =============================================================================
# build_A_c : assemble the system matrix (as three diagonals) and vector c,
#             evaluated at the current iterate theta_v.
#             Mirrors the A and c assembly in Pred_correct.m.
#
# Returns:
#   lower : sub-diagonal,   length Nz+1, lower[i] multiplies theta[i-1] (lower[0] unused)
#   main  : main diagonal,  length Nz+1
#   upper : super-diagonal, length Nz+1, upper[i] multiplies theta[i+1] (upper[Nz] unused)
#   c     : vector c, length Nz+1
#
# Both solvers consume this same output, so they cannot diverge in assembly.
# =============================================================================
def build_A_c(theta_v, R, theta_s, theta_r, Ks, alpha, n, method, Dz, Dt):
    Nz = len(theta_v) - 1

    S = S_theta(theta_v, theta_r, theta_s)
    h = h_theta(S, n, alpha, method)
    K = K_theta(S, Ks, n, method)
    dtheta_dh = dtheta_dh_theta(h, theta_s, theta_r, n, alpha, method)
    D = D_theta(K, dtheta_dh)

    D0 = D[0]
    K0 = K[0]
    theta_0 = theta_v[0]
    theta_m1, D_m1, K_m1 = theta_m1_theta(
        theta_0, D0, K0, Dz, R, n, alpha, Ks, theta_r, theta_s, method
    )

    w1 = Dt / (2.0 * Dz ** 2)
    w2 = Dt / (4.0 * Dz)

    lower = np.zeros(Nz + 1)
    main = np.zeros(Nz + 1)
    upper = np.zeros(Nz + 1)
    c = np.zeros(Nz + 1)

    # --- Top node (MATLAB i = 1 -> Python 0) ---
    i = 0
    Dff = (D[i + 1] + D[i]) / 2.0
    Dbf = (D[i] + D_m1) / 2.0
    main[i] = 1.0 + w1 * (Dff + Dbf)
    upper[i] = -w1 * Dff
    c[i] = -w2 * (K[i + 1] - K_m1) + w1 * Dbf * theta_m1

    # --- Interior nodes (MATLAB i = 2..Nz -> Python 1..Nz-1) ---
    for i in range(1, Nz):
        Dff = (D[i + 1] + D[i]) / 2.0
        Dbf = (D[i] + D[i - 1]) / 2.0
        main[i] = 1.0 + w1 * (Dff + Dbf)
        upper[i] = -w1 * Dff
        lower[i] = -w1 * Dbf
        c[i] = -w2 * (K[i + 1] - K[i - 1])

    # --- Bottom node (MATLAB i = Nz+1 -> Python Nz) ---
    i = Nz
    Dff = D[i]
    Dbf = (D[i] + D[i - 1]) / 2.0
    main[i] = 1.0 + w1 * (Dff + Dbf)
    lower[i] = -w1 * Dbf
    c[i] = -w2 * (K[i] - K[i - 1]) + w1 * Dff * theta_v[i]

    return lower, main, upper, c


# =============================================================================
# Solver 1 (reference): dense build + numpy.linalg.solve.
#   Exact analogue of MATLAB A\B. Treats the matrix as fully dense.
# =============================================================================
def solve_full(lower, main, upper, B):
    N = len(main)
    A = np.zeros((N, N))
    for i in range(N):
        A[i, i] = main[i]
        if i + 1 < N:
            A[i, i + 1] = upper[i]
        if i - 1 >= 0:
            A[i, i - 1] = lower[i]
    return np.linalg.solve(A, B)


# =============================================================================
# Solver 2 (fast): scipy.linalg.solve_banded on the three diagonals.
#   solve_banded wants the banded storage layout:
#       ab[0, 1:]  = upper diagonal (shifted right by one)
#       ab[1, :]   = main diagonal
#       ab[2, :-1] = lower diagonal (shifted left by one)
# =============================================================================
def solve_tridiag(lower, main, upper, B):
    N = len(main)
    ab = np.zeros((3, N))
    ab[0, 1:] = upper[:-1]   # super-diagonal
    ab[1, :] = main          # main diagonal
    ab[2, :-1] = lower[1:]   # sub-diagonal
    return solve_banded((1, 1), ab, B)


# =============================================================================
# pred_correct : one fixed-point iteration.
#   Assembles A and c at theta_v, solves A*theta_new = b + c with the chosen
#   solver, then clamps theta_new to [theta_r, theta_s].
#   Mirrors Pred_correct.m (clamp included, applied every iteration).
#
#   solver: "tridiag" (default, fast) or "full" (dense reference)
# =============================================================================
def pred_correct(theta_v, b, R, theta_s, theta_r, Ks, alpha, n, method, Dz, Dt,
                 solver="tridiag"):
    lower, main, upper, c = build_A_c(
        theta_v, R, theta_s, theta_r, Ks, alpha, n, method, Dz, Dt
    )
    B = c + b

    if solver == "full":
        theta_v_p = solve_full(lower, main, upper, B)
    else:
        theta_v_p = solve_tridiag(lower, main, upper, B)

    # Clamp to physical bounds [theta_r, theta_s]
    theta_v_p = np.clip(theta_v_p, theta_r, theta_s)
    return theta_v_p


# =============================================================================
# Standalone inspection block.
# Runs ONE hour (build b once, then 20 fixed-point iterations) from the
# run_0000 baseline, with BOTH solvers, and:
#   1. confirms the two solvers agree (max abs difference),
#   2. times each over repeated runs to show the performance gap.
# =============================================================================
if __name__ == "__main__":
    import time

    # run_0000 parameters
    theta_s = 0.33
    theta_r = 0.0
    Ks = 170.0        # cm/day
    alpha = 0.035     # 1/cm
    n = 2.267
    method = 2
    Dz = 2.5          # cm
    Dt = 1.0 / 24.0   # days (1 hour)
    H = 5.50
    Nz = int(H * 100 / Dz)   # 220 -> 221 nodes

    Area = 308.0
    R = (12.0 / Area) * 100.0   # day-1 irrigation, cm/day

    theta_init = np.ones(Nz + 1) * 0.14

    def run_one_hour(solver):
        theta_act = theta_init.copy()
        b = build_b(theta_act, R, theta_s, theta_r, Ks, alpha, n, method, Dz, Dt)
        theta_v = theta_act.copy()
        for _ in range(20):
            theta_v = pred_correct(theta_v, b, R, theta_s, theta_r, Ks, alpha, n,
                                   method, Dz, Dt, solver=solver)
        return theta_v

    # 1. Agreement check
    theta_full = run_one_hour("full")
    theta_trid = run_one_hour("tridiag")
    max_diff = np.max(np.abs(theta_full - theta_trid))
    print("Solver agreement (max abs diff):", max_diff)
    print("Top 3 nodes (full)   :", theta_full[:3])
    print("Top 3 nodes (tridiag):", theta_trid[:3])

    # 2. Timing comparison (repeat the one-hour solve many times)
    reps = 50
    t0 = time.perf_counter()
    for _ in range(reps):
        run_one_hour("full")
    t_full = (time.perf_counter() - t0) / reps

    t0 = time.perf_counter()
    for _ in range(reps):
        run_one_hour("tridiag")
    t_trid = (time.perf_counter() - t0) / reps

    print(f"\nPer-hour solve time, dense  A\\B   : {t_full*1e3:.3f} ms")
    print(f"Per-hour solve time, tridiagonal  : {t_trid*1e3:.3f} ms")
    print(f"Speed-up (dense / tridiag)        : {t_full/t_trid:.1f}x")
