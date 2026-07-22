"""
physics_residual.py  -- Part D, Option 1 (full implicit balance check)
======================================================================
The residual grades a predicted next-state Se_pred(t+1) against the simulator's
own discretised scheme:  A * theta_next = b + c.

It does NOT solve (no Thomas). It plugs the prediction in and measures the
leftover:  res = A @ theta_pred - (b + c).

If theta_pred obeys the scheme (e.g. the true data), res ~ 0 (machine precision).
Non-zero res = physical violation, and its size is the physics-loss signal.

VETTING: equivalence test -- feeding the TRUE Se(t+1) must give res ~ 0.
"""
import numpy as np
from pred_correct import build_b, build_A_c


def physics_residual(theta_t, theta_pred, R, theta_s, theta_r, Ks, alpha, n,
                     method, Dz, Dt):
    """
    theta_t    : current state (start-of-hour), shape (Nz+1,)
    theta_pred : predicted next state to be graded, shape (Nz+1,)
    Returns    : residual vector, shape (Nz+1,)

    Scheme:  A * theta_next = b + c
      - b is built ONCE from the start-of-hour state theta_t (build_b).
      - A and c are evaluated at the iterate; at convergence the iterate IS
        theta_next, so we evaluate A,c at theta_pred (mirrors how c uses the
        current iterate -- the b-vs-c timescale point).
    Residual: A @ theta_pred - (b + c)
    """
    # b: from start-of-hour state (frozen)
    b = build_b(theta_t, R, theta_s, theta_r, Ks, alpha, n, method, Dz, Dt)

    # A, c: evaluated AT the prediction (the converged-iterate state)
    lower, main, upper, c = build_A_c(theta_pred, R, theta_s, theta_r, Ks,
                                      alpha, n, method, Dz, Dt)

    # A @ theta_pred  using the three diagonals
    N = len(main)
    Ax = main * theta_pred
    Ax[:-1] += upper[:-1] * theta_pred[1:]    # super-diagonal
    Ax[1:]  += lower[1:]  * theta_pred[:-1]   # sub-diagonal

    res = Ax - (b + c)
    return res


# =============================================================================
# EQUIVALENCE TEST against real strategy-1 data (run_0000)
# =============================================================================
if __name__ == "__main__":
    import scipy.io as sio

    m = sio.loadmat('run0000_fresh.mat')
    Se_out = m['Se_out']
    R_out  = m['R_out'].ravel()

    theta_s, theta_r = 0.33, 0.0
    Ks, alpha, n = 170.0, 0.035, 2.267
    method, Dz, Dt = 2, 2.5, 1.0/24.0
    dtheta = theta_s - theta_r

    def se_to_theta(Se): return theta_r + Se*dtheta

    # test across several timesteps
    print(f"{'t':>5} {'R':>8} {'max|res|':>14}")
    print("-"*30)
    worst = 0.0
    for t in [10, 50, 100, 200, 300, 400, 500, 598]:
        theta_t   = se_to_theta(Se_out[:, t])
        theta_tp1 = se_to_theta(Se_out[:, t+1])   # TRUE next, in theta
        R_t = R_out[t+1]
        res = physics_residual(theta_t, theta_tp1, R_t, theta_s, theta_r,
                               Ks, alpha, n, method, Dz, Dt)
        mx = np.max(np.abs(res))
        worst = max(worst, mx)
        print(f"{t:>5} {R_t:>8.4f} {mx:>14.3e}")
    print("-"*30)
    print(f"WORST max|res| across all = {worst:.3e}")
    print("PASS (res ~ machine precision)" if worst < 1e-9 else "FAIL -- residual too large, bug in assembly")
