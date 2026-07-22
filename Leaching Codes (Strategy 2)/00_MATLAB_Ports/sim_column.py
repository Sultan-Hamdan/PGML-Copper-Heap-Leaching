"""
sim_column.py
=============
Port of the serial VGM simulator:
    Sim_column.m  -> run_simulation()   (top-level: day loop, downsampling, output)
    columna.m     -> simulate_day()     (one day: 24 hourly sub-steps, 20 inner fixed-point iterations each)

run_simulation() runs a full simulation over tf days, advancing the moisture profile hourly,
and returns the complete Se and theta histories alongside downsampled sensor readings.
All input parameters (soil properties, grid settings, irrigation schedule) are bundled in a SimParams object
so that generating a dataset is just calling run_simulation() once per parameter set.
Default values reproduce run_0000, the baseline used for verification.

run_simulation() returns a SimResult holding:
    Se_full    : (Nz+1) x N_hours   full-resolution effective saturation (training target)
    theta_full : (Nz+1) x N_hours   full-resolution water content
    meas_matrix: (N_meas + 2) x N_hours   downsampled sensors + R row + q_out row
    R_hour     : (N_hours,)         hourly irrigation (cm/day)
    q_out      : (N_hours,)         hourly outflow proxy (m^3/day-equiv)
    Z_meas     : sensor depths (cm)
    params     : the SimParams used, recorded alongside the output for dataset labelling

File dependencies:
    hydraulics.py        -- pure VGM hydraulic relations, no project imports
    theta_m1_theta.py    -- top boundary ghost node, uses hydraulics.py
    pred_correct.py      -- matrix assembly and solver, uses hydraulics.py + theta_m1_theta.py
    sim_column.py        -- this file, serial driver, uses pred_correct.py
    thomas_kernel.py     -- standalone CUDA Thomas solver, no project imports
    sim_gpu.py           -- GPU batch driver, uses thomas_kernel.py

Origin: port of Sim_column.m, columna.m
Status: stable

Changelog
---------
2026.06.09  v1.0  Initial port of Sim_column.m and columna.m from MATLAB
"""

from dataclasses import dataclass, field
import numpy as np

from hydraulics import S_theta, K_theta
from pred_correct import build_b, pred_correct


# Default daily inlet flow schedule (44 values, m^3/day). run_0000 uses the
# first tf entries. Mirrors Q_0_vect in Sim_column.m.
_DEFAULT_Q = [12, 19, 19, 0, 0, 0, 12, 32, 36, 39, 46, 27, 23, 24, 27, 30, 29,
              28, 28, 27, 36, 35, 26, 26, 26, 25, 22, 26, 24, 24, 26, 22, 22,
              25, 27, 27, 23, 19, 20, 19, 16, 17, 21, 30]


@dataclass
class SimParams:
    """All parameters defining one simulation run. Defaults reproduce run_0000."""
    Ks: float = 170.0          # saturated hydraulic conductivity (cm/day) -- already *100
    theta_s: float = 0.33      # saturated water content
    theta_r: float = 0.0       # residual water content
    theta_i: float = 0.14      # initial (uniform) water content
    alpha: float = 0.035       # VGM parameter (1/cm)
    n: float = 2.267           # VGM parameter
    method: int = 2            # 2 = VGM power-law K = Ks*S^delta
    H: float = 5.50            # column height (m)
    Area: float = 308.0        # cross-sectional area (m^2)
    tf: int = 25               # simulation horizon (days)
    Dz: float = 2.5            # spatial step (cm)
    Dt: float = 1.0 / 24.0     # time step (days) = 1 hour
    D_H: int = 24              # hours per day
    n_iter: int = 20           # inner fixed-point iterations per hour
    Dz_sampling: float = 10.0  # sensor spacing for downsampled output (cm)
    solver: str = "tridiag"    # "tridiag" (fast) or "full" (dense reference)
    Q_schedule: list = field(default_factory=lambda: list(_DEFAULT_Q))


@dataclass
class SimResult:
    Se_full: np.ndarray
    theta_full: np.ndarray
    meas_matrix: np.ndarray
    R_hour: np.ndarray
    q_out: np.ndarray
    Z_meas: np.ndarray
    params: SimParams


# =============================================================================
# simulate_day : advance the state through 24 hourly sub-steps.
#                Mirrors columna.m.
#   Returns:
#     theta_day  : (Nz+1) x D_H   end-of-hour profiles for the day
#     qout_day   : (D_H,)         hourly outflow proxy
# =============================================================================
def simulate_day(theta_act, R, p):
    Nz = len(theta_act) - 1
    theta_day = np.zeros((Nz + 1, p.D_H))
    qout_day = np.zeros(p.D_H)

    for h in range(p.D_H):
        # Build RHS once from the start-of-hour state
        b = build_b(theta_act, R, p.theta_s, p.theta_r, p.Ks,
                    p.alpha, p.n, p.method, p.Dz, p.Dt)

        # Fixed-point iterations
        theta_v = theta_act.copy()
        for _ in range(p.n_iter):
            theta_v = pred_correct(theta_v, b, R, p.theta_s, p.theta_r, p.Ks,
                                   p.alpha, p.n, p.method, p.Dz, p.Dt,
                                   solver=p.solver)

        # End-of-hour update and storage
        theta_act = theta_v
        theta_day[:, h] = theta_act

        # Hourly q_out from the end-of-hour state (bottom-node conductivity)
        S_h = S_theta(theta_act, p.theta_r, p.theta_s)
        K_h = K_theta(S_h, p.Ks, p.n, p.method)
        qout_day[h] = K_h[Nz] * (p.Area / 100.0)

    return theta_day, qout_day


# =============================================================================
# run_simulation : full run over tf days. Mirrors Sim_column.m.
# =============================================================================
def run_simulation(p: SimParams = None) -> SimResult:
    if p is None:
        p = SimParams()

    Nz = int(p.H * 100 / p.Dz)          # 220 -> 221 nodes
    N_days = p.tf
    N_hours = N_days * p.D_H            # 600 for tf=25

    Z_vect = np.arange(0, p.H * 100 + p.Dz / 2, p.Dz)   # node depths (cm)

    # Daily irrigation rate, cm/day: R = (Q/Area)*100. First tf of the schedule.
    Q = np.asarray(p.Q_schedule[:N_days], dtype=float)
    R_eval = (Q / p.Area) * 100.0

    # Initial uniform state
    theta_act = np.ones(Nz + 1) * p.theta_i

    theta_full = np.zeros((Nz + 1, N_hours))
    R_hour = np.zeros(N_hours)
    q_out = np.zeros(N_hours)

    for dia in range(N_days):
        R_vgm = R_eval[dia]
        theta_day, qout_day = simulate_day(theta_act, R_vgm, p)

        h_slice = slice(dia * p.D_H, (dia + 1) * p.D_H)
        theta_full[:, h_slice] = theta_day
        R_hour[h_slice] = R_vgm
        q_out[h_slice] = qout_day

        # Next day starts from the last hour of this day
        theta_act = theta_day[:, -1]

    # Effective saturation surface (the source of truth)
    Se_full = (theta_full - p.theta_r) / (p.theta_s - p.theta_r)

    # --- Axial downsampling to mimic sparse sensors (mirrors Sim_column.m) ---
    ratio = p.Dz_sampling / p.Dz
    if abs(ratio - round(ratio)) < 1e-12:
        step = int(round(ratio))
        idx_meas = list(range(0, Nz + 1, step))
        if idx_meas[-1] != Nz:
            idx_meas.append(Nz)
        idx_meas = np.array(idx_meas)
        Z_meas = Z_vect[idx_meas]
        theta_meas = theta_full[idx_meas, :]
    else:
        Z_meas = np.arange(0, p.H * 100 + p.Dz_sampling / 2, p.Dz_sampling)
        theta_meas = np.zeros((len(Z_meas), N_hours))
        for hh in range(N_hours):
            theta_meas[:, hh] = np.interp(Z_meas, Z_vect, theta_full[:, hh])

    # meas_matrix: sensor theta rows, then R row, then q_out row
    meas_matrix = np.vstack([theta_meas, R_hour, q_out])

    return SimResult(
        Se_full=Se_full,
        theta_full=theta_full,
        meas_matrix=meas_matrix,
        R_hour=R_hour,
        q_out=q_out,
        Z_meas=Z_meas,
        params=p,
    )


# =============================================================================
# Standalone inspection block.
# Runs the full run_0000 simulation and prints shapes + the Se range, which
# should match the known baseline [0.424, 0.794] (Olivares Fig. 6).
# =============================================================================
if __name__ == "__main__":
    import time

    t0 = time.perf_counter()
    res = run_simulation()
    elapsed = time.perf_counter() - t0

    print(f"Run completed in {elapsed:.2f} s")
    print("Se_full shape    :", res.Se_full.shape, "(expect 221 x 600)")
    print("meas_matrix shape:", res.meas_matrix.shape)
    print("Z_meas (cm)      :", res.Z_meas[:5], "...", res.Z_meas[-1])
    print("Se range         : [%.4f, %.4f]" % (res.Se_full.min(), res.Se_full.max()))
    print("                   (expect approx [0.4242, 0.7940])")
    print("q_out[-1]        :", res.q_out[-1])
