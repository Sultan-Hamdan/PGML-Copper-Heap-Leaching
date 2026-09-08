"""
column_solver.py
================
GPU batch driver for the one-dimensional Richards solver under van Genuchten
Mualem closure. Each run is an independent 5.5 m column on 221 depth nodes; the
driver integrates many of them in parallel on the GPU rather than one at a time.

The hydraulic functions (S_theta, h_theta, K_theta, D_theta, dtheta_dh_theta)
are implemented inline here on purpose. Every array is a CuPy array when running
on GPU, so importing NumPy versions from elsewhere would silently pull data back
to the host and defeat the batching. Do not refactor them into a shared import
without accounting for that.

Functions provided:
    run_batch(Q_batch, **params)   -- runs one batch of B simulations,
                                      returns Se of shape (B, Nz+1, N_hours)
                                      sat_eps holds theta and the ghost node
                                      just inside both clip bounds so D stays
                                      finite when a column floods or fully
                                      drains. diag=True additionally returns a
                                      dict of convergence counters; see the
                                      v1.5 and v1.6 changelog entries.
    run_many(Q_all, chunk, ...)    -- splits a large set of runs into
                                      GPU-safe chunks and concatenates results.
                                      Optionally saves each chunk to disk for
                                      crash resilience.

For dataset generation with varied soil parameters (Ks, alpha, n, theta_i), pass them as arrays of length B instead of 
scalars and the physics will apply each parameter set to its corresponding run automatically.

Status: stable
Depends on: tridiagonal_solve.py

Changelog
---------
2026.09.07  v1.9  Repository release. Renamed from sim_gpu.py, import updated
                  to tridiagonal_solve. Comment wording only.
2026.08.01  v1.8  Boolean reductions in the diag block cast to float64 first,
                  avoiding a CuPy JIT path that fails against the CUDA headers.
                  Toolchain only.
2026.08.01  v1.7  Convergence test now reads the last Picard pass only. v1.6
                  flagged any pass, which counted normal early overshoot and
                  inflated the count to 22 of 64 soils, mixing excursions of
                  0.007 to 0.34 of the theta span with four real failures at
                  1.2e5. n_iter stays at 20; the released dataset used 20.
2026.08.01  v1.6  Ghost clamp on tm1, the extrapolated node above the surface,
                  which was the only unclipped state and left [theta_r,
                  theta_s] by up to 1e30 in all four failures. Added diag,
                  off by default: the Picard solve was returning theta from
                  -13.0 to +38.6 across 56 of 221 nodes on a quantity bounded
                  in [0, 0.3315], which the clip then flattened into something
                  that looked physical. Finite is not converged.
2026.08.01  v1.5  Guard made symmetric; v1.4 capped only the wet bound.
                  Failures 24 of 64 (v1.3), 4 of 64 (v1.4), 0 (v1.5).
2026.08.01  v1.4  Saturation guard on the Picard clip. theta clipped to exactly
                  theta_s gave D = -K/0 = -inf and poisoned the batch with NaN:
                  24 of 64 soils went non-finite at rates the cap permits, so
                  R/Ks <= 0.30 is a steady-state bound and does not guarantee
                  the transient integrates.
2026.07.28  v1.3  Comments naming the numerical scheme at the lines that set it.
2026.07.19  v1.2  Optional theta0 for run_batch, so a run can start from a given
                  profile. Needed for day-at-a-time closed-loop stepping.
2026.07.19  v1.1  Optional return_qout for run_batch, plus daily_qout() to
                  collapse it to the daily average.
2026.06.09  v1.0  Initial GPU batch implementation, verified against a
                  single-column reference to machine precision.
"""

import os
import time
import numpy as np

try:
    import cupy as cp
    xp = cp
    ON_GPU = True
except Exception:
    xp = np
    ON_GPU = False

from tridiagonal_solve import solve_thomas_kernel, solve_thomas_pure


def _sync():
    if ON_GPU:
        cp.cuda.runtime.deviceSynchronize()


def _solve(lower, main, upper, d):
    """Use the CUDA kernel on GPU; fall back to pure loop on CPU."""
    if ON_GPU:
        return solve_thomas_kernel(lower, main, upper, d)
    return solve_thomas_pure(lower, main, upper, d)


# --- hydraulics (method=2 VGM), xp-native -----------------------------------
def S_theta(theta, tr, ts): return (theta - tr) / (ts - tr)

def h_theta(S, n, alpha):
    m = 1.0 - 2.0 / n
    return (1.0 / alpha) * ((1.0 / S) ** (1.0 / m) - 1.0) ** (1.0 / n)

def K_theta(S, Ks, n):
    m = 1.0 - 2.0 / n
    delta = 3.0 + 2.0 / (m * n)
    return Ks * S ** delta

def dtheta_dh_theta(h, ts, tr, n, alpha):
    m = 1.0 - 2.0 / n
    mult = (ts - tr) * n * alpha * (-m)
    ah = alpha * h
    return mult * (ah ** (n - 1.0)) / ((1.0 + ah ** n) ** (m + 1.0))

def D_theta(K, dthdh): return -K / dthdh


def _ghost(theta0, D0, K0, Dz, R, n, alpha, Ks, tr, ts, sat_eps=1e-6):
    # This works on single-column (B,) quantities. Params may arrive as (B,1)
    # from the main solver; flatten them so everything stays (B,).
    def _f(p):
        p = xp.asarray(p)
        return p.ravel() if p.ndim > 1 else p
    n = _f(n); alpha = _f(alpha); Ks = _f(Ks); tr = _f(tr); ts = _f(ts)
    R = _f(R)
    tm1 = theta0 + (R - K0) * Dz / D0
    # GHOST CLAMP (v1.6). tm1 is the only state in the scheme that is never
    # clipped, and it is a linear extrapolation above the surface, so nothing
    # bounds it. Measured leaving [theta_r, theta_s] by factors up to 1e30 once
    # the top node pins at saturation and the Picard iteration stops
    # converging. Outside that interval S_theta gives S < 0 or S > 1, and
    # h_theta then raises a negative base to the fractional power 1/n, which is
    # NaN. Every value this clamp touches was already NaN downstream, so it
    # cannot change a finite result. It stops the arithmetic being undefined;
    # it does NOT make a non-converged step trustworthy -- that is what the
    # diag counters below are for.
    tm1 = xp.clip(tm1, tr + sat_eps * (ts - tr), ts - sat_eps * (ts - tr))
    S = S_theta(tm1, tr, ts)
    K0g = K_theta(S, Ks, n)
    h0 = h_theta(S, n, alpha)
    dth = dtheta_dh_theta(h0, ts, tr, n, alpha)
    return tm1, D_theta(K0g, dth), K0g


def _build_b(theta_v, R, ts, tr, Ks, alpha, n, Dz, Dt, sat_eps=1e-6):
    # CRANK-NICOLSON, EXPLICIT HALF.
    # Builds the right-hand-side vector b = theta^n + (Dt/2) * L(theta^n),
    # i.e. the spatial operator L evaluated at the OLD time level and weighted by Dt/2.
    # Its implicit twin (Dt/2 at the NEW time level) is the matrix in _build_diags_c.
    # Equal Dt/2 weight on old and new levels is exactly what
    # makes the time scheme Crank-Nicolson (second order in time)
    # Scheme inherited unchanged from Olivares 2025.
    S = S_theta(theta_v, tr, ts)
    K = K_theta(S, Ks, n)
    h = h_theta(S, n, alpha)
    D = D_theta(K, dtheta_dh_theta(h, ts, tr, n, alpha))
    tm1, Dm1, Km1 = _ghost(theta_v[:, 0], D[:, 0], K[:, 0], Dz, R, n, alpha, Ks, tr, ts, sat_eps)
    w1 = Dt / (2.0 * Dz ** 2); w2 = Dt / (4.0 * Dz)  # w1 carries the Dt/2 Crank-Nicolson time weight
    b = xp.empty_like(theta_v)
    Df = (D[:, 1] + D[:, 0]) / 2.0; Db = (D[:, 0] + Dm1) / 2.0
    b[:, 0] = theta_v[:, 0] + w1 * (Df * (theta_v[:, 1] - theta_v[:, 0]) - Db * (theta_v[:, 0] - tm1)) - w2 * (K[:, 1] - Km1)
    Df = (D[:, 2:] + D[:, 1:-1]) / 2.0; Db = (D[:, 1:-1] + D[:, :-2]) / 2.0
    b[:, 1:-1] = theta_v[:, 1:-1] + w1 * (Df * (theta_v[:, 2:] - theta_v[:, 1:-1]) - Db * (theta_v[:, 1:-1] - theta_v[:, :-2])) - w2 * (K[:, 2:] - K[:, :-2])
    Df = D[:, -1]; Db = (D[:, -1] + D[:, -2]) / 2.0
    b[:, -1] = theta_v[:, -1] + w1 * (-Db * (theta_v[:, -1] - theta_v[:, -2])) - w2 * (K[:, -1] - K[:, -2])
    return b


def _build_diags_c(theta_v, R, ts, tr, Ks, alpha, n, Dz, Dt, sat_eps=1e-6):
    # CRANK-NICOLSON, IMPLICIT HALF.
    # Builds the matrix A = I - (Dt/2) * L, returned as its lower/main/upper
    # diagonals, plus the vector c holding boundary and gravity terms.
    # The full time step is  A * theta^{n+1} = b + c , with b from _build_b (explicit half).
    # A is tridiagonal because central finite differences in space couple each depth node only to its two neighbours.
    # That tridiagonal system is what tridiagonal_solve.py inverts.
    # Scheme inherited from Olivares 2025.
    S = S_theta(theta_v, tr, ts)
    K = K_theta(S, Ks, n)
    h = h_theta(S, n, alpha)
    D = D_theta(K, dtheta_dh_theta(h, ts, tr, n, alpha))
    tm1, Dm1, Km1 = _ghost(theta_v[:, 0], D[:, 0], K[:, 0], Dz, R, n, alpha, Ks, tr, ts, sat_eps)
    w1 = Dt / (2.0 * Dz ** 2); w2 = Dt / (4.0 * Dz)  # w1 carries the Dt/2 Crank-Nicolson time weight
    lower = xp.zeros_like(theta_v); main = xp.zeros_like(theta_v)
    upper = xp.zeros_like(theta_v); c = xp.zeros_like(theta_v)
    Dff = (D[:, 1] + D[:, 0]) / 2.0; Dbf = (D[:, 0] + Dm1) / 2.0
    main[:, 0] = 1.0 + w1 * (Dff + Dbf); upper[:, 0] = -w1 * Dff
    c[:, 0] = -w2 * (K[:, 1] - Km1) + w1 * Dbf * tm1
    Dff = (D[:, 2:] + D[:, 1:-1]) / 2.0; Dbf = (D[:, 1:-1] + D[:, :-2]) / 2.0
    main[:, 1:-1] = 1.0 + w1 * (Dff + Dbf); upper[:, 1:-1] = -w1 * Dff; lower[:, 1:-1] = -w1 * Dbf
    c[:, 1:-1] = -w2 * (K[:, 2:] - K[:, :-2])
    Dff = D[:, -1]; Dbf = (D[:, -1] + D[:, -2]) / 2.0
    main[:, -1] = 1.0 + w1 * (Dff + Dbf); lower[:, -1] = -w1 * Dbf
    c[:, -1] = -w2 * (K[:, -1] - K[:, -2]) + w1 * Dff * theta_v[:, -1]
    return lower, main, upper, c


def run_batch(Q_batch, Ks=170.0, theta_s=0.33, theta_r=0.0, theta_i=0.14,
              alpha=0.035, n=2.267, H=5.50, Area=308.0, tf=25,
              Dz=2.5, Dt=1.0 / 24.0, D_H=24, n_iter=20, heartbeat=False,
              return_qout=False, theta0=None, sat_eps=1e-6,
              diag=False, div_tol=1e-6):
    Q_batch = xp.asarray(Q_batch, dtype=xp.float64)
    B = Q_batch.shape[0]
    Nz = int(H * 100 / Dz)
    N_hours = tf * D_H

    # Move all parameters onto the active backend (numpy or cupy). Per-run
    # arrays (shape (B,)) are reshaped to (B,1) so they broadcast against the
    # (B, Nz+1) state. Scalars pass through unchanged.
    def _prep(p):
        a = xp.asarray(p, dtype=xp.float64)
        return a[:, None] if a.ndim == 1 else a
    Ks = _prep(Ks); alpha = _prep(alpha); n = _prep(n)
    theta_s = _prep(theta_s); theta_r = _prep(theta_r)

    R_eval = (Q_batch[:, :tf] / Area) * 100.0
    ti = xp.asarray(theta_i, dtype=xp.float64)
    if theta0 is not None:
        # Closed-loop use: start from a supplied profile (B, Nz+1) instead of
        # the uniform theta_i initial condition. Default path is unchanged.
        theta_act = xp.asarray(theta0, dtype=xp.float64)
        if theta_act.ndim == 1:
            theta_act = theta_act[None, :] * xp.ones((B, 1))
    else:
        theta_act = xp.ones((B, Nz + 1)) * (ti[:, None] if ti.ndim else ti)
    # Interior clip bounds, computed once. Both raw bounds are singular:
    # theta == theta_s gives dtheta_dh = 0 and D = -K/0 = -inf; theta == theta_r
    # gives h = inf and then inf/inf = NaN in dtheta_dh_theta.
    lo_b = theta_r + sat_eps * (theta_s - theta_r)
    hi_b = theta_s - sat_eps * (theta_s - theta_r)

    # CONVERGENCE DIAGNOSTICS (v1.6), off by default so the standard path is
    # byte-identical. When on, records where the Picard solve returned a value
    # outside [theta_r, theta_s] BEFORE clipping. That is the signature of a
    # non-converged step: the clip then flattens the divergence into a profile
    # that looks physical and scores like a real result. A finite number is not
    # the same as a converged one, and without this there is no way to tell
    # them apart after the fact.
    if diag:
        span = theta_s - theta_r
        # ANY-PASS counters. Picard is a fixed-point iteration: overshooting on
        # an early pass and settling later is normal behaviour, not divergence.
        # These are kept for continuity with v1.6 but they are NOT the test.
        n_div_steps = xp.zeros(B, dtype=xp.int64)
        max_excursion = xp.zeros(B)
        first_bad_hour = xp.full(B, -1, dtype=xp.int64)
        # FINAL-PASS counters (v1.7). This is the test that matters: is the
        # iterate still outside [theta_r, theta_s] when the loop STOPS? If so
        # the clip is manufacturing a physical-looking profile out of a
        # non-converged solve, and the resulting Se is not a solution of
        # anything.
        n_div_final = xp.zeros(B, dtype=xp.int64)
        max_excursion_final = xp.zeros(B)
        first_bad_hour_final = xp.full(B, -1, dtype=xp.int64)
        # PICARD RESIDUAL (v1.7). max |theta_k - theta_{k-1}| / span on the last
        # pass. Contracting to ~0 means converged; staying large or growing
        # means the iteration is not settling regardless of bounds.
        max_residual = xp.zeros(B)
        sum_residual = xp.zeros(B)
        n_res_steps = 0
        n_at_hi = xp.zeros(B, dtype=xp.int64)       # nodes pinned at wet bound
        n_at_lo = xp.zeros(B, dtype=xp.int64)       # nodes pinned at dry bound

    Se_full = xp.empty((B, Nz + 1, N_hours))
    # Hourly outlet-flow proxy, bottom node:
    #   qout_hour = K_theta(S_end_of_hour)(bottom) * (Area/100), m3/day-equiv.
    # Only filled when return_qout is requested; default path is unchanged.
    qout_full = xp.empty((B, N_hours)) if return_qout else None

    t0 = time.perf_counter()
    for dia in range(tf):
        R_day = R_eval[:, dia]  # (B,)
        for h in range(D_H):
            # One Crank-Nicolson time step per hour. The explicit half b is
            # fixed from theta^n; the implicit half A is rebuilt each pass below.
            b = _build_b(theta_act, R_day, theta_s, theta_r, Ks, alpha, n, Dz, Dt,
                         sat_eps)
            theta_v = theta_act
            hour_bad = xp.zeros(B, dtype=bool) if diag else None
            exc_final = None
            res_final = None
            theta_prev = theta_v if diag else None
            # Fixed-point iteration (Picard) for the nonlinear K(theta), D(theta).
            # The matrix coefficients depend on the unknown theta^{n+1}, so we
            # solve, rebuild the coefficients from the new estimate, and repeat
            # until the profile settles. n_iter=20 is the value the released
            # dataset was generated at. Each pass solves the system via _solve.
            for _ in range(n_iter):
                lo, ma, up, c = _build_diags_c(theta_v, R_day, theta_s, theta_r, Ks, alpha, n, Dz, Dt,
                                               sat_eps)
                theta_v = _solve(lo, ma, up, b + c)
                if diag:
                    exc = xp.maximum(
                        xp.maximum(theta_v - theta_s, theta_r - theta_v).max(axis=1),
                        0.0) / span.ravel()
                    hour_bad |= exc > div_tol
                    max_excursion = xp.maximum(max_excursion, exc)
                    # overwritten every pass, so after the loop these hold the
                    # FINAL pass values
                    exc_final = exc
                    res_final = xp.abs(theta_v - theta_prev).max(axis=1) \
                        / span.ravel()
                    theta_prev = theta_v
                # SATURATION GUARD (v1.5). Both clip bounds are singular.
                # At theta == theta_s, S = 1 gives h = 0, dtheta_dh = 0 and
                # D = -K/0 = -inf. At theta == theta_r, S = 0 gives h = inf
                # and then inf/inf = NaN inside dtheta_dh_theta. Either one
                # poisons the whole batch. Holding theta a hair inside both
                # bounds keeps D large but finite, so a flooded or fully
                # drained column stays scoreable instead of destroying the
                # run. This fires ONLY where the previous code produced a
                # non-finite value, so it cannot change any result that was
                # already finite. Inertness against the dataset is verified
                # by replay, not assumed.
                theta_v = xp.clip(theta_v, lo_b, hi_b)
            theta_act = theta_v
            if diag:
                idx = dia * D_H + h
                n_div_steps += hour_bad.astype(xp.float64).astype(xp.int64)
                first_bad_hour = xp.where((first_bad_hour < 0) & hour_bad,
                                          idx, first_bad_hour)
                bad_final = exc_final > div_tol
                n_div_final += bad_final.astype(xp.float64).astype(xp.int64)
                max_excursion_final = xp.maximum(max_excursion_final, exc_final)
                first_bad_hour_final = xp.where(
                    (first_bad_hour_final < 0) & bad_final, idx,
                    first_bad_hour_final)
                max_residual = xp.maximum(max_residual, res_final)
                sum_residual += res_final
                n_res_steps += 1
                # Cast to float64 before reducing. Summing a BOOLEAN array
                # sends CuPy down a CUB accumulate path that JIT-compiles a
                # kernel against the CUDA toolkit headers, and CUDA 13.3's
                # cuda_fp8/fp6/fp4 headers fail to parse under this CuPy build
                # (NVRTC_ERROR_COMPILATION). Float64 reductions use the path
                # every other reduction here already uses. Environment issue,
                # not a numerical one; CUPY_ACCELERATORS="" also avoids it.
                n_at_hi += (theta_act >= hi_b).astype(xp.float64).sum(axis=1).astype(xp.int64)
                n_at_lo += (theta_act <= lo_b).astype(xp.float64).sum(axis=1).astype(xp.int64)
            Se_full[:, :, dia * D_H + h] = (theta_act - theta_r) / (theta_s - theta_r)
            if return_qout:
                # End-of-hour bottom-node conductivity.
                S_end = S_theta(theta_act, theta_r, theta_s)
                K_end = K_theta(S_end, Ks, n)
                qout_full[:, dia * D_H + h] = K_end[:, -1] * (Area / 100.0)
        if heartbeat:
            _sync(); el = time.perf_counter() - t0
            print(f"  day {dia+1}/{tf}  {el:.1f}s  (~{el/(dia+1)*tf:.0f}s total)", flush=True)
    if diag:
        _to = (lambda a: cp.asnumpy(a)) if ON_GPU else (lambda a: np.asarray(a))
        diag_out = dict(
            n_div_steps=_to(n_div_steps),
            max_excursion=_to(max_excursion),
            first_bad_hour=_to(first_bad_hour),
            n_div_final=_to(n_div_final),
            max_excursion_final=_to(max_excursion_final),
            first_bad_hour_final=_to(first_bad_hour_final),
            max_residual=_to(max_residual),
            mean_residual=_to(sum_residual / max(n_res_steps, 1)),
            n_at_hi=_to(n_at_hi),
            n_at_lo=_to(n_at_lo),
            n_hours=N_hours,
            n_nodes=Nz + 1,
            n_iter=n_iter,
            div_tol=div_tol,
            sat_eps=sat_eps,
        )
        if return_qout:
            return Se_full, qout_full, diag_out
        return Se_full, diag_out
    if return_qout:
        return Se_full, qout_full
    return Se_full


def daily_qout(qout_full, tf, D_H=24):
    """Collapse hourly outlet-flow proxy (B, tf*D_H) to the daily average
    proxy (B, tf) used by the identifier, matching mean(qout_hour) per day
    the identifier consumes."""
    B = qout_full.shape[0]
    return qout_full.reshape(B, tf, D_H).mean(axis=2)


def run_many(Q_all, chunk=512, checkpoint_dir=None, **kw):
    """Cover many runs in VRAM-safe chunks. Returns host numpy (B, Nz+1, hours).
    If checkpoint_dir given, saves each chunk to disk (resume-safe)."""
    Q_all = np.asarray(Q_all, dtype=float)
    total = Q_all.shape[0]
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    out = []
    for s in range(0, total, chunk):
        Qc = Q_all[s:s + chunk]
        Se = run_batch(Qc, **kw)
        Se_h = cp.asnumpy(Se) if ON_GPU else Se
        if checkpoint_dir:
            np.save(os.path.join(checkpoint_dir, f"se_{s:06d}.npy"), Se_h)
        out.append(Se_h)
        print(f"chunk {s}-{s+len(Qc)} / {total} done", flush=True)
    return np.concatenate(out, axis=0)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    B = 500
    Qs = np.clip(rng.integers(0, 49, size=(B, 25)).astype(float), 0, 48)
    print(f"backend: {'GPU' if ON_GPU else 'CPU'}, B={B}")
    t0 = time.perf_counter()
    Se = run_batch(Qs, heartbeat=True)
    _sync()
    print(f"TOTAL {B} runs: {time.perf_counter()-t0:.1f}s  shape "
          f"{(cp.asnumpy(Se) if ON_GPU else Se).shape}")
