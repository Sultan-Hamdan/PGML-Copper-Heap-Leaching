"""
sim_gpu.py
==========
GPU batch driver for the VGM simulator. Runs many independent simulation runs in parallel on the GPU,
producing the same Se output as sim_column.py but across a full batch of B runs at once.

The hydraulic functions (S_theta, h_theta, K_theta, D_theta, dtheta_dh_theta) are intentionally re-implemented inline 
here rather than imported from hydraulics.py.
This is because hydraulics.py uses NumPy, and all arrays here are CuPy arrays when running on GPU.
Importing from hydraulics.py would silently pull data back to CPU, defeating the purpose of GPU batching.
Do not refactor this into a shared import without accounting for this.

Functions provided:
    run_batch(Q_batch, **params)   -- runs one batch of B simulations, returns Se of shape (B, Nz+1, N_hours)
    run_many(Q_all, chunk, ...)    -- splits a large set of runs into GPU-safe chunks and concatenates results.
                                      Optionally saves each chunk to disk for crash resilience.

For dataset generation with varied soil parameters (Ks, alpha, n, theta_i), pass them as arrays of length B instead
of scalars and the physics will apply each parameter set to its corresponding run automatically.

Origin: original (not a MATLAB port)
Status: stable
Depends on: thomas_kernel.py

Changelog
---------
2026.06.09  v1.0  Initial GPU batch implementation, verified vs sim_column.py to ~1e-14
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

from thomas_kernel import solve_thomas_kernel, solve_thomas_pure


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


def _ghost(theta0, D0, K0, Dz, R, n, alpha, Ks, tr, ts):
    # This works on single-column (B,) quantities. Params may arrive as (B,1)
    # from the main solver; flatten them so everything stays (B,).
    def _f(p):
        p = xp.asarray(p)
        return p.ravel() if p.ndim > 1 else p
    n = _f(n); alpha = _f(alpha); Ks = _f(Ks); tr = _f(tr); ts = _f(ts)
    R = _f(R)
    tm1 = theta0 + (R - K0) * Dz / D0
    S = S_theta(tm1, tr, ts)
    K0g = K_theta(S, Ks, n)
    h0 = h_theta(S, n, alpha)
    dth = dtheta_dh_theta(h0, ts, tr, n, alpha)
    return tm1, D_theta(K0g, dth), K0g


def _build_b(theta_v, R, ts, tr, Ks, alpha, n, Dz, Dt):
    S = S_theta(theta_v, tr, ts)
    K = K_theta(S, Ks, n)
    h = h_theta(S, n, alpha)
    D = D_theta(K, dtheta_dh_theta(h, ts, tr, n, alpha))
    tm1, Dm1, Km1 = _ghost(theta_v[:, 0], D[:, 0], K[:, 0], Dz, R, n, alpha, Ks, tr, ts)
    w1 = Dt / (2.0 * Dz ** 2); w2 = Dt / (4.0 * Dz)
    b = xp.empty_like(theta_v)
    Df = (D[:, 1] + D[:, 0]) / 2.0; Db = (D[:, 0] + Dm1) / 2.0
    b[:, 0] = theta_v[:, 0] + w1 * (Df * (theta_v[:, 1] - theta_v[:, 0]) - Db * (theta_v[:, 0] - tm1)) - w2 * (K[:, 1] - Km1)
    Df = (D[:, 2:] + D[:, 1:-1]) / 2.0; Db = (D[:, 1:-1] + D[:, :-2]) / 2.0
    b[:, 1:-1] = theta_v[:, 1:-1] + w1 * (Df * (theta_v[:, 2:] - theta_v[:, 1:-1]) - Db * (theta_v[:, 1:-1] - theta_v[:, :-2])) - w2 * (K[:, 2:] - K[:, :-2])
    Df = D[:, -1]; Db = (D[:, -1] + D[:, -2]) / 2.0
    b[:, -1] = theta_v[:, -1] + w1 * (-Db * (theta_v[:, -1] - theta_v[:, -2])) - w2 * (K[:, -1] - K[:, -2])
    return b


def _build_diags_c(theta_v, R, ts, tr, Ks, alpha, n, Dz, Dt):
    S = S_theta(theta_v, tr, ts)
    K = K_theta(S, Ks, n)
    h = h_theta(S, n, alpha)
    D = D_theta(K, dtheta_dh_theta(h, ts, tr, n, alpha))
    tm1, Dm1, Km1 = _ghost(theta_v[:, 0], D[:, 0], K[:, 0], Dz, R, n, alpha, Ks, tr, ts)
    w1 = Dt / (2.0 * Dz ** 2); w2 = Dt / (4.0 * Dz)
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
              Dz=2.5, Dt=1.0 / 24.0, D_H=24, n_iter=20, heartbeat=False):
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
    theta_act = xp.ones((B, Nz + 1)) * (ti[:, None] if ti.ndim else ti)
    Se_full = xp.empty((B, Nz + 1, N_hours))

    t0 = time.perf_counter()
    for dia in range(tf):
        R_day = R_eval[:, dia]  # (B,)
        for h in range(D_H):
            b = _build_b(theta_act, R_day, theta_s, theta_r, Ks, alpha, n, Dz, Dt)
            theta_v = theta_act
            for _ in range(n_iter):
                lo, ma, up, c = _build_diags_c(theta_v, R_day, theta_s, theta_r, Ks, alpha, n, Dz, Dt)
                theta_v = _solve(lo, ma, up, b + c)
                theta_v = xp.clip(theta_v, theta_r, theta_s)
            theta_act = theta_v
            Se_full[:, :, dia * D_H + h] = (theta_act - theta_r) / (theta_s - theta_r)
        if heartbeat:
            _sync(); el = time.perf_counter() - t0
            print(f"  day {dia+1}/{tf}  {el:.1f}s  (~{el/(dia+1)*tf:.0f}s total)", flush=True)
    return Se_full


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
