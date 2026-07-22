"""
physics_residual_gardner.py
===========================
The PGML physics loss with Gardner hydraulics. A Gardner swap of the verified VGM baseline physics_residual.py:
ONLY the hydraulics (K, D) and the TOP boundary condition change.
Everything structural -- the three-point stencil, the weights w1/w2, the face-averaging,
the b/A/c assembly and the residual form res = A @ theta_pred - (b + c) -- is identical to the VGM version.

    VGM (baseline)                  ->   Gardner (here)
        K = Ks * Se**delta                   K = Ks * Se**m
        D = K / (dtheta/dh)                  D = C * Se**(m-1), C=(dtheta*alpha_g)/(m*Ks)
      (delta from n; rational D)           (clean power laws; smooth everywhere)

Gardner uses its OWN parameters alpha_g, m_g (Olivares offline fit: ~4.0047, ~16.4094), DIFFERENT from VGM's alpha, n.
They are explicit inputs (future-proof for Strategy 3, where soil varies per run).

TOP BOUNDARY -- why it differs from VGM (the fix):
    VGM imposes the surface flux via a ghost node, rearranged as
        theta_{-1} = theta_0 + (R - K0) * Dz / D0.
    This DIVIDES by D0. Gardner's D is tiny (~1e-6..1e-9), so the ghost explodes (Se**16 -> residual ~1e+130 on any R>0 step).
    Instead we use Gardner's own surface condition "flux = R" and substitute the KNOWN flux R directly into the stencil,
    eliminating the ghost theta entirely:
        Db*(theta_0 - theta_m1)  ->  Dz * (K0 - R)     (no 1/D anywhere)
        K_{-1}                    ->  K0                (surface advective value)
    Matrix consequence: lower0 = 0 (no ghost coupling), the backward face moves to the RHS (c0),
    main0/upper0 keep only the forward face.
    Derived from the dimensional Gardner flux balance (sympy), verified numerically on run_0000.

VETTING MINDSET (different from the VGM version):
    On VGM-generated data the Gardner residual will NOT be ~0. A meaningful,
    BOUNDED non-zero is CORRECT -- it is the structural-uncertainty signal
    (the gap between the VGM truth and the Gardner model).
    We verify:
      1. runs cleanly, finite, no NaN/Inf (smoothness),
      2. residual magnitude is sensible (non-zero but bounded ~1e-3), and
      3. moving Gardner's m toward a VGM-like value shrinks it (sanity).

Origin: Gardner swap of physics_residual.py (top BC re-derived, not ported)
Status: verified (top BC fixed; interior & bottom unchanged from VGM)
Depends on: numpy (self-contained hydraulics; does NOT import VGM pred_correct)

Changelog
---------
2026.06.13  v1.0  Top boundary re-derived from the dimensional Gardner flux
                  balance "surface flux = R" (Path A). Ghost theta eliminated:
                  Db*(theta_0-theta_m1) -> Dz*(K0-R), K_{-1} -> K0, lower0 = 0.
                  No division by D0. Verified on run_0000: finite, worst
                  |res| ~4.2e-3 (was 1e+130), VGM<->Gardner gap visible and
                  bounded, residual shrinks as m -> VGM-like.
2026.06.13  v0.1  Initial Gardner swap. Interior & Gardner K/D correct, but
                  top boundary reused the VGM ghost form (divides by D0) ->
                  residual blew up to ~1e+130 on R>0 steps. (WIP, broken.)
"""

import numpy as np


def gardner_K(Se, Ks, m):                   # Gardner hydraulic conductivity (m/s) from Se, Ks, m
    return Ks * np.power(Se, m)             # was Ks·Se^δ for VGM


def gardner_D(Se, Ks, m, alpha_g, dtheta):  # Gardner soil water diffusivity (m²/s) from Se, Ks, m, alpha_g, dtheta
    C = (dtheta * alpha_g) / (m * Ks)
    return C * np.power(Se, m - 1.0)        # was K/(dθ/dh) for VGM


def build_b_gardner(theta_v, R, theta_s, theta_r, Ks, m, alpha_g, Dz, Dt):
    Nz = len(theta_v) - 1
    dtheta = theta_s - theta_r

    Se = (theta_v - theta_r) / dtheta
    Se = np.clip(Se, 1e-9, None)
    K = gardner_K(Se, Ks, m)
    D = gardner_D(Se, Ks, m, alpha_g, dtheta)

    w1 = Dt / (2.0 * Dz ** 2)
    w2 = Dt / (4.0 * Dz)
    b = np.zeros(Nz + 1)

    # --- top node (i=0): surface flux = R, ghost eliminated ---
    # diffusive ghost group Db*(th0-thm1) -> Dz*(K0 - R)
    # advective ghost K_{-1} -> K0
    i = 0
    Df = (D[i + 1] + D[i]) / 2.0
    G_diff = Dz * (K[i] - R)        # replaces Db*(theta_0 - theta_m1)
    b[i] = (theta_v[i]
            + w1 * (Df * (theta_v[i + 1] - theta_v[i]) - G_diff)
            - w2 * (K[i + 1] - K[i]))   # K_{-1} -> K[0]

    # --- interior nodes ---
    for i in range(1, Nz):
        Df = (D[i + 1] + D[i]) / 2.0
        Db = (D[i] + D[i - 1]) / 2.0
        b[i] = (theta_v[i]
                + w1 * (Df * (theta_v[i + 1] - theta_v[i]) - Db * (theta_v[i] - theta_v[i - 1]))
                - w2 * (K[i + 1] - K[i - 1]))

    # --- bottom node (i=Nz): zero-gradient ---
    i = Nz
    Df = D[i]
    Db = (D[i] + D[i - 1]) / 2.0
    b[i] = (theta_v[i]
            + w1 * (Df * 0.0 - Db * (theta_v[i] - theta_v[i - 1]))
            - w2 * (K[i] - K[i - 1]))
    return b


def build_A_c_gardner(theta_v, R, theta_s, theta_r, Ks, m, alpha_g, Dz, Dt):
    Nz = len(theta_v) - 1
    dtheta = theta_s - theta_r

    Se = (theta_v - theta_r) / dtheta
    Se = np.clip(Se, 1e-9, None)
    K = gardner_K(Se, Ks, m)
    D = gardner_D(Se, Ks, m, alpha_g, dtheta)

    w1 = Dt / (2.0 * Dz ** 2)
    w2 = Dt / (4.0 * Dz)
    lower = np.zeros(Nz + 1)
    main = np.zeros(Nz + 1)
    upper = np.zeros(Nz + 1)
    c = np.zeros(Nz + 1)

    # --- top node: only forward face couples implicitly; backward face = known flux ---
    i = 0
    Dff = (D[i + 1] + D[i]) / 2.0
    G_diff = Dz * (K[i] - R)
    main[i] = 1.0 + w1 * Dff
    upper[i] = -w1 * Dff
    lower[i] = 0.0
    c[i] = -w2 * (K[i + 1] - K[i]) + w1 * G_diff

    # --- interior ---
    for i in range(1, Nz):
        Dff = (D[i + 1] + D[i]) / 2.0
        Dbf = (D[i] + D[i - 1]) / 2.0
        main[i] = 1.0 + w1 * (Dff + Dbf)
        upper[i] = -w1 * Dff
        lower[i] = -w1 * Dbf
        c[i] = -w2 * (K[i + 1] - K[i - 1])

    # --- bottom node ---
    i = Nz
    Dff = D[i]
    Dbf = (D[i] + D[i - 1]) / 2.0
    main[i] = 1.0 + w1 * (Dff + Dbf)
    lower[i] = -w1 * Dbf
    c[i] = -w2 * (K[i] - K[i - 1]) + w1 * Dff * theta_v[i]

    return lower, main, upper, c


def physics_residual_gardner(theta_t, theta_pred, R, theta_s, theta_r,
                             Ks, m, alpha_g, Dz, Dt):
    b = build_b_gardner(theta_t, R, theta_s, theta_r, Ks, m, alpha_g, Dz, Dt)
    lower, main, upper, c = build_A_c_gardner(theta_pred, R, theta_s, theta_r,
                                              Ks, m, alpha_g, Dz, Dt)
    Ax = main * theta_pred
    Ax[:-1] += upper[:-1] * theta_pred[1:]
    Ax[1:]  += lower[1:]  * theta_pred[:-1]
    return Ax - (b + c)
