"""
hydraulics.py
=============
Port of the five elementwise hydraulic functions from the MATLAB VGM simulator.
method=2 (VGM power-law) is the form used throughout the project.

MATLAB sources mirrored here:
    S_theta.m          -> S_theta()
    h_theta.m          -> h_theta()
    K_theta.m          -> K_theta()
    D_theta.m          -> D_theta()
    dtheta_dh_theta.m  -> dtheta_dh_theta()

These are pure elementwise relations, so they are written as NumPy vector
operations, meaning each function operates on whole arrays at once rather than
looping node by node. This is mathematically identical to the MATLAB
per-element loops and is compatible with GPU batch processing in sim_gpu.py.

Convention: `method == 1` selects van Genuchten (VG); anything else selects
the VGM variant used in this codebase, exactly as the MATLAB does.

Origin: port of S_theta.m, h_theta.m, K_theta.m, D_theta.m, dtheta_dh_theta.m
Status: stable
Depends on: nothing (base layer)

Changelog
---------
2026.06.09  v1.0  Initial port of all five VGM hydraulic functions from MATLAB
"""

import numpy as np


# -----------------------------------------------------------------------------
# S_theta : volumetric water content theta -> effective saturation S
#   S = (theta - theta_r) / (theta_s - theta_r)
# Mirrors S_theta.m
# -----------------------------------------------------------------------------
def S_theta(theta, theta_r, theta_s):
    dtheta = theta_s - theta_r
    return (theta - theta_r) / dtheta


# -----------------------------------------------------------------------------
# h_theta : effective saturation S -> pressure head h  (van Genuchten form)
#   m = 1 - 1/n      (method == 1, VG)
#   m = 1 - 2/n      (otherwise, VGM)
#   h = (1/alpha) * ( (1/S)^(1/m) - 1 )^(1/n)
# Mirrors h_theta.m
# -----------------------------------------------------------------------------
def h_theta(S, n, alpha, method):
    if method == 1:
        m = 1.0 - 1.0 / n
    else:
        m = 1.0 - 2.0 / n
    return (1.0 / alpha) * ((1.0 / S) ** (1.0 / m) - 1.0) ** (1.0 / n)


# -----------------------------------------------------------------------------
# K_theta : effective saturation S -> hydraulic conductivity K
#   method == 1 (VG / Mualem):
#       m = 1 - 1/n
#       K = Ks * sqrt(S) * (1 - (1 - S^(1/m))^m)^2
#   otherwise (VGM power-law, the form used in this project):
#       m = 1 - 2/n
#       delta = 3 + 2/(m*n)
#       K = Ks * S^delta
# Mirrors K_theta.m
# -----------------------------------------------------------------------------
def K_theta(S, Ks, n, method):
    if method == 1:
        m = 1.0 - 1.0 / n
        return Ks * np.sqrt(S) * (1.0 - (1.0 - S ** (1.0 / m)) ** m) ** 2
    else:
        m = 1.0 - 2.0 / n
        delta = 3.0 + 2.0 / (m * n)
        return Ks * S ** delta


# -----------------------------------------------------------------------------
# dtheta_dh_theta : water capacity term dtheta/dh
#   m = 1 - 1/n   (method == 1, VG)  ;  m = 1 - 2/n  (otherwise, VGM)
#   mult = (theta_s - theta_r) * n * alpha * (-m)
#   dtheta/dh = mult * (alpha*h)^(n-1) / (1 + (alpha*h)^n)^(m+1)
# Mirrors dtheta_dh_theta.m
# -----------------------------------------------------------------------------
def dtheta_dh_theta(h, theta_s, theta_r, n, alpha, method):
    dtheta = theta_s - theta_r
    if method == 1:
        m = 1.0 - 1.0 / n
    else:
        m = 1.0 - 2.0 / n
    mult = dtheta * n * alpha * (-m)
    ah = alpha * h
    return mult * (ah ** (n - 1.0)) / ((1.0 + ah ** n) ** (m + 1.0))


# -----------------------------------------------------------------------------
# D_theta : diffusivity-like coefficient
#   D = -K / (dtheta/dh)
# Mirrors D_theta.m
# -----------------------------------------------------------------------------
def D_theta(K, dtheta_dh):
    return -K / dtheta_dh


# -----------------------------------------------------------------------------
# Standalone inspection block.
# Run `python hydraulics.py` to evaluate all five functions on the run_0000
# baseline state (uniform theta_i = 0.14) and print the results, so the output
# can be eyeballed without a notebook.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # run_0000 parameters
    theta_s = 0.33
    theta_r = 0.0
    Ks = 170.0          # cm/day (Ks * 100 internally in the MATLAB driver)
    alpha = 0.035       # 1/cm
    n = 2.267
    method = 2

    # Uniform initial state, a few nodes is enough to inspect
    theta = np.array([0.14, 0.14, 0.14])

    S = S_theta(theta, theta_r, theta_s)
    h = h_theta(S, n, alpha, method)
    K = K_theta(S, Ks, n, method)
    dth_dh = dtheta_dh_theta(h, theta_s, theta_r, n, alpha, method)
    D = D_theta(K, dth_dh)

    print("theta      =", theta)
    print("S          =", S)
    print("h          =", h)
    print("K          =", K)
    print("dtheta_dh  =", dth_dh)
    print("D          =", D)

    # Quick sanity hand-check for K at S = theta_i/theta_s:
    m = 1.0 - 2.0 / n
    delta = 3.0 + 2.0 / (m * n)
    print("\ndelta      =", delta)
    print("K check    =", Ks * S[0] ** delta, "(should match K[0])")
