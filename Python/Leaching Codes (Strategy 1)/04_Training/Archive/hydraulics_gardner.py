"""
hydraulics_gardner.py
=====================
Gardner-form hydraulic functions: effective saturation Se -> (K, D).

Gardner twin of hydraulics.py. Where the VGM file computes diffusivity D through
a four-step chain (S -> h -> K -> dtheta_dh -> D), the Gardner forms are DIRECT
power laws in Se: no pressure head h, no water-capacity term dtheta_dh. That is
the point of Gardner -- smooth, differentiable power laws everywhere, chosen for
gradient-solver friendliness (continuity), not accuracy against the VGM "truth".

GROUNDING (Sim_Colocation_OnlineID_Gardner.m, lines 280, 285-286, 299):
    C   = dtheta * alpha / (m * Ks)        # line 280  (D_theta there = dtheta)
    K   = Ks * Se**m                       # outlet-flux proxy, line 299
    D   = C * Se**(m-1)                    # diffusivity factor, lines 285-286

    Gardner-specific meanings (NOT the VGM alpha/n):
        m      : Gardner exponent  (paper: beta; code: m -- SAME quantity)
        alpha  : Gardner alpha
        Ks     : saturated hydraulic conductivity (same Ks as VGM)
        dtheta : theta_s - theta_r  (effective porosity)

Independent confirmation: Benfanich, Bourgault, Beljadid, "A finite element
method using a bounded auxiliary variable for solving the Richards equation",
J. Comput. Phys. (2026, in press, doi:10.1016/j.jcp.2026.115127), Table 1, lists
the Gardner constitutive relations alongside van Genuchten and confirms Gardner
satisfies the low-saturation boundedness condition 0 <= lim_{S->0} Kr(S) S^-a < inf.

Strategy note: alpha_g, m_g are EXPLICIT INPUTS, never hardcoded. Strategy 1 uses
the fixed offline-fit constants (alpha_g=4.0047, m_g=16.4094); Strategy 3 would
vary them per sample -- identical code path.

Origin: Gardner forms extracted from Sim_Colocation_OnlineID_Gardner.m
Status: verified -- forms confirmed against MATLAB and an independent peer-
        reviewed source; used inside the verified Gardner residual
Depends on: nothing (base layer, mirrors hydraulics.py)

Changelog
---------
2026.06.12  v0.1  Initial Gardner hydraulics: K_gardner, C_gardner, D_gardner.
2026.06.13  v1.0  Forms cross-checked against an independent peer-reviewed
                  source (J. Comput. Phys. 2026, Table 1). Promoted to v1.0:
                  these K/D feed the verified Gardner residual (599 timesteps
                  x 2 runs, all finite).
"""

import numpy as np


def K_gardner(Se, Ks, m):
    """Gardner conductivity: K = Ks * Se**m.
    Grounded: outlet-flux proxy (MATLAB line 299); gravity term Se**m (line 285)."""
    return Ks * Se ** m


def C_gardner(Ks, alpha, m, theta_s, theta_r):
    """Gardner diffusivity constant: C = dtheta * alpha / (m * Ks).
    Grounded: MATLAB line 280 (C = D_theta*alpha_sym/(m_sym*Ks), D_theta=dtheta)."""
    dtheta = theta_s - theta_r
    return dtheta * alpha / (m * Ks)


def D_gardner(Se, Ks, alpha, m, theta_s, theta_r):
    """Gardner diffusivity: D = C * Se**(m-1).
    Grounded: diffusivity factor Se**(m-1) scaled by C (MATLAB lines 285-286)."""
    C = C_gardner(Ks, alpha, m, theta_s, theta_r)
    return C * Se ** (m - 1.0)


if __name__ == "__main__":
    theta_s, theta_r, Ks = 0.33, 0.0, 170.0
    alpha_g, m_g = 4.0047, 16.4094
    theta = np.array([0.14, 0.14, 0.14])
    Se = (theta - theta_r) / (theta_s - theta_r)
    print("Se =", Se)
    print("C  =", C_gardner(Ks, alpha_g, m_g, theta_s, theta_r))
    print("K  =", K_gardner(Se, Ks, m_g))
    print("D  =", D_gardner(Se, Ks, alpha_g, m_g, theta_s, theta_r))
