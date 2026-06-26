"""
theta_m1_theta.py
=================
Port of theta_m1_theta.m, computes a virtual node one step above the physical
top of the column (z = 0), used to enforce the irrigation/infiltration boundary
condition. Because the finite-difference formula at the top node requires a
neighbour above it that does not physically exist, a virtual node (called a
ghost node in numerical PDE literature) is constructed from the boundary
condition itself.

This function is mirrored statement-by-statement against the MATLAB rather
than restructured, because the boundary condition assembly is sensitive to
index ordering and sign conventions. A small mistake produces wrong results
silently with no error raised.

The ghost-node water content theta_{-1} is found by rearranging the top
boundary condition (Eq. 6 in the Olivares formulation):

        -D * dSe/dxi * ... + K = R    at xi = 0

discretised at the first physical node, giving:

        theta_{-1} = theta_0 + (R - K0) * Dz / D0

The hydraulic properties (K, D) are then re-evaluated at this ghost node so
the finite-difference formula at the top node can use them.

Origin: port of theta_m1_theta.m
Status: stable
Depends on: hydraulics.py

Changelog
---------
2026.06.09  v1.0  Initial port of theta_m1_theta.m from MATLAB
"""

from hydraulics import S_theta, h_theta, K_theta, D_theta, dtheta_dh_theta


# -----------------------------------------------------------------------------
# theta_m1_theta : ghost-node quantities at the top boundary
#
# Inputs:
#   theta_0 : water content at the first physical node (scalar)
#   D0      : diffusivity at the first physical node (scalar)
#   K0      : conductivity at the first physical node (scalar)
#   Dz      : spatial step (cm)
#   R       : irrigation/infiltration rate (cm/day, consistent with Ks)
#   n, alpha, Ks, theta_r, theta_s, method : VGM parameters
#
# Returns:
#   theta_m1 : ghost-node water content theta_{-1}
#   D_m1     : diffusivity evaluated at the ghost node
#   K_m1     : conductivity evaluated at the ghost node
#
# Mirrors theta_m1_theta.m. The MATLAB passed Nz=0 to the hydraulic helpers
# purely to force scalar returns; our hydraulics functions are shape-agnostic,
# so a scalar in gives a scalar out and the Nz argument is not needed.
# -----------------------------------------------------------------------------
def theta_m1_theta(theta_0, D0, K0, Dz, R, n, alpha, Ks, theta_r, theta_s, method):
    # Solve for theta_{-1} from the top boundary condition
    theta_m1 = theta_0 + (R - K0) * Dz / D0

    # Convert ghost-node theta to effective saturation
    S = S_theta(theta_m1, theta_r, theta_s)

    # Evaluate hydraulic functions at the ghost node
    K_0 = K_theta(S, Ks, n, method)
    h_0 = h_theta(S, n, alpha, method)
    dtheta_dh_0 = dtheta_dh_theta(h_0, theta_s, theta_r, n, alpha, method)

    D_m1 = D_theta(K_0, dtheta_dh_0)
    K_m1 = K_0

    return theta_m1, D_m1, K_m1


# -----------------------------------------------------------------------------
# Standalone inspection block.
# Reconstructs the first-node state for the run_0000 baseline at hour 1 of
# day 1, then computes the ghost-node quantities so the output can be eyeballed.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    # run_0000 parameters
    theta_s = 0.33
    theta_r = 0.0
    Ks = 170.0        # cm/day
    alpha = 0.035     # 1/cm
    n = 2.267
    method = 2
    Dz = 2.5          # cm

    # First day's irrigation: Q_0 = 12 m3/day, Area = 308 m2
    #   R_eval = Q/Area (m/day) -> *100 -> cm/day  (matches Sim_column.m)
    Area = 308.0
    R = (12.0 / Area) * 100.0   # cm/day

    # First-node state at the uniform initial condition theta_i = 0.14
    theta_0 = 0.14
    S0 = S_theta(theta_0, theta_r, theta_s)
    K0 = K_theta(S0, Ks, n, method)
    h0 = h_theta(S0, n, alpha, method)
    dth0 = dtheta_dh_theta(h0, theta_s, theta_r, n, alpha, method)
    D0 = D_theta(K0, dth0)

    theta_m1, D_m1, K_m1 = theta_m1_theta(
        theta_0, D0, K0, Dz, R, n, alpha, Ks, theta_r, theta_s, method
    )

    print("R (cm/day) =", R)
    print("theta_0    =", theta_0)
    print("K0         =", K0)
    print("D0         =", D0)
    print("---- ghost node ----")
    print("theta_m1   =", theta_m1)
    print("K_m1       =", K_m1)
    print("D_m1       =", D_m1)
