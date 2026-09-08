"""
physics_loss.py
===============
Gardner physics residual, batched and gradient-safe, for the PGML physics term.
Runs a whole BATCH of runs at once on the GPU, and gradients flow through it so
the physics term can steer the network.

STRUCTURE. The scheme is a finite-volume flux balance. With flux positive
downward, q = K - D*(dtheta/dz), one half-level of the update at any node is

    theta_i  <-  theta_i + (Dt / (2*Dz)) * ( q_{i-1/2} - q_{i+1/2} )
    q_{i+1/2} = (K_{i+1} + K_i)/2  -  ((D_{i+1} + D_i)/2) * (theta_{i+1} - theta_i)/Dz

and Crank-Nicolson sums that increment at level j and at level j+1. Every
boundary in this scheme is therefore a statement about a face flux, and must be
derived as one rather than rearranged from a ghost node:

    top    : q_surface = R exactly. Total flux, advective and diffusive
             together, so no separate K term survives on that face. No ghost
             node, no division by D0, no evaluation of K or D off the grid.
    bottom : zero theta-gradient, so the base face carries gravity only,
             q_base = K_N. Free drainage.

Note on the top cell: node 0 is treated as a FULL cell of width Dz, so the
surface face sits half a cell above the surface. That is the plant's convention
(the plant's ghost sits at -Dz and reuses the interior stencil) and is inherited
deliberately, so that any mismatch at node 0 is attributable to Gardner against
van Genuchten rather than to a discretisation choice of ours. A strictly
conservative vertex-centred form would give node 0 a half cell; we do not take
it.

Why not a ghost node at the top. The ghost form needs
theta_{-1} = theta_0 + (R - K0)*Dz/D0, and D0 is proportional to Se0**(m-1) with
m between 5 and 13 across the soil box. D0 therefore collapses toward zero at
any dry surface node faster than a finite prefactor can compensate, while R
stays of order 10 cm/day. Across the corners of the box the ghost lands outside
[0, 1] in four cases out of five, including the reference soil, reaching
Se_{-1} ~ 3.4e+03. Taming it would need alpha_g near 2.9e-04, three orders below
any identified value, and alpha_g is a parameter under measurement. The flux
balance removes the ghost entirely, imposes the surface flux exactly rather than
up to a factor (D0 + D_{-1})/(2*D0), and leaves the top row strictly diagonally
dominant with unit margin.

Gardner params m and alpha_g are explicit inputs, supplied per run by the
caller. Note that `m` here is Olivares' beta, the conductivity exponent. It is
not a free parameter: it equals delta(n) = 3 + 2/(n - 2), computed from the
sampled van Genuchten n. train.py evaluates that expression and passes the
result in, so this file never derives it.

Used as the physics term in the convex loss in train.py:
    (1 - lam) * data_n + lam * phys_n

VETTING. This file is the only implementation of the operator.
    It is checked by differencing it numerically against the closed form
    written out in the STRUCTURE block above, node by node, at machine
    precision. Checking against the algebra rather than against a second
    implementation is deliberate: two ports of the same wrong derivation agree
    with each other perfectly, so agreement between implementations cannot
    detect an error in the maths. Differencing against the closed form does.

Status: verified against the closed form above, node by node, at machine
        precision
Depends on: torch

Changelog
---------
2026.09.07  v1.3  Repository release. Header wording only, no code changed.
2026.07.31  v1.2  Upper domain guard. Se is now clamped to [SE_LO, SE_HI] with
                  SE_HI = 1 wherever it enters K or D, completing a guard that
                  previously existed only at the lower end. The network has no
                  output activation by design, so predictions are unbounded; in
                  train mode BatchNorm renormalises each layer to unit variance
                  and predicted Se lands well outside [0, 1], where Se**m with m
                  up to 13 took the mean squared residual to 1.5e15 and made it
                  vary 34-fold with batch size. The guard is confined to the
                  constitutive evaluation: theta_pred is not clamped where it
                  enters the residual linearly, so over-saturation is still
                  penalised, just linearly rather than as a power law, and the
                  gradient survives. Neither 2026.07.30 fix is disturbed: R
                  enters G_diff linearly and is never clamped, so the standing
                  dR test is unchanged, and the bottom-node cancellation uses
                  the same D tensor on both sides.
2026.07.30  v1.1  Two maths fixes, both derived from the flux balance rather
                  than patched by inspection.
                  gardner_D: the prefactor was the reciprocal of the published
                  group. C = (dtheta*alpha_g)/(m*Ks) becomes
                  C = (m*Ks)/(alpha_g*dtheta), which is Olivares 2025 Eq. 14,
                  traceable to Tracy 2010 Eqs. 5 and 8. Units are now cm2/day,
                  as the scheme requires.
                  build_A_c_gardner top node: sign on the surface-flux group in
                  c[0]. Plus w1*G_diff becomes minus w1*G_diff. Previously R
                  cancelled exactly between b and c, so the residual was blind
                  to irrigation. After the fix d(res_0)/dR = -Dt/Dz and
                  (b + c)_0 carries 4*w2*R = Dt*R/Dz.
                  build_b_gardner UNCHANGED: its top node was already the
                  flux-balance form, confirmed by symbolic differencing.
                  Comment and docstring corrections so the wrong convention
                  cannot be re-derived from them.
2026.06.13  v1.0  Initial torch implementation of the Gardner residual.
                  Batched (B, Nz+1), interior vectorised with no node loop,
                  gradient-safe clamp.
"""

import torch


# =============================================================================
# Gardner hydraulics (torch, batched)
#   K(Se) = Ks * Se**m
#   D(Se) = C  * Se**(m-1),   C = (m * Ks) / (alpha_g * dtheta)
#
# C is Olivares 2025 Eq. 14. Derivation, from Gardner Se**m = exp(alpha_g*h):
#   h         = (m/alpha_g) * ln(Se)
#   dtheta/dh = dtheta * (alpha_g/m) * Se
#   D := K / (dtheta/dh) = [ m*Ks / (alpha_g*dtheta) ] * Se**(m-1)
# Units: [-]*[cm/day] / ([1/cm]*[-]) = cm2/day, which is what the scheme needs,
# since C multiplies Dt/(2*Dz**2) [day/cm2] and the product is added to a
# dimensionless theta. The earlier form was the reciprocal of this.
#
# Se has shape (B, Nz+1); Ks, m and alpha_g are either scalars or shape (B, 1)
# per-run tensors. Broadcasting handles both.
# =============================================================================
# The physical domain of the constitutive relations. K and D are functions of
# effective saturation and are defined on [0, 1]; outside it they are not wrong
# so much as undefined, and the exponent m reaching 13 turns an out-of-domain
# argument into a several-decade error in a single coefficient.
#
# The guard is applied ONLY where Se enters K and D. theta_pred is left
# untouched everywhere it appears linearly in the residual, so the residual
# still responds to an over-saturated prediction: it grows linearly in theta
# rather than as Se**m, and the gradient survives, because every row of the
# implicit matrix keeps a diagonal margin of at least one whatever the
# coefficients do.
#
# The upper end is the completion of a guard that already existed at the lower
# end. Note that D is NOT singular at saturation for Gardner: dtheta/dh is
# dtheta*(alpha_g/m)*Se, finite and non-zero at Se = 1, so D(1) = C exactly and
# the saturated extension is continuous. That is a Gardner property and does
# not carry over to van Genuchten, where the same argument would fail.
#
# What the guard does NOT do: it does not penalise the bound violation itself.
# Above saturation the residual measures conservation error under saturated
# coefficients, and the bound is left to the data term. Under a light-guardrail
# physics weight that is the intended division of labour, but it means a small
# physics loss no longer implies a physical prediction, so the fraction of
# nodes above saturation has to be monitored separately during training.
SE_LO = 1e-9
SE_HI = 1.0


def gardner_K(Se, Ks, m):
    return Ks * torch.pow(Se, m)


def gardner_D(Se, Ks, m, alpha_g, dtheta):
    C = (m * Ks) / (alpha_g * dtheta)
    return C * torch.pow(Se, m - 1.0)


# =============================================================================
# build_b (torch, batched, vectorised)
#   theta_v : (B, Nz+1) start-of-hour state
#   R       : (B, 1) or scalar irrigation rate, PHYSICAL units (cm/day)
# Returns b : (B, Nz+1)
# =============================================================================
def build_b_gardner(theta_v, R, theta_s, theta_r, Ks, m, alpha_g, Dz, Dt):
    dtheta = theta_s - theta_r

    Se = (theta_v - theta_r) / dtheta
    Se = Se.clamp(min=SE_LO, max=SE_HI)      # physical domain guard, both ends
    K = gardner_K(Se, Ks, m)
    D = gardner_D(Se, Ks, m, alpha_g, dtheta)

    w1 = Dt / (2.0 * Dz ** 2)
    w2 = Dt / (4.0 * Dz)

    b = torch.zeros_like(theta_v)

    # --- interior nodes i = 1..Nz-1, all at once via slices ---
    # forward face D at i+1/2, backward face D at i-1/2
    Df = (D[:, 2:] + D[:, 1:-1]) / 2.0       # faces above interior nodes
    Db = (D[:, 1:-1] + D[:, :-2]) / 2.0      # faces below interior nodes
    b[:, 1:-1] = (theta_v[:, 1:-1]
                  + w1 * (Df * (theta_v[:, 2:] - theta_v[:, 1:-1])
                          - Db * (theta_v[:, 1:-1] - theta_v[:, :-2]))
                  - w2 * (K[:, 2:] - K[:, :-2]))

    # --- top node i = 0: the surface face flux is known and equals R ---
    # Flux balance:  theta_0 + (Dt/2Dz) * ( R - q_{1/2} )
    # expands to:    theta_0 + w1*Df0*(theta_1 - theta_0)
    #                        - w2*(K_1 + K_0) + 2*w2*R
    # Note the conductivity group is a SUM, not the difference the interior
    # stencil carries. In the interior the face means telescope into
    # (K_{i+1} - K_{i-1})/2; at the top the upper face has no K at all, because
    # R has absorbed the whole flux.
    # The form written below is algebraically identical to that expansion, with
    # the -2*w2*K_0 carried inside -w1*G_diff and the +w2*K_0 supplied by
    # -w2*(K_1 - K_0). Confirmed equal by symbolic differencing. This node was
    # already correct before the 2026.07.30 patch and was NOT changed by it.
    Df0 = (D[:, 1] + D[:, 0]) / 2.0
    R_flat = R.squeeze(-1) if (torch.is_tensor(R) and R.dim() > 1) else R
    G_diff = Dz * (K[:, 0] - R_flat)          # shape (B,)
    b[:, 0] = (theta_v[:, 0]
               + w1 * (Df0 * (theta_v[:, 1] - theta_v[:, 0]) - G_diff)
               - w2 * (K[:, 1] - K[:, 0]))

    # --- bottom node i = Nz: zero theta-gradient, base face carries K_N only ---
    Df_b = D[:, -1]
    Db_b = (D[:, -1] + D[:, -2]) / 2.0
    b[:, -1] = (theta_v[:, -1]
                + w1 * (Df_b * 0.0 - Db_b * (theta_v[:, -1] - theta_v[:, -2]))
                - w2 * (K[:, -1] - K[:, -2]))
    return b


# =============================================================================
# build_A_c (torch, batched, vectorised)
# Returns the three diagonals (each (B, Nz+1)) and c (B, Nz+1).
# =============================================================================
def build_A_c_gardner(theta_v, R, theta_s, theta_r, Ks, m, alpha_g, Dz, Dt):
    dtheta = theta_s - theta_r

    Se = (theta_v - theta_r) / dtheta
    Se = Se.clamp(min=SE_LO, max=SE_HI)      # K(Se>=1) = Ks, saturated extension
    K = gardner_K(Se, Ks, m)
    D = gardner_D(Se, Ks, m, alpha_g, dtheta)

    w1 = Dt / (2.0 * Dz ** 2)
    w2 = Dt / (4.0 * Dz)

    lower = torch.zeros_like(theta_v)
    main = torch.zeros_like(theta_v)
    upper = torch.zeros_like(theta_v)
    c = torch.zeros_like(theta_v)

    # --- interior nodes ---
    Dff = (D[:, 2:] + D[:, 1:-1]) / 2.0
    Dbf = (D[:, 1:-1] + D[:, :-2]) / 2.0
    main[:, 1:-1] = 1.0 + w1 * (Dff + Dbf)
    upper[:, 1:-1] = -w1 * Dff
    lower[:, 1:-1] = -w1 * Dbf
    c[:, 1:-1] = -w2 * (K[:, 2:] - K[:, :-2])

    # --- top node: only the forward face couples implicitly; the surface face
    # is the known flux R, so lower0 = 0 and the rest moves to c.
    # THE SIGN ON w1*G_diff IS NEGATIVE. The Crank-Nicolson derivation puts
    # +w1*G on the left-hand side at level j+1, so it arrives on the right-hand
    # side as -w1*G. A positive sign here makes R cancel exactly against the
    # -w1*G carried in b, leaving only w1*Dz*(K_pred[0] - K_t[0]) and a residual
    # that cannot see irrigation at all. The bottom node is the internal control
    # for this: its ghost term also flips sign on the move across the equals
    # sign, and does so correctly.
    # The line below is algebraically identical to the derived
    #     c_0 = -w2*(K_1 + K_0) + 2*w2*R
    # and gives (b + c)_0 a surface term of 4*w2*R = 2*w1*Dz*R = Dt*R/Dz.
    # Standing test: d(b+c)_0/dR must equal Dt/Dz exactly, every other node zero.
    Dff0 = (D[:, 1] + D[:, 0]) / 2.0
    R_flat = R.squeeze(-1) if (torch.is_tensor(R) and R.dim() > 1) else R
    G_diff = Dz * (K[:, 0] - R_flat)
    main[:, 0] = 1.0 + w1 * Dff0
    upper[:, 0] = -w1 * Dff0
    lower[:, 0] = 0.0
    c[:, 0] = -w2 * (K[:, 1] - K[:, 0]) - w1 * G_diff

    # --- bottom node ---
    # main and c both carry +w1*Dff_b*theta. build_A_c is called with
    # theta_v = theta_pred, so those are the same tensor and cancel exactly in
    # A@theta_pred - (b + c), leaving the zero-gradient form. Confirmed equal to
    # the flux balance with q_base = K_N.
    Dff_b = D[:, -1]
    Dbf_b = (D[:, -1] + D[:, -2]) / 2.0
    main[:, -1] = 1.0 + w1 * (Dff_b + Dbf_b)
    lower[:, -1] = -w1 * Dbf_b
    c[:, -1] = -w2 * (K[:, -1] - K[:, -2]) + w1 * Dff_b * theta_v[:, -1]

    return lower, main, upper, c


# =============================================================================
# physics_residual_gardner_torch : res = A @ theta_pred - (b + c), batched
# =============================================================================
def physics_residual_gardner_torch(theta_t, theta_pred, R, theta_s, theta_r,
                                    Ks, m, alpha_g, Dz, Dt):
    b = build_b_gardner(theta_t, R, theta_s, theta_r, Ks, m, alpha_g, Dz, Dt)
    lower, main, upper, c = build_A_c_gardner(theta_pred, R, theta_s, theta_r,
                                              Ks, m, alpha_g, Dz, Dt)
    Ax = main * theta_pred
    Ax[:, :-1] = Ax[:, :-1] + upper[:, :-1] * theta_pred[:, 1:]
    Ax[:, 1:] = Ax[:, 1:] + lower[:, 1:] * theta_pred[:, :-1]
    return Ax - (b + c)


# =============================================================================
# physics_loss : scalar loss from the residual (mean squared residual).
# This is the term multiplied by lambda and added to the data loss.
# =============================================================================
def physics_loss(theta_t, theta_pred, R, theta_s, theta_r,
                 Ks, m, alpha_g, Dz, Dt):
    res = physics_residual_gardner_torch(theta_t, theta_pred, R, theta_s,
                                         theta_r, Ks, m, alpha_g, Dz, Dt)
    return torch.mean(res ** 2)
