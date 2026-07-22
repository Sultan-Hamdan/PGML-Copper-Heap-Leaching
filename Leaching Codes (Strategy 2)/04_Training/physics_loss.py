"""
physics_loss.py
===============
PyTorch version of the Gardner physics residual (physics_residual_gardner.py).
Same maths, made training-ready: it runs a whole BATCH of runs at once on the GPU,
and gradients flow through it so the physics term can steer the network.

What changed from the NumPy reference (container only, not the maths):
    - numpy  -> torch  (np.power -> torch.pow, etc.)
    - one run, shape (Nz+1,)  ->  a batch, shape (B, Nz+1)
    - the interior node loop  ->  vectorised slicing (all nodes at once, no loop)
Everything physical is identical: Gardner K/D, the stencil, w1/w2, the top-boundary flux fix (ghost eliminated, no division by D0),
 bottom zero-grad, and res = A @ theta_pred - (b + c).

Gardner params m, alpha_g are explicit inputs (fixed for Strategy 1; per-run for Strategy 3).
Used as the physics term in train_pgml's convex loss: (1-lam)*data_n + lam*phys_n.

VETTING: the torch result MUST match the verified NumPy result to machine precision (single run AND each row of a batch).
That equivalence is the safety net -- if they diverge, the torch port has a bug.

Origin: torch port of physics_residual_gardner.py (NumPy stays as reference)
Status: under verification (equivalence + gradient checks)
Depends on: torch

Changelog
---------
2026.06.13  v1.0  Initial torch port. Batched (B, Nz+1), interior vectorised
                  (no node loop), gradient-safe clamp. Verified against the
                  NumPy reference to machine precision.
"""

import torch


# =============================================================================
# Gardner hydraulics (torch, batched)
#   K(Se) = Ks * Se**m
#   D(Se) = C  * Se**(m-1),   C = (dtheta * alpha_g) / (m * Ks)
# Se has shape (B, Nz+1); Ks, m, alpha_g are scalars (Strategy 1) or
# shape (B, 1) per-run tensors (Strategy 3). Broadcasting handles both.
# =============================================================================
def gardner_K(Se, Ks, m):
    return Ks * torch.pow(Se, m)


def gardner_D(Se, Ks, m, alpha_g, dtheta):
    C = (dtheta * alpha_g) / (m * Ks)
    return C * torch.pow(Se, m - 1.0)


# =============================================================================
# build_b (torch, batched, vectorised)
#   theta_v : (B, Nz+1) start-of-hour state
#   R       : (B, 1) or scalar irrigation rate
# Returns b : (B, Nz+1)
# =============================================================================
def build_b_gardner(theta_v, R, theta_s, theta_r, Ks, m, alpha_g, Dz, Dt):
    dtheta = theta_s - theta_r

    Se = (theta_v - theta_r) / dtheta
    Se = Se.clamp(min=1e-9)                 # gradient-safe power-law guard
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

    # --- top node i = 0: surface flux = R, ghost eliminated ---
    Df0 = (D[:, 1] + D[:, 0]) / 2.0
    R_flat = R.squeeze(-1) if (torch.is_tensor(R) and R.dim() > 1) else R
    G_diff = Dz * (K[:, 0] - R_flat)          # shape (B,)
    b[:, 0] = (theta_v[:, 0]
               + w1 * (Df0 * (theta_v[:, 1] - theta_v[:, 0]) - G_diff)
               - w2 * (K[:, 1] - K[:, 0]))

    # --- bottom node i = Nz: zero-gradient ---
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
    Se = Se.clamp(min=1e-9)
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

    # --- top node: only forward face couples; backward face is known flux ---
    Dff0 = (D[:, 1] + D[:, 0]) / 2.0
    R_flat = R.squeeze(-1) if (torch.is_tensor(R) and R.dim() > 1) else R
    G_diff = Dz * (K[:, 0] - R_flat)
    main[:, 0] = 1.0 + w1 * Dff0
    upper[:, 0] = -w1 * Dff0
    lower[:, 0] = 0.0
    c[:, 0] = -w2 * (K[:, 1] - K[:, 0]) + w1 * G_diff

    # --- bottom node ---
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
