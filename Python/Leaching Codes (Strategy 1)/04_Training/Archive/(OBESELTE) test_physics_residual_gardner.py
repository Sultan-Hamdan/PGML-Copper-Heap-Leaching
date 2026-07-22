"""
test_physics_residual_gardner.py
================================
Verification for the Gardner physics residual. Two checks:

TEST 1 (boundary stability): the OLD ghost-node top treatment explodes under
  Gardner (~1e172). The NEW one-sided top treatment must return a FINITE,
  physical value on the same state. We assert finiteness and a sane magnitude.

TEST 2 (end-to-end): across all timesteps of clean run_0000 and run_0001, the
  full Gardner residual must be (a) finite everywhere and (b) non-zero (the
  structural-uncertainty signal). A machine-zero would mean Gardner is not
  actually different from the VGM data -> bug.

Run from a directory where hydraulics_gardner.py, pred_correct_gardner.py and
physics_residual_gardner.py are importable, with run_0000.mat / run_0001.mat
present (adjust DATA_DIR as needed).
"""
import numpy as np
import scipy.io as sio
from physics_residual_gardner import physics_residual_gardner
from hydraulics_gardner import K_gardner, D_gardner

# ---- run_0000 / Strategy-1 params ----
theta_s, theta_r = 0.33, 0.0
Ks, Dz, Dt = 170.0, 2.5, 1.0 / 24.0
alpha_g, m_g = 4.0047, 16.4094          # Gardner offline-fit constants
dth = theta_s - theta_r
def se2th(S): return theta_r + S * dth

DATA_DIR = "."   # adjust to where run_000x.mat live

# -----------------------------------------------------------------------------
print("TEST 1: ghost-free top node returns a FINITE, physical value")
print("        (old ghost approach gave ~1e172 on this state)")
# Reconstruct the exact baseline that exploded
theta = np.full(221, 0.14)
Area = 308.0
R = (12.0 / Area) * 100.0               # first-day irrigation, cm/day
res = physics_residual_gardner(theta, theta, R, theta_s, theta_r,
                               Ks, alpha_g, m_g, Dz, Dt)
top = abs(res[0])
ok1 = np.all(np.isfinite(res)) and top < 1.0
print(f"  max|res| = {np.max(np.abs(res)):.3e}   top node |res[0]| = {top:.3e}")
print(f"  {'PASS' if ok1 else 'FAIL'} (finite and physical)\n")

# -----------------------------------------------------------------------------
print("TEST 2: full residual finite & non-zero across all timesteps, both runs")
all_ok = True
for fn in ["run_0000", "run_0001"]:
    m = sio.loadmat(f"{DATA_DIR}/{fn}.mat")
    Se_out = m["Se_out"]; R_out = m["R_out"].ravel()
    worst = 0.0; finite = True
    for t in range(Se_out.shape[1] - 1):
        th_t = se2th(Se_out[:, t]); th_p = se2th(Se_out[:, t + 1]); R = R_out[t + 1]
        r = physics_residual_gardner(th_t, th_p, R, theta_s, theta_r,
                                     Ks, alpha_g, m_g, Dz, Dt)
        finite &= np.all(np.isfinite(r))
        worst = max(worst, np.max(np.abs(r)))
    nonzero = worst > 1e-3
    ok = finite and nonzero
    all_ok &= ok
    print(f"  {fn}: finite={finite}  worst|res|={worst:.3e}  non-zero={nonzero}  "
          f"{'PASS' if ok else 'FAIL'}")

print()
print("ALL CHECKS PASS" if (ok1 and all_ok) else "SOME CHECKS FAILED")
