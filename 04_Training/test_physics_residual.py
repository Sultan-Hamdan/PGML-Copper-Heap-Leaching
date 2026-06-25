"""Corrected equivalence test: residual must MATCH the simulator's own
fixed-point residual (the data's inherent ~1e-5 floor), AND must drop to
machine-zero on fully-converged states. Both confirm the residual is correct."""
import numpy as np
import scipy.io as sio
from pred_correct import build_b, build_A_c, pred_correct
from physics_residual import physics_residual

m = sio.loadmat('run0000_fresh.mat')
Se_out = m['Se_out']; R_out = m['R_out'].ravel()
ts,tr = 0.33,0.0; Ks,al,n = 170.0,0.035,2.267; meth,Dz,Dt = 2,2.5,1/24
dth = ts-tr; se2th = lambda S: tr+S*dth

print("TEST 1: residual on a FULLY CONVERGED state -> must be ~machine zero")
t=200; th_t=se2th(Se_out[:,t]); R=R_out[t+1]
b=build_b(th_t,R,ts,tr,Ks,al,n,meth,Dz,Dt)
th=th_t.copy()
for _ in range(300): th=pred_correct(th,b,R,ts,tr,Ks,al,n,meth,Dz,Dt,solver='full')
res=physics_residual(th_t,th,R,ts,tr,Ks,al,n,meth,Dz,Dt)
print(f"  max|res| on converged state = {np.max(np.abs(res)):.3e}  {'PASS' if np.max(np.abs(res))<1e-9 else 'FAIL'}")

print("\nTEST 2: residual on stored data == simulator's own 20-iter residual")
print("  (proves our residual measures the TRUE scheme inconsistency, not a bug)")
print(f"  {'t':>5} {'our_res':>12} {'sim_res':>12} {'match':>8}")
allmatch=True
for t in [50,100,200,400,500]:
    th_t=se2th(Se_out[:,t]); th_tp1=se2th(Se_out[:,t+1]); R=R_out[t+1]
    our=np.max(np.abs(physics_residual(th_t,th_tp1,R,ts,tr,Ks,al,n,meth,Dz,Dt)))
    # simulator's own residual on the same stored next-state:
    b=build_b(th_t,R,ts,tr,Ks,al,n,meth,Dz,Dt)
    lo,ma,up,cc=build_A_c(th_tp1,R,ts,tr,Ks,al,n,meth,Dz,Dt)
    Ax=ma*th_tp1; Ax[:-1]+=up[:-1]*th_tp1[1:]; Ax[1:]+=lo[1:]*th_tp1[:-1]
    sim=np.max(np.abs(Ax-(b+cc)))
    ok=np.isclose(our,sim,rtol=1e-9)
    allmatch&=ok
    print(f"  {t:>5} {our:>12.3e} {sim:>12.3e} {'OK' if ok else 'X':>8}")
print(f"\n  {'ALL MATCH -> residual assembly VERIFIED' if allmatch else 'mismatch'}")
