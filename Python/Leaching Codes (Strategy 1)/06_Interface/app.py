"""
app.py
======
06_Interface -- PGML-MPC Demo Wrapper

Streamlit interface for the PGML heap leaching surrogate.
Runs the VGM simulator and PGML side by side on a user-defined Q schedule,
then shows Se(z,t) heatmaps, timing, MSE, and physics residual.

This file is a wrapper only. It does not modify any sealed file.
All logic is imported from existing scripts in sibling phase folders.

Usage (from Leaching Codes\06_Interface\):
    streamlit run app.py

Adjust PATHS block below if your folder layout differs.

Depends on: sim_gpu.py, preprocess.py, train_pgml.py, physics_loss.py
Changelog
---------
2026.06.16  v1.0  Initial wrapper. Q slider, rollout, heatmaps, timing, MSE,
                  physics residual. Streamlit layout, matplotlib plots.
"""

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st
import torch

# =============================================================================
# PATHS -- adjust these to match your machine if needed
# =============================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
_LEACHING = os.path.dirname(_HERE)   # Leaching Codes/

for _folder in ("01_Simulator", "03_Preprocessing", "04_Training"):
    _p = os.path.join(_LEACHING, _folder)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Default paths -- override via sidebar if needed
DEFAULT_CKPT   = os.path.join(_LEACHING, "05_Evaluation", "sweep_ckpts", "pgml_lam100.pt")
DEFAULT_SCALER = os.path.join(_LEACHING, "03_Preprocessing", "scaler.json")

# =============================================================================
# SIMULATION CONSTANTS (Strategy 1)
# =============================================================================
SIM = dict(
    theta_s=0.33, theta_r=0.0, theta_i=0.14,
    Ks=170.0, alpha=0.035, n=2.267,
    H=5.50, Area=308.0, Dz=2.5,
    Dt=1.0 / 24.0, D_H=24, tf=25, n_iter=20,
)
NZ1     = int(SIM["H"] * 100 / SIM["Dz"]) + 1   # 221
N_STEPS = SIM["tf"] * SIM["D_H"]                  # 600

# Gardner params (for physics residual; matches physics_loss.py defaults)
GARDNER = dict(
    m=2.0, alpha_g=0.01,
    Ks=SIM["Ks"], theta_s=SIM["theta_s"], theta_r=SIM["theta_r"],
    Dz=SIM["Dz"], Dt=SIM["Dt"],
)

# z-axis for plots (height in cm, top = 0)
Z_AXIS = np.linspace(0, SIM["H"] * 100, NZ1)   # 0 to 550 cm

# =============================================================================
# CACHED LOADERS (run once per session)
# =============================================================================
@st.cache_resource
def load_imports():
    """Import GPU/ML modules once and return them."""
    from sim_gpu   import run_batch, xp, ON_GPU, _sync
    from preprocess import load_scalers, apply_scaler, INPUT_DIM, OUTPUT_DIM
    from train_pgml import MLP
    from physics_loss import physics_loss as gardner_physics_loss
    return run_batch, xp, ON_GPU, _sync, load_scalers, apply_scaler, INPUT_DIM, OUTPUT_DIM, MLP, gardner_physics_loss


@st.cache_resource
def load_model_and_scalers(ckpt_path, scaler_dir):
    """Load PGML checkpoint and scalers once."""
    _, _, _, _, load_scalers, _, INPUT_DIM, OUTPUT_DIM, MLP, _ = load_imports()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MLP(INPUT_DIM, OUTPUT_DIM,
                width=ckpt["args"]["width"],
                depth=ckpt["args"]["depth"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    scalers = load_scalers(scaler_dir)
    return model, scalers, device


# =============================================================================
# CORE FUNCTIONS
# =============================================================================
def build_R_schedule(Q_schedule_lh):
    """Convert Q schedule (L/h per day, length 25) to R (cm/h, length 25)."""
    return np.array(Q_schedule_lh, dtype=np.float64) / SIM["Area"] * 100.0


def run_simulator(Q_schedule_lh):
    """Run the VGM simulator for 25 days. Returns (Se_traj, elapsed_s)."""
    run_batch, xp, ON_GPU, _sync, *_ = load_imports()
    Q = np.array(Q_schedule_lh, dtype=np.float64)[np.newaxis, :]  # (1, 25)
    t0 = time.perf_counter()
    Se = run_batch(Q,
                   Ks=SIM["Ks"], theta_s=SIM["theta_s"], theta_r=SIM["theta_r"],
                   theta_i=SIM["theta_i"], alpha=SIM["alpha"], n=SIM["n"],
                   H=SIM["H"], Area=SIM["Area"], tf=SIM["tf"],
                   Dz=SIM["Dz"], Dt=SIM["Dt"], D_H=SIM["D_H"],
                   n_iter=SIM["n_iter"])
    _sync()
    elapsed = time.perf_counter() - t0
    Se_np = Se.get()[0] if ON_GPU else Se[0]   # (221, 600)
    return Se_np, elapsed


def run_pgml(Q_schedule_lh, model, scalers, device):
    """Run PGML rollout for 25 days. Returns (Se_traj, elapsed_s)."""
    _, _, _, _, _, apply_scaler, INPUT_DIM, _, _, _ = load_imports()

    R_schedule = build_R_schedule(Q_schedule_lh)   # (25,) cm/h

    # Scale soil params (all 0.0 for Strategy 1 since min==max)
    soil_vec = np.array([
        apply_scaler(np.array([SIM["Ks"]]),      "Ks",      scalers)[0],
        apply_scaler(np.array([SIM["alpha"]]),   "alpha",   scalers)[0],
        apply_scaler(np.array([SIM["n"]]),       "n_vgm",   scalers)[0],
        apply_scaler(np.array([SIM["theta_i"]]), "theta_i", scalers)[0],
    ], dtype=np.float32)

    Se0_val = (SIM["theta_i"] - SIM["theta_r"]) / (SIM["theta_s"] - SIM["theta_r"])
    Se_cur  = torch.full((1, NZ1), Se0_val, device=device, dtype=torch.float32)
    traj    = torch.empty(1, NZ1, N_STEPS, device=device, dtype=torch.float32)

    suffix_base = torch.tensor(soil_vec, device=device, dtype=torch.float32).unsqueeze(0)  # (1,4)

    t0 = time.perf_counter()
    with torch.no_grad():
        for t in range(N_STEPS):
            day = t // SIM["D_H"]
            R_val = float(R_schedule[day])
            R_scaled = float(apply_scaler(np.array([R_val]), "R_out", scalers)[0])
            R_t = torch.tensor([[R_scaled]], device=device, dtype=torch.float32)
            x = torch.cat([Se_cur, R_t, suffix_base], dim=1)   # (1, 226)
            Se_cur = model(x)
            traj[:, :, t] = Se_cur
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return traj[0].cpu().numpy(), elapsed   # (221, 600)


def compute_physics_residual(Se_traj, Q_schedule_lh):
    """
    Compute mean absolute Gardner physics residual over the full trajectory.
    Se_traj : (221, 600) float32 numpy
    Returns scalar (mean |residual| over all nodes and timesteps).
    """
    _, _, _, _, _, _, _, _, _, gardner_physics_loss = load_imports()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    R_schedule = build_R_schedule(Q_schedule_lh)   # (25,) cm/h

    # Build (1, 221, 600) batch for physics loss
    Se_t = torch.tensor(Se_traj, dtype=torch.float32, device=device).unsqueeze(0)

    residuals = []
    for t in range(N_STEPS - 1):
        day = t // SIM["D_H"]
        R_val = float(R_schedule[day])
        R_t = torch.full((1, 1), R_val, device=device, dtype=torch.float32)

        theta_now  = Se_t[:, :, t]   * (SIM["theta_s"] - SIM["theta_r"]) + SIM["theta_r"]
        theta_next = Se_t[:, :, t+1] * (SIM["theta_s"] - SIM["theta_r"]) + SIM["theta_r"]

        loss = gardner_physics_loss(
            theta_now, theta_next, R_t,
            theta_s=GARDNER["theta_s"], theta_r=GARDNER["theta_r"],
            Ks=GARDNER["Ks"], m=GARDNER["m"], alpha_g=GARDNER["alpha_g"],
            Dz=GARDNER["Dz"], Dt=GARDNER["Dt"],
        )
        residuals.append(loss.item())

    return float(np.mean(residuals))


def make_heatmap(Se_traj, title, vmin=0.0, vmax=1.0):
    """
    Return a matplotlib figure: Se(z,t) heatmap.
    Se_traj : (221, 600) -- rows = depth nodes, cols = timesteps
    x-axis  : time (days), y-axis : depth (cm, top=0)
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(
        Se_traj,
        aspect="auto",
        origin="upper",
        extent=[0, SIM["tf"], Z_AXIS[-1], Z_AXIS[0]],
        vmin=vmin, vmax=vmax,
        cmap="viridis",
    )
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Depth (cm)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Se (-)")
    fig.tight_layout()
    return fig


# =============================================================================
# STREAMLIT APP
# =============================================================================
st.set_page_config(page_title="PGML-MPC Demo", layout="wide")

st.title("PGML Heap Leaching Demo")
st.caption("CENG0057 -- UCL MSc Chemical Engineering -- Supervisor: Dr. Paulina Quintanilla")

# --- Sidebar: paths and Q schedule ---
with st.sidebar:
    st.header("Paths")
    ckpt_path  = st.text_input("Checkpoint (.pt)", value=DEFAULT_CKPT)
    scaler_dir = st.text_input("Scaler directory", value=os.path.dirname(DEFAULT_SCALER))

    st.header("Irrigation Schedule")
    st.caption("Set Q (L/h) for each of the 25 days. All days default to 24 L/h.")

    q_uniform = st.slider("Uniform Q for all days (L/h)", 5.0, 80.0, 24.0, step=1.0)
    use_uniform = st.checkbox("Use uniform Q for all days", value=True)

    if use_uniform:
        Q_schedule = [q_uniform] * 25
    else:
        Q_schedule = []
        for d in range(25):
            q = st.slider(f"Day {d+1}", 5.0, 80.0, 24.0, step=1.0, key=f"q_{d}")
            Q_schedule.append(q)

    st.markdown("---")
    run_button = st.button("Run simulation", type="primary", use_container_width=True)

# --- Main area placeholder before run ---
if not run_button:
    st.info("Set the irrigation schedule in the sidebar, then press **Run simulation**.")
    st.stop()

# --- Load model ---
if not os.path.isfile(ckpt_path):
    st.error(f"Checkpoint not found: {ckpt_path}\n\nUpdate the path in the sidebar.")
    st.stop()
if not os.path.isfile(os.path.join(scaler_dir, "scaler.json")):
    st.error(f"scaler.json not found in: {scaler_dir}\n\nUpdate the path in the sidebar.")
    st.stop()

with st.spinner("Loading model and scalers..."):
    try:
        model, scalers, device = load_model_and_scalers(ckpt_path, scaler_dir)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# --- Run simulator ---
sim_bar = st.progress(0, text="Running VGM simulator...")
try:
    Se_sim, sim_time = run_simulator(Q_schedule)
    sim_bar.progress(50, text="VGM simulator done.")
except Exception as e:
    st.error(f"Simulator failed: {e}")
    st.stop()

# --- Run PGML ---
sim_bar.progress(50, text="Running PGML rollout...")
try:
    Se_pgml, pgml_time = run_pgml(Q_schedule, model, scalers, device)
    sim_bar.progress(100, text="PGML rollout done.")
except Exception as e:
    st.error(f"PGML rollout failed: {e}")
    st.stop()

sim_bar.empty()

# --- Compute metrics ---
mse = float(np.mean((Se_pgml.astype(np.float64) - Se_sim) ** 2))
mae = float(np.mean(np.abs(Se_pgml.astype(np.float64) - Se_sim)))

with st.spinner("Computing physics residual..."):
    try:
        phys_res = compute_physics_residual(Se_pgml.astype(np.float32), Q_schedule)
    except Exception as e:
        phys_res = None
        st.warning(f"Physics residual failed: {e}")

speedup = sim_time / pgml_time if pgml_time > 0 else float("nan")

# =============================================================================
# OUTPUT: metrics row
# =============================================================================
st.subheader("Results")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Sim time",   f"{sim_time*1000:.1f} ms")
c2.metric("PGML time",  f"{pgml_time*1000:.1f} ms")
c3.metric("Speedup",    f"{speedup:.1f}x")
c4.metric("MSE",        f"{mse:.2e}")
c5.metric("Physics res", f"{phys_res:.2e}" if phys_res is not None else "N/A")

st.markdown("---")

# =============================================================================
# OUTPUT: heatmaps
# =============================================================================
col_sim, col_pgml = st.columns(2)

vmin = min(Se_sim.min(), Se_pgml.min())
vmax = max(Se_sim.max(), Se_pgml.max())

with col_sim:
    st.pyplot(make_heatmap(Se_sim,  "VGM Simulator  --  Se(z,t)", vmin, vmax))

with col_pgml:
    st.pyplot(make_heatmap(Se_pgml, "PGML Prediction  --  Se(z,t)", vmin, vmax))

# --- Error map ---
with st.expander("Show absolute error  |PGML - Sim|"):
    err = np.abs(Se_pgml.astype(np.float64) - Se_sim)
    fig_err, ax_err = plt.subplots(figsize=(7, 4))
    im_err = ax_err.imshow(
        err, aspect="auto", origin="upper",
        extent=[0, SIM["tf"], Z_AXIS[-1], Z_AXIS[0]],
        cmap="hot",
    )
    ax_err.set_xlabel("Time (days)")
    ax_err.set_ylabel("Depth (cm)")
    ax_err.set_title("|PGML - Sim|  absolute error", fontsize=11, fontweight="bold")
    fig_err.colorbar(im_err, ax=ax_err, label="|Se error|")
    fig_err.tight_layout()
    st.pyplot(fig_err)

# --- Q schedule display ---
with st.expander("Q schedule used"):
    fig_q, ax_q = plt.subplots(figsize=(7, 2.5))
    ax_q.bar(range(1, 26), Q_schedule, color="#4a90d9")
    ax_q.set_xlabel("Day")
    ax_q.set_ylabel("Q (L/h)")
    ax_q.set_title("Irrigation schedule", fontsize=10)
    fig_q.tight_layout()
    st.pyplot(fig_q)

st.caption(
    f"Device: {device}  |  "
    f"Checkpoint: {os.path.basename(ckpt_path)}  |  "
    f"Soil: Strategy 1 fixed (Ks={SIM['Ks']}, alpha={SIM['alpha']}, n={SIM['n']})"
)
