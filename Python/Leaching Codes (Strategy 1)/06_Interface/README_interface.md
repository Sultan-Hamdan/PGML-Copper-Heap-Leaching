# PGML-MPC Demo Interface
# CENG0057 -- UCL MSc Chemical Engineering
# Supervisor: Dr. Paulina Quintanilla
# 2026.06.16

## What this is

A lightweight Streamlit interface that wraps the existing PGML pipeline.
Set an irrigation schedule (Q per day), press Run, and see:

  - Se(z,t) heatmap from the VGM simulator (ground truth)
  - Se(z,t) heatmap from the PGML prediction
  - Absolute error map
  - Timing: how long each took (ms) and speedup factor
  - MSE and physics residual for the PGML prediction

No sealed files are modified. This is a pure wrapper.

---

## Setup

1. Install Streamlit (if not already installed):

       pip install streamlit matplotlib

2. Place app.py in:

       Leaching Codes\06_Interface\app.py

3. Confirm that sweep_ckpts\pgml_lam100.pt exists in 05_Evaluation.
   Confirm that scaler.json exists in 03_Preprocessing.

---

## Run

Open a PowerShell terminal in Leaching Codes\06_Interface\ and run:

    streamlit run app.py

The browser will open automatically at http://localhost:8501

---

## Path defaults

The app resolves paths relative to its own location. If your folder layout
matches the standard CENG0057 structure, no changes are needed.

If paths differ, update them in the Streamlit sidebar at runtime -- no
code changes required.

---

## Notes

- Strategy 1 only (fixed soil: Ks=170, alpha=0.035, n=2.267, theta_i=0.14).
- Q range: 5 to 80 L/h per day (slider). Use the per-day mode for a
  non-uniform schedule.
- Physics residual uses Gardner params (m=2.0, alpha_g=0.01) matching
  the PGML training loss in physics_loss.py.
- Both simulator and PGML run on GPU if available. Timing reflects actual
  wall time including GPU sync.

---

## Changelog

2026.06.16  v1.0  Initial wrapper. Q slider/per-day schedule, Se heatmaps,
                  timing, MSE, physics residual.
