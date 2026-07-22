![Python](https://img.shields.io/badge/python-3.14-blue)
![CUDA](https://img.shields.io/badge/CUDA-13.2-green)
![PyTorch](https://img.shields.io/badge/PyTorch-nightly%20cu128-orange)
![MATLAB](https://img.shields.io/badge/MATLAB-R2026a-red)
# PGML Copper Heap Leaching
**Physics-Guided Machine Learning for Sustainable Heap Leaching Control** <br>
MSc Chemical Engineering Research Project, UCL. <br>
Author: Sultan Alhamdan <br>
Supervisor: Dr. Paulina Quintanilla
## Overview
This project replaces the slow online soil identification step in a
conventional nonlinear model predictive control (NMPC) loop for copper heap
leaching utilising a physics-guided machine learning (PGML) surrogate. The
surrogate is trained on the Richards/Gardner constitutive relations and
predicts column hydraulic state in sub-second time, versus the 1.4 to 3.5
second per-column for the conventional approach (baseline: Olivares et al., 2025).
## Repository structure
**`Python/`**
- **`Leaching Codes (Strategy 1)/`** : Strategy 1 pipeline (fixed soil; vary Q and theta_i).
- **`Leaching Codes (Strategy 2)/`**
  - **`00_MATLAB_Ports/`**
    - `hydraulics.py` : VGM/Gardner constitutive relations
    - `pred_correct.py` : predictor-corrector integration
    - `sim_column.py` : single-column forward simulation
    - `theta_m1_theta.py` : saturation update step
  - **`01_Simulator/`**
    - `sim_gpu.py` : GPU batched forward solver (CuPy)
    - `thomas_kernel.py` : tridiagonal (Thomas) solver
  - **`02_Data_Generator/`**
    - `generate_dataset.py` : batched runs to strategy2.h5
  - **`03_Preprocessing/`**
    - `preprocess.py` : dataset prep, writes scaler.json + split.json
    - `scaler.json`, `split.json` : preprocessing outputs
  - **`04_Training/`**
    - `measure_ranges.py` : input/output range measurement
    - `physics_loss.py` : Gardner physics-residual penalty
    - `train_pgml.py` : PGML training loop
  - **`05_Evaluation/`**
    - `evaluate.py` : metrics and figures
    - **`benchmarks/`**
      - `benchmark_rollout.py` : full-rollout speed benchmark
  - **`06_Control/`**
    - `pgml_closed_loop.py` : closed-loop control run
    - `pgml_lam0p4.pt` : PGML surrogate checkpoint (4-layer MLP, lambda = 0.4)
    - `plot_closed_loop.m` : MATLAB closed-loop plotting
    - `closed_loop_surface.csv`, `closed_loop_surface.mat` : closed-loop surface outputs
    - `closed_loop_traj.csv` : closed-loop trajectory output

**`Presentations/`** : supervisor meeting decks (PDF)
## Method
The hydraulic transport layer is split across two constitutive relations:
- **Van Genuchten-Mualem (VGM)** drives data generation.
- **Gardner** drives the controller and the surrogate physics penalty.

The surrogate is a 4-layer MLP (width 512) trained with a Gardner physics-residual penalty (lambda = 0.4).
## Installation
In VS Code, use the terminal as follows.

PyTorch nightly and CuPy come from their own package indexes, so install them first:

    pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
    pip install cupy-cuda13x

Then the remaining packages:

    pip install -r requirements.txt

Tested on: Python 3.14, CUDA 13.2, RTX 5060 Laptop GPU.
## Data and model
Trained surrogate checkpoint is included in `06_Control/`:
- `pgml_lam0p4.pt` : PGML surrogate (4-layer MLP, lambda = 0.4), 4 MB.
Training datasets are hosted on OneDrive (click on hyperlink for download):
- [`strategy1.h5`](https://1drv.ms/u/c/b1dd7053cf2fa097/IQD7zMgEJCj5QalOYTCVYUetAVyODMCobTd87nXSmFX7t1I?e=EKgTeg) : 201 runs, 171 MB.
- [`strategy2.h5`](https://1drv.ms/u/c/b1dd7053cf2fa097/IQCyyPBhvtUlQK3b8-rAnTTJASbimsEb2RhYLCU3Cgs5-0U?e=jktx0h) : 512 runs, 435 MB.
## Status
- Strategy 1 (fixed soil; vary Q and theta_i): complete.
- Strategy 2 (space-filling coverage of Q and theta_i; controlled irrigation schedules): pending.
- Strategy 3 (vary Ks, alpha, n via Sobol sampling): pending.
