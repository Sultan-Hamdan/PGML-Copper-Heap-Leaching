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
- **`Leaching Codes (Strategy 1)/`**
  - **`01_Simulator/`**
    - `hydraulics.py` : VGM/Gardner constitutive relations
    - `thomas_kernel.py` : tridiagonal (Thomas) solver
    - `theta_m1_theta.py` : saturation update step
    - `pred_correct.py` : predictor-corrector integration
    - `sim_column.py` : single-column forward simulation
    - `sim_gpu.py` : GPU batched forward solver (CuPy)
  - **`02_Data_Generator/`**
    - `generate_dataset.py` : batched runs to strategy1.h5
  - **`03_Preprocessing/`**
    - `preprocess.py` : dataset prep, writes scaler.json + split.json
  - **`04_Training/`**
    - `train_pgml.py` : PGML training loop
    - `physics_loss.py` : Gardner physics-residual penalty
    - `physics_residual_gardner.py` : Gardner residual computation
    - `test_physics_residual.py` : residual unit test
    - `test_physics_residual_gardner.py` : Gardner residual unit test
  - **`05_Evaluation/`**
    - `evaluate.py` : metrics and figures
    - `benchmark_forward_step.py` : forward-step speedup benchmark
    - `benchmark_rollout.py` : full-rollout speed benchmark
    - `benchmark_b.py` : controller decision pipeline benchmark
    - `check_steadystate_coverage.py` : steady-state coverage check
    - `diagnose_rollout_drift.py` : open-loop drift diagnostic
    - `sweep_lambda.py` : lambda trade-off sweep
- **`Leaching Codes (Strategy 2)/`** : Strategy 2 pipeline (space-filling coverage of Q and theta_i, controlled irrigation schedules).
- **`verify_all.py`** : verifies the Python simulator port (CPU and GPU) against the 201 MATLAB reference runs

**`Presentations/`** : supervisor meeting decks (PDF)

## Method
The hydraulic transport layer is split across two constitutive relations:

- **Van Genuchten-Mualem (VGM)** drives data generation (considered ground truth).
- **Gardner** drives the controller and the surrogate physics penalty (structural uncertainty signal and NMPC consistency).

The surrogate is a 4-layer MLP (width 512) trained with a physics-residual weight lambda = 100, the knee of the accuracy/physics trade-off.

## Installation
Can be used in a terminal from the repo root. In VS Code, used in terminal with `` Ctrl+` `` (PowerShell on Windows) using the following:

PyTorch nightly and CuPy come from their own package indexes, so install them first:

    pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
    pip install cupy-cuda13x

Then the remaining packages:

    pip install -r requirements.txt

Tested on: Python 3.14, CUDA 13.2, RTX 5060 Laptop GPU.

## Data and model

Trained surrogate checkpoint is included in 05_Evaluation:

- `pgml_lam100.pt` : PGML surrogate (4-layer MLP, lambda = 100), 4 MB.

Training datasets are hosted on OneDrive (click on hyperlink for download):

- [`strategy1.h5`](https://1drv.ms/u/c/b1dd7053cf2fa097/IQD7zMgEJCj5QalOYTCVYUetAVyODMCobTd87nXSmFX7t1I?e=EKgTeg) : 201 runs, 171 MB.
- [`strategy2.h5`](https://1drv.ms/u/c/b1dd7053cf2fa097/IQCyyPBhvtUlQK3b8-rAnTTJASbimsEb2RhYLCU3Cgs5-0U?e=jktx0h) : 512 runs, 435 MB.

The training datasets can also be regenerated from runs stored in `02_Data_Generator/generate_dataset.py`.

## Status
- Strategy 1 (fixed soil; vary Q and theta_i): complete.
- Strategy 2 (space-filling coverage of Q and theta_i; controlled irrigation schedules): pending.
- Strategy 3 (vary Ks, alpha, n via Sobol sampling): pending.
