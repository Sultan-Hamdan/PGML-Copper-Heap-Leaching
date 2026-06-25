# PGML-Copper-Heap-Leaching

**Physics-Guided Machine Learning for Sustainable Heap Leaching Control**
MSc Chemical Engineering Research Project (CENG0057), UCL
Supervisor: Dr. Paulina Quintanilla

## Overview

This project replaces the slow online soil-identification step in a
conventional nonlinear model predictive control (NMPC) loop for heap
leaching with a physics-guided machine learning (PGML) surrogate. The
surrogate is trained on the Richards/Gardner closure equations and
predicts column hydraulic state in sub-second time, versus the 1.4 to 3.5
second per-column identification cost of the conventional approach
(baseline: Olivares et al., 2025, Minerals Engineering 229:109346).

The PGML model is trained with a physics-residual penalty so its
predictions stay physically consistent, not merely data-fitted.

## Repository structure

    Data generation/      MATLAB column simulator and GPU batch pipeline
      *.m                 Richards solver, hydraulics, predictor-corrector
      sim_gpu.py          GPU forward solver (CuPy)
      generate_dataset.py .mat -> strategy1.h5 batch generation
      preprocess.py       dataset preparation

    Training/
      train_pgml.py       PGML training (physics-guided loss)
      physics_loss.py     Gardner physics-residual penalty
      physics_residual*.py residual computation and tests

    Evaluation/
      evaluate.py         metrics and figures
      benchmark_*.py      forward-step and rollout speed benchmarks

## Method

The hydraulic transport layer is modelled with a deliberate split:
- **Van Genuchten-Mualem (VGM)** drives data generation (ground-truth fidelity).
- **Gardner** drives the controller and the surrogate physics penalty
  (structural-uncertainty signal and NMPC consistency).

The surrogate is a 4-layer MLP (width 512) trained with a physics-residual
weight lambda = 100, the knee of the accuracy/physics trade-off.

## Installation

GPU packages need custom index URLs, so install them first:

    pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
    pip install cupy-cuda13x

Then the remaining packages:

    pip install -r requirements.txt

Tested on: Python 3.14, CUDA 13.2, RTX 5060 Laptop GPU.

## Data

Large artefacts (the dataset and model checkpoints) are not tracked in git.
- `strategy1.h5` (201 runs, ~171 MB) is produced by `generate_dataset.py`.
- `pgml_lam100.pt` is the trained surrogate checkpoint.

## Status

- Strategy 1 (fixed soil, 201 runs): complete.
- Strategy 3 (varied soils, Sobol sampling): pending.

## Citation / reference

Baseline NMPC framework: Olivares et al., *Minerals Engineering*, 2025,
229:109346.

## Author

Sultan Alhamdan, MSc Chemical Engineering, UCL.
