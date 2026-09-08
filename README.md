![Python](https://img.shields.io/badge/python-3.14-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-nightly%20cu128-orange)
![CUDA](https://img.shields.io/badge/CUDA-12.8-green)

# PGML Copper Heap Leaching

**Physics-Guided Machine Learning for Sustainable Heap Leaching Control** <br>
MSc Chemical Engineering Research Project, UCL. <br>
Author: Sultan Alhamdan <br>
Supervisor: Dr. Paulina Quintanilla

## Overview

Source code for a physics-guided machine learning model that predicts and
regulates bed saturation in copper heap leaching. It replaces the per-step
hydraulic re-fitting a Richards-based nonlinear model predictive controller
needs, moving identification into training so one controller can run at heap
scale. Baseline: Olivares et al., 2025.

## Method

The hydraulic transport layer uses two constitutive relations:

- **van Genuchten Modified (VGM)** drives data generation
- **Gardner** drives the controller and the physics penalty during training

The network is a 4-layer MLP of width 512, trained on a convex combination of a
data term and a Gardner physics residual at λ = 0.4. Input is 226 values:
the 221-node saturation profile, the irrigation rate over the step, and four
soil properties. Output is the 221-node profile one hour ahead.

## Setup

1. Download `dataset.h5` and put it in `data/`. See `data/README.md`.
2. Install dependencies. `torch` is a nightly build and is not on PyPI, so it
   installs separately:

       pip install torch==2.12.0.dev20260408+cu128 --index-url https://download.pytorch.org/whl/nightly/cu128
       pip install -r requirements.txt

`cupy` is optional. Without it the solver runs on the CPU, correctly but slowly.

No configuration. Every script finds `data/` and `reference/` through `paths.py`
when it starts. If the dataset is elsewhere, set `PGML_DATA` to the folder
holding it.

## Run the controller

    cd 06_Control
    python closed_loop.py                    # 614 validation soils
    python closed_loop.py --split test       # 615 test soils

Uses the shipped weights in `04_Training/models/` and the shipped inputs in
`reference/`. Writes to `results/closed_loop/<split>/`.

Add `--limit_ckpts 1 --tf 3` for a short run.

## Retrain

    cd 03_Preprocessing && python preprocess.py
    cd ../04_Training    && python train.py --lam 0.4 --seed 42 --out runs/s42_lam0p4
    cd ../04_Training    && python summarise_training.py --root runs
    cd ../06_Control     && python closed_loop.py --sweep_dir ../04_Training/runs

`preprocess.py` overwrites `reference/scaler.json` and `reference/split.json`
with identical values, since the split seed is fixed at 42.

## Repository structure

    data/                 dataset.h5 goes here, not in the repository
    reference/            inputs the code reads: scalers, splits, soil lists, ceilings
    results/              records from the released runs
    paths.py              resolves every path

    01_Simulator/
        column_solver.py           1D Richards solver, batched over columns
        tridiagonal_solve.py       Thomas algorithm as a CUDA kernel
    02_Data_Generator/
        generate_dataset.py        4096 Sobol designs to dataset.h5
    03_Preprocessing/
        preprocess.py              split, scaling, per-timestep samples
    04_Training/
        train.py                   training loop
        physics_loss.py            Gardner residual
        fit_alpha_g.py             derives the Gardner alpha_g rule
        summarise_training.py      reduces run logs to one summary
        models/                    three checkpoints, λ 0.4, seeds 7/42/123
    05_Evaluation/
        openloop_drift.py          drift when the network runs unanchored
    06_Control/
        closed_loop.py             the controller
        rmax_ceiling.py            per-soil irrigation ceiling
        speed_benchmark.py         times the solver on the CPU
        extrapolation/
            build_extrapolation_soils.py    soils outside the training domain

Each script's docstring lists its flags.

## Data and weights

Three checkpoints ship here, at λ = 0.4 and seeds 7, 42 and 123. Held
separately, being too large for the repository:

- `dataset.h5`, 4096 simulated leach cycles
- the remaining trained checkpoints, covering the λ sweep and earlier
  studies
- per-run training logs, console output and diagnostic probes

Download link: [OneDrive](ADD_LINK_HERE)

## Notes

`--n_soils` draws its own subset and will not match the shipped ceiling, which
covers all 614 or 615 soils. Regenerate one with `rmax_ceiling.py` first, or
leave the flag off.

The controller derives each run's identity from its checkpoint. Sweep runs sit
one folder deep and take the folder name; released weights sit flat in `models/`
and take the file name.

## Reproducibility

Data, models and code are published; dependencies install in one command; run
order, operating system and hardware are recorded; all random components are
seeded. Developed on Windows with an RTX 5060 Laptop, 8.55 GB.
