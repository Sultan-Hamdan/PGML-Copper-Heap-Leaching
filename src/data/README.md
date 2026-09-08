# data

This folder is empty in the repository. Put `dataset.h5` here.

    data/dataset.h5

Download link: see the main README.

## What the file contains

4096 simulated leach cycles. Each run is one 5.5 m column, discretised into 221
depth nodes, integrated for 600 hours at hourly steps under its own irrigation
schedule. Each run carries the four soil properties it was generated with:
saturated hydraulic conductivity, the van Genuchten alpha and n, and saturated
water content, together with its initial effective saturation.

The file was produced by `02_Data_Generator/generate_dataset.py`, which calls the
solver in `01_Simulator/`. The generation log is beside that script.

## How it is used

`03_Preprocessing/preprocess.py` splits the 4096 runs into 2867 for training,
614 for validation and 615 for testing, under a fixed seed of 42. The split is
recorded in `reference/split.json`, so the same runs land in the same split on
any machine.

The control scripts in `06_Control/` also read this file directly, for the soil
properties of the runs they simulate.

## If you keep it somewhere else

Set `PGML_DATA` to the folder holding the file.

    PowerShell   $env:PGML_DATA = "D:\pgml_data"
    bash         export PGML_DATA=/mnt/data/pgml

`paths.py` checks that variable first, then this folder, and reports where it
looked if the file is not found.
