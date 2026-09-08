"""
paths.py
========
Single place where every path in this repository is resolved.

Scripts sit in stage folders (01_Simulator, 04_Training, ...) but import each
other by module name and read shared inputs from reference/. Rather than every
script walking the tree with its own os.path.join chain, they import this and
ask for what they need.

Usage, at the top of any script:

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import paths
    paths.add_stage_paths()

    from column_solver import run_batch
    scal = load_scalers(paths.REFERENCE)
    h5   = paths.H5

WHERE THE DATASET LIVES
The HDF5 file is too large for the repository and is downloaded separately.
This module looks for it in three places, in order:

    1. the PGML_DATA environment variable, if set
    2. a data/ folder inside the repository
    3. the local development layout, three levels up from the repository

Set PGML_DATA if the file is anywhere else:

    PowerShell   $env:PGML_DATA = "D:\\pgml_data"
    bash         export PGML_DATA=/mnt/data/pgml

Origin: original
Status: stable

Changelog
---------
2026.09.07  v1.1  Dataset renamed from strategy3.h5 to dataset.h5. Strategy3
                  was an internal codename and meant nothing to a reader.
2026.09.07  v1.0  Created for the repository rebuild. Replaces the per-script
                  relative path chains, which broke when inputs moved into
                  reference/ and the checkpoints were flattened into
                  04_Training/models/. ASCII only.
"""

import os
import sys

# ---------------------------------------------------------------------------
# Repository layout
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

REFERENCE = os.path.join(REPO_ROOT, "reference")
RESULTS   = os.path.join(REPO_ROOT, "results")
MODELS    = os.path.join(REPO_ROOT, "04_Training", "models")

EXTRAP_SOILS    = os.path.join(REFERENCE, "extrapolation", "soils")
EXTRAP_CEILINGS = os.path.join(REFERENCE, "extrapolation", "ceilings")

STAGES = [
    "01_Simulator",
    "02_Data_Generator",
    "03_Preprocessing",
    "04_Training",
    "05_Evaluation",
    "06_Control",
]

# Kept for scripts that still call load_scalers(prep_dir). The scaler and split
# files now live in reference/, so this is an alias, not a folder of its own.
PREP_DIR = REFERENCE


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

H5_NAME = "dataset.h5"


def _find_data_dir():
    """Return the folder holding the HDF5 dataset, or None if not found."""
    env = os.environ.get("PGML_DATA")
    if env and os.path.isfile(os.path.join(env, H5_NAME)):
        return os.path.abspath(env)

    local = os.path.join(REPO_ROOT, "data")
    if os.path.isfile(os.path.join(local, H5_NAME)):
        return local

    # Development layout: repository sits at
    #   <project>\Python\Leaching Codes (Strategy 3)\_REPO_STAGING
    # and the dataset at
    #   <project>\Data generation\Python dataset
    up3 = os.path.abspath(os.path.join(REPO_ROOT, "..", "..", ".."))
    dev = os.path.join(up3, "Data generation", "Python dataset")
    if os.path.isfile(os.path.join(dev, H5_NAME)):
        return dev

    return None


DATA_DIR = _find_data_dir()
H5 = os.path.join(DATA_DIR, H5_NAME) if DATA_DIR else None


def require_h5():
    """Return the HDF5 path, or exit with an instruction rather than a trace."""
    if H5 and os.path.isfile(H5):
        return H5
    raise SystemExit(
        "\nDataset not found: " + H5_NAME + "\n"
        "Looked in:\n"
        "  1. PGML_DATA environment variable\n"
        "  2. " + os.path.join(REPO_ROOT, "data") + "\n"
        "  3. the local development layout three levels up\n\n"
        "Download the dataset and either place it in a data/ folder inside\n"
        "the repository, or point PGML_DATA at the folder holding it.\n"
    )


# ---------------------------------------------------------------------------
# Imports across stage folders
# ---------------------------------------------------------------------------

def add_stage_paths():
    """Put every stage folder on sys.path so modules import by plain name."""
    for stage in STAGES:
        d = os.path.join(REPO_ROOT, stage)
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def reference(name):
    """Path to a file in reference/."""
    return os.path.join(REFERENCE, name)


def result(name):
    """Path to a file in results/."""
    return os.path.join(RESULTS, name)


def model(name):
    """Path to a checkpoint in 04_Training/models/."""
    return os.path.join(MODELS, name)


if __name__ == "__main__":
    print("REPO_ROOT :", REPO_ROOT)
    print("REFERENCE :", REFERENCE, "exists" if os.path.isdir(REFERENCE) else "MISSING")
    print("RESULTS   :", RESULTS, "exists" if os.path.isdir(RESULTS) else "MISSING")
    print("MODELS    :", MODELS, "exists" if os.path.isdir(MODELS) else "MISSING")
    print("DATA_DIR  :", DATA_DIR if DATA_DIR else "NOT FOUND")
    print("H5        :", H5 if H5 else "NOT FOUND")
    print()
    add_stage_paths()
    print("stage folders on sys.path:")
    for stage in STAGES:
        d = os.path.join(REPO_ROOT, stage)
        print("  ", "ok " if os.path.isdir(d) else "MISSING", stage)
