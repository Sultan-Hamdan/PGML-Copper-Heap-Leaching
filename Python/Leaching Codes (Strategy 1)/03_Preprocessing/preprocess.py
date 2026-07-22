"""
preprocess.py
=============
ML preprocessing pipeline for heap leaching surrogate training.
Implements five steps in sequence:

    Step 1 · Load     -- open strategy1.h5, read run inventory, check shapes/NaN
    Step 2 · Split    -- 70/15/15 run-level split, fixed seed, no timestep mixing
    Step 3 · Scale    -- fit min-max scalers on train only, save to JSON
    Step 4 · Slice    -- build per-timestep samples (226 inputs, 221 outputs)
    Step 5 · Dataset  -- PyTorch Dataset + DataLoader, all runs cached in RAM

Run from the Windows terminal:
    python preprocess.py --h5 "path\to\strategy1.h5" --out "path\to\03_Preprocessing"

Outputs written to --out directory:
    scaler.json     fitted min-max scaler (train set only)
    split.json      run IDs for each split (reproducibility record)

The three DataLoaders (train, val, test) are returned by build_dataloaders()
and consumed directly by the training scripts in 04_Training/.

Origin: original (not a MATLAB port)
Status: active development
Depends on: generate_dataset.py (consumes strategy1.h5)

Changelog
---------
2026.06.10  v1.1  HeapDataset: load all runs into RAM at init instead of per-sample h5 reads.
                  Eliminates bottleneck of 83,860 file opens per epoch.
                  RAM cost ~171 MB (Strategy 1).
2026.06.09  v1.0  Initial preprocessing pipeline, Strategy 1, 201 runs
"""

import sys
import os
import json
import time
import argparse
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# STEP 1 · LOAD RUNS
# Open the h5 file, inventory run keys, verify shapes, check for NaN.
# Nothing is transformed here — purely a sanity check.
# =============================================================================

def load_inventory(h5_path: str) -> list[str]:
    """
    Read the list of run keys from strategy1.h5.
    Verifies that every run has the expected arrays and shapes.
    Drops any run with NaN values and warns.

    Returns
    -------
    run_ids : list[str]
        Sorted list of clean run keys, e.g. ['run_0000', 'run_0001', ...]
    """
    EXPECTED = {
        "Se_out":     (221, 600),
        "R_out":      (600,),
        "q_out":      (600,),
        "Q_schedule": None,        # shape varies (25 or 44) — not checked strictly
    }

    clean = []
    dropped = []
    _t0 = time.perf_counter()

    with h5py.File(h5_path, "r") as f:
        all_keys = sorted(k for k in f.keys() if k.startswith("run_"))
        print(f"[Step 1] Found {len(all_keys)} run keys in {os.path.basename(h5_path)}")

        for key in all_keys:
            grp = f[key]

            # --- shape check ---
            ok = True
            for arr_name, expected_shape in EXPECTED.items():
                if arr_name not in grp:
                    print(f"  WARNING {key}: missing '{arr_name}' — dropping")
                    ok = False
                    break
                if expected_shape is not None:
                    actual = grp[arr_name].shape
                    if actual != expected_shape:
                        print(f"  WARNING {key}: '{arr_name}' shape {actual} "
                              f"expected {expected_shape} — dropping")
                        ok = False
                        break
            if not ok:
                dropped.append(key)
                continue

            # --- NaN check ---
            for arr_name in ("Se_out", "R_out", "q_out"):
                data = grp[arr_name][:]
                if np.any(np.isnan(data)):
                    print(f"  WARNING {key}: NaN in '{arr_name}' — dropping")
                    ok = False
                    break
            if not ok:
                dropped.append(key)
                continue

            clean.append(key)

    if dropped:
        print(f"[Step 1] Dropped {len(dropped)} runs: {dropped}")
    print(f"[Step 1] Clean inventory: {len(clean)} runs ready  ({time.perf_counter()-_t0:.2f}s)")
    return clean


# =============================================================================
# STEP 2 · RUN-LEVEL SPLIT
# Shuffle run IDs with a fixed seed, then cut at 70 / 15 / 15.
# Whole runs go to one partition — never split across sets.
# Splitting before scaling prevents data leakage.
# =============================================================================

SPLIT_SEED = 42          # fixed — must never change after first run
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# TEST_FRAC  = 0.15      # implicit: whatever remains


def split_runs(run_ids: list[str]) -> tuple[list, list, list]:
    """
    Randomly assign whole runs to train / val / test at 70/15/15.
    Uses SPLIT_SEED — reproducible across machines and time.

    Returns
    -------
    train_ids, val_ids, test_ids : list[str]
    """
    _t0 = time.perf_counter()
    rng = np.random.default_rng(SPLIT_SEED)
    ids = np.array(run_ids, dtype=object)
    rng.shuffle(ids)

    n      = len(ids)
    n_train = int(np.floor(n * TRAIN_FRAC))
    n_val   = int(np.floor(n * VAL_FRAC))

    train_ids = ids[:n_train].tolist()
    val_ids   = ids[n_train : n_train + n_val].tolist()
    test_ids  = ids[n_train + n_val :].tolist()

    print(f"[Step 2] Split — train: {len(train_ids)}  "
          f"val: {len(val_ids)}  test: {len(test_ids)}  "
          f"(seed={SPLIT_SEED})  ({time.perf_counter()-_t0:.2f}s)")
    return train_ids, val_ids, test_ids


def save_split(train_ids, val_ids, test_ids, out_dir: str):
    """Save split record to JSON for reproducibility."""
    record = {
        "seed":      SPLIT_SEED,
        "train":     train_ids,
        "val":       val_ids,
        "test":      test_ids,
        "n_train":   len(train_ids),
        "n_val":     len(val_ids),
        "n_test":    len(test_ids),
    }
    path = os.path.join(out_dir, "split.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"[Step 2] Split record saved -> {path}")


# =============================================================================
# STEP 3 · FIT SCALERS ON TRAIN SET ONLY
# Se is already in [0, 1] — no scaling needed.
# R_out, q_out, Ks, alpha, n, theta_i are min-max scaled to [0, 1].
# Scaler is fit on the train set ONLY, then applied to val and test.
# Fitting on all runs would leak test statistics into normalisation.
# =============================================================================

# Variables that need scaling and where to find them in the h5 file.
# Se_out is excluded — already [0, 1] by definition.
# Q_schedule is excluded — not a model input (role ends at data generation).
SCALAR_VARS = {
    "R_out":   "per_timestep",   # shape (600,) — one value per hour
    "q_out":   "per_timestep",   # shape (600,) — one value per hour
}

# Soil parameters are stored as file-level attributes in strategy1.h5
# (same for all runs in Strategy 1; will vary per run in Strategy 3).
SOIL_PARAM_KEYS = ["Ks", "alpha", "n_vgm", "theta_i"]


def fit_scalers(h5_path: str, train_ids: list[str]) -> dict:
    """
    Compute min and max for each variable over the training runs only.
    Soil parameters for Strategy 1 are fixed (file-level attrs) —
    the scaler stores them as single-point ranges for code consistency.

    Returns
    -------
    scalers : dict
        { variable_name: {"min": float, "max": float} }
    """
    R_vals  = []
    q_vals  = []
    _t0 = time.perf_counter()

    with h5py.File(h5_path, "r") as f:
        # --- read soil params from file-level attrs (Strategy 1: fixed) ---
        Ks      = float(f.attrs.get("Ks",      170.0))
        alpha   = float(f.attrs.get("alpha",   0.035))
        n_vgm   = float(f.attrs.get("n_vgm",   2.267))
        theta_i = float(f.attrs.get("theta_i", 0.14))

        # --- accumulate R and q over training runs ---
        for run_id in train_ids:
            R_vals.append(f[run_id]["R_out"][:])
            q_vals.append(f[run_id]["q_out"][:])

    R_all = np.concatenate(R_vals)
    q_all = np.concatenate(q_vals)

    scalers = {
        "R_out":   {"min": float(R_all.min()),  "max": float(R_all.max())},
        "q_out":   {"min": float(q_all.min()),  "max": float(q_all.max())},
        # Soil params: Strategy 1 is fixed, so min == max == the single value.
        # Strategy 3 will have ranges here once the dataset is varied.
        "Ks":      {"min": Ks,      "max": Ks},
        "alpha":   {"min": alpha,   "max": alpha},
        "n_vgm":   {"min": n_vgm,   "max": n_vgm},
        "theta_i": {"min": theta_i, "max": theta_i},
    }

    print(f"[Step 3] Scalers fitted on train set  ({time.perf_counter()-_t0:.2f}s):")
    for var, bounds in scalers.items():
        print(f"         {var:10s}  min={bounds['min']:.6f}  max={bounds['max']:.6f}")
    return scalers


def apply_scaler(x: np.ndarray, var_name: str, scalers: dict) -> np.ndarray:
    """
    Apply min-max scaling to a single variable using the pre-fitted scaler.
    x_scaled = (x - x_min) / (x_max - x_min)
    If min == max (Strategy 1 soil params), returns array of zeros (safe).
    """
    lo = scalers[var_name]["min"]
    hi = scalers[var_name]["max"]
    if hi == lo:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def save_scalers(scalers: dict, out_dir: str):
    """Save fitted scalers to JSON so they can be reloaded at inference time."""
    path = os.path.join(out_dir, "scaler.json")
    with open(path, "w") as f:
        json.dump(scalers, f, indent=2)
    print(f"[Step 3] Scalers saved -> {path}")


def load_scalers(out_dir: str) -> dict:
    """Reload scalers from JSON (used by inference scripts in 04_Training/)."""
    path = os.path.join(out_dir, "scaler.json")
    with open(path, "r") as f:
        return json.load(f)


# =============================================================================
# STEP 4 · SLICE INTO SAMPLES
# Each run of 600 hourly states becomes 599 per-timestep transitions.
# One sample: [Se[:,t], R[t], Ks, alpha, n, theta_i] -> Se[:,t+1]
# Input: 226 values  (221 Se + 1 R + 4 soil params)
# Output: 221 values (next-hour Se profile)
# Slicing is materialised in HeapDataset.__init__ (RAM cache).
# =============================================================================

SAMPLES_PER_RUN = 599    # 600 hours -> 599 transitions (no t+1 for last step)
INPUT_DIM  = 226         # 221 Se + 1 R + 4 soil params
OUTPUT_DIM = 221         # full Se profile at t+1


def slice_summary(train_ids: list, val_ids: list, test_ids: list):
    """Print a summary of what slicing produces."""
    _t0 = time.perf_counter()
    n_train = len(train_ids) * SAMPLES_PER_RUN
    n_val   = len(val_ids)   * SAMPLES_PER_RUN
    n_test  = len(test_ids)  * SAMPLES_PER_RUN
    print(f"[Step 4] Slicing runs into per-timestep samples  ({time.perf_counter()-_t0:.2f}s):")
    print(f"         {SAMPLES_PER_RUN} samples per run  "
          f"(600 hours -> {SAMPLES_PER_RUN} transitions)")
    print(f"         input  : Se[:,t] (221) + R[t] (1) + soil (4) = {INPUT_DIM}")
    print(f"         output : Se[:,t+1] = {OUTPUT_DIM}")
    print(f"         train  : {len(train_ids):>3} runs x {SAMPLES_PER_RUN} = {n_train:>6} samples")
    print(f"         val    : {len(val_ids):>3} runs x {SAMPLES_PER_RUN} = {n_val:>6} samples")
    print(f"         test   : {len(test_ids):>3} runs x {SAMPLES_PER_RUN} = {n_test:>6} samples")


# =============================================================================
# STEP 5 · PYTORCH DATASET + DATALOADER
# HeapDataset loads ALL runs into RAM at init (one h5 read per run).
# __getitem__ slices from in-memory arrays — no disk access during training.
# RAM cost: ~171 MB for Strategy 1 (201 runs). ~1.7 GB for Strategy 3 (2048
# runs) — still fits comfortably in 16 GB.
# =============================================================================

class HeapDataset(Dataset):
    """
    PyTorch Dataset for the PGML-MPC heap leaching surrogate.

    One sample = one hourly state transition from a single simulation run:
        input  : Se[:,t] (221) + R[t] (1) + [Ks, alpha, n, theta_i] (4) = 226
        output : Se[:,t+1] (221)

    All runs are loaded into RAM at init. __getitem__ slices from memory —
    no h5 file access during training. Eliminates per-sample file open
    bottleneck (83,860 opens per epoch in the previous version).
    """

    def __init__(
        self,
        h5_path:  str,
        run_ids:  list[str],
        scalers:  dict,
        h5_attrs: dict,        # soil params: {"Ks": float, "alpha": float, ...}
    ):
        self.run_ids  = run_ids
        self.scalers  = scalers

        # --- pre-compute fixed soil param scalars (same for all runs, Strategy 1) ---
        self.soil_vec = np.array([
            apply_scaler(np.array([h5_attrs["Ks"]]),      "Ks",      scalers)[0],
            apply_scaler(np.array([h5_attrs["alpha"]]),   "alpha",   scalers)[0],
            apply_scaler(np.array([h5_attrs["n_vgm"]]),   "n_vgm",   scalers)[0],
            apply_scaler(np.array([h5_attrs["theta_i"]]), "theta_i", scalers)[0],
        ], dtype=np.float32)   # shape (4,)

        # --- load all runs into RAM ---
        # Se_cache[i] : (221, 600) float32
        # R_cache[i]  : (600,)     float32  already scaled
        _t0 = time.perf_counter()
        self.Se_cache = []
        self.R_cache  = []

        with h5py.File(h5_path, "r") as f:
            for run_key in run_ids:
                Se = f[run_key]["Se_out"][:].astype(np.float32)   # (221, 600)
                R  = f[run_key]["R_out"][:]                        # (600,)
                R_scaled = apply_scaler(R, "R_out", scalers)       # (600,) float32
                self.Se_cache.append(Se)
                self.R_cache.append(R_scaled)

        print(f"[Step 5] RAM cache loaded: {len(run_ids)} runs  "
              f"({time.perf_counter()-_t0:.2f}s)")

    def __len__(self) -> int:
        return len(self.run_ids) * SAMPLES_PER_RUN

    def __getitem__(self, idx: int):
        run_idx = idx // SAMPLES_PER_RUN
        t       = idx  % SAMPLES_PER_RUN

        Se = self.Se_cache[run_idx]   # (221, 600) — already in RAM
        R  = self.R_cache[run_idx]    # (600,)     — already scaled

        # --- build input vector (226,) ---
        x = np.concatenate([
            Se[:, t],          # 221 Se values at time t
            [R[t]],            # 1 irrigation rate at time t
            self.soil_vec,     # 4 soil parameters
        ])

        # --- target output (221,) ---
        y = Se[:, t + 1]       # Se profile at time t+1

        return torch.from_numpy(x), torch.from_numpy(y)


def build_dataloaders(
    h5_path:   str,
    out_dir:   str,
    batch_size: int = 256,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Full preprocessing pipeline: runs Steps 1-5 in sequence and returns
    three DataLoaders ready for the training loop.

    Parameters
    ----------
    h5_path    : path to strategy1.h5
    out_dir    : directory to write scaler.json and split.json
    batch_size : mini-batch size fed to the GPU (default 256)
    num_workers: parallel data workers (0 = main process, safe on Windows)

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    """
    os.makedirs(out_dir, exist_ok=True)
    _t_total = time.perf_counter()

    # -- Step 1: inventory --
    run_ids = load_inventory(h5_path)

    # -- Step 2: split --
    train_ids, val_ids, test_ids = split_runs(run_ids)
    save_split(train_ids, val_ids, test_ids, out_dir)

    # -- Step 3: fit scalers on train only, save --
    scalers = fit_scalers(h5_path, train_ids)
    save_scalers(scalers, out_dir)

    # -- Read file-level soil attrs (same for all Strategy 1 runs) --
    with h5py.File(h5_path, "r") as f:
        h5_attrs = {
            "Ks":      float(f.attrs.get("Ks",      170.0)),
            "alpha":   float(f.attrs.get("alpha",   0.035)),
            "n_vgm":   float(f.attrs.get("n_vgm",   2.267)),
            "theta_i": float(f.attrs.get("theta_i", 0.14)),
        }
    print(f"[Step 3] Soil params (Strategy 1 fixed): {h5_attrs}")

    # -- Step 4: slice summary --
    slice_summary(train_ids, val_ids, test_ids)

    # -- Step 5: Dataset and DataLoader for each split --
    _t5 = time.perf_counter()
    train_ds = HeapDataset(h5_path, train_ids, scalers, h5_attrs)
    val_ds   = HeapDataset(h5_path, val_ids,   scalers, h5_attrs)
    test_ds  = HeapDataset(h5_path, test_ids,  scalers, h5_attrs)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"\n[Step 5] DataLoaders ready  ({time.perf_counter()-_t5:.2f}s):")
    print(f"         train  {len(train_ds):>6} samples  "
          f"-> {len(train_loader):>4} batches (batch_size={batch_size})")
    print(f"         val    {len(val_ds):>6} samples  "
          f"-> {len(val_loader):>4} batches")
    print(f"         test   {len(test_ds):>6} samples  "
          f"-> {len(test_loader):>4} batches")

    print(f"\nTotal preprocessing time: {time.perf_counter()-_t_total:.2f}s")

    return train_loader, val_loader, test_loader


# =============================================================================
# ENTRY POINT
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="PGML preprocessing pipeline")
    p.add_argument("--h5",  required=True, help="Path to strategy1.h5")
    p.add_argument("--out", required=True, help="Output directory for scaler.json and split.json")
    p.add_argument("--batch_size", type=int, default=256)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("PGML Preprocessing Pipeline")
    print("=" * 60)

    train_loader, val_loader, test_loader = build_dataloaders(
        h5_path    = args.h5,
        out_dir    = args.out,
        batch_size = args.batch_size,
    )

    # --- quick smoke test: fetch one batch and print shapes ---
    print("\n[Smoke test] Fetching one batch from train_loader...")
    _t_smoke = time.perf_counter()
    x_batch, y_batch = next(iter(train_loader))
    print(f"  x_batch shape : {tuple(x_batch.shape)}   "
          f"expected ({args.batch_size}, {INPUT_DIM})")
    print(f"  y_batch shape : {tuple(y_batch.shape)}   "
          f"expected ({args.batch_size}, {OUTPUT_DIM})")
    print(f"  x dtype       : {x_batch.dtype}")
    print(f"  x min/max     : {x_batch.min():.4f} / {x_batch.max():.4f}  "
          f"(should be near [0, 1])")
    print(f"  y min/max     : {y_batch.min():.4f} / {y_batch.max():.4f}  "
          f"(should be in [0, 1])")
    print(f"  batch fetch   : {time.perf_counter()-_t_smoke:.2f}s")
    print("\nPreprocessing complete.")
