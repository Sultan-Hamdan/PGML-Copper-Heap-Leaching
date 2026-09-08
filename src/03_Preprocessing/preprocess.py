r"""
preprocess.py
=============
ML preprocessing pipeline for PGML training on heap leaching column runs.
Implements five steps in sequence:

    Step 1 · Load     -- open the dataset h5, read run inventory, check shapes/NaN
    Step 2 · Split    -- 70/15/15 run-level split, fixed seed, no timestep mixing
    Step 3 · Scale    -- fit min-max scalers on train only, save to JSON
    Step 4 · Slice    -- build per-timestep samples (226 inputs, 221 outputs)
    Step 5 · Dataset  -- PyTorch Dataset + DataLoader, all runs cached in RAM

Run from 03_Preprocessing:
    python preprocess.py

Both paths default through paths.py: the dataset is found in data/, and the two
JSON files are written to reference/. Override either if the files are elsewhere:
    python preprocess.py --h5 "D:\other.h5" --out ".\tmp"

Outputs written to the --out directory:
    scaler.json     fitted min-max scaler (train set only)
    split.json      run IDs for each split (reproducibility record)

The three DataLoaders (train, val, test) are returned by build_dataloaders()
and consumed directly by the training scripts in 04_Training/.

Status: stable
Depends on: generate_dataset.py (consumes the h5 it writes)

Changelog
---------
2026.09.07  v2.7  Repository release. --h5 and --out default through paths.py,
                  so the script runs with no arguments once dataset.h5 is in
                  data/. Default batch_size 256 -> 4096, the size the released
                  checkpoints used. Header wording only otherwise.
2026.07.30  v2.6  R indexing corrected. A sample now pairs Se[:,t] with
                  R_out[t+1], the rate driving that step, not R_out[t]. The two
                  agree except at t mod 24 == 23, so 24 of 599 samples per run
                  were unlearnable as written. Max abs error against a one-hour
                  plant step: 5.8e-16 with R_out[t+1], 1.3e-2 with R_out[t].
                  Shapes, scaler and split unchanged, but cached tensors must be
                  rebuilt and any model trained before this is not comparable.
2026.07.28  v2.5  Fourth soil input changed from theta_i to theta_s. theta_i is
                  an initial condition already carried by Se[:,t] and so is
                  redundant for a state-transition operator; theta_s scales the
                  capacity term in D = K/C and varies per run. INPUT_DIM stays
                  226. Key order is [Ks, alpha, n_vgm, theta_s] and 04_Training
                  must match it.
2026.07.18  v2.4  Speed. Input and output tensors are built once at init and
                  held on the GPU, removing 214,442 per-sample rebuilds per
                  epoch and the per-batch host-to-device copy. The card was
                  drawing about 24 W waiting on the CPU. Output identical.
2026.07.17  v2.3  Docstring pass. The file reads whatever h5 it is given, so
                  the header no longer names one dataset. Usage line made a raw
                  string; backslashes were raising a SyntaxWarning.
2026.07.17  v2.2  The Dataset holds one soil vector per run rather than one
                  shared by all runs, and reads the parameters itself through
                  read_soil_param. Removes the last place a missing attribute
                  could be replaced by a hardcoded default. Input width
                  unchanged at 226.
2026.07.17  v2.1  fit_scalers() raises when a parameter is stored per-run but
                  does not vary across training runs. That collapse means the
                  generator failed to wire it through, and the network would
                  otherwise receive a constant zero column. The h5 file declares
                  which parameters vary; there is no hardcoded list.
2026.07.17  v2.0  Soil parameters read per-run where the h5 provides them.
                  Previously f.attrs.get("theta_i", 0.14) returned 0.14 for
                  every run and the scaler collapsed to min == max with no
                  error. read_soil_param() reads the run attrs, falls back to
                  file level, and raises if neither exists. No default value.
2026.06.10  v1.1  All runs loaded into RAM at init instead of per-sample h5
                  reads, removing 83,860 file opens per epoch.
2026.06.09  v1.0  Initial preprocessing pipeline, 201 runs.
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

# One device for the whole pipeline. The dataset is built here at init and the
# training loop runs here too, so batches never leave the GPU once loaded.
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# STEP 1 · LOAD RUNS
# Open the h5 file, inventory run keys, verify shapes, check for NaN.
# Nothing is transformed here — purely a sanity check.
# =============================================================================

def load_inventory(h5_path: str) -> list[str]:
    """
    Read the list of run keys from the dataset h5.
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

    print(f"[Step 2] Split, train: {len(train_ids)}  "
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
# R_out, q_out, Ks, alpha, n, theta_s are min-max scaled to [0, 1].
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

# Soil parameters fed to the network. These are the TRANSITION COEFFICIENTS of
# the Se-space operator: Ks, alpha, n set the VGM shape, theta_s (= theta_r + dtheta,
# theta_r=0) sets the moisture capacity that scales how fast Se moves under R.
# theta_i is deliberately NOT here: it is an initial condition, already carried
# by the input state Se[:,t] (Se_0 = theta_i/theta_s), so feeding it again is
# redundant for a Markovian state-transition operator. theta_s varies per run,
# which is why it must be an input.
# A parameter may be stored per run or at file level. read_soil_param() reads
# the run's own attrs first, falls back to the file-level attr, and raises if
# neither exists, so either layout works without an edit. In the released
# dataset all four are per run.
SOIL_PARAM_KEYS = ["Ks", "alpha", "n_vgm", "theta_s"]


def read_soil_param(f, key: str, run_ids: list[str]) -> tuple[np.ndarray, str]:
    """
    Read one soil parameter for each run in run_ids, and report where it lives.

    Looks in the run's own attrs first, then the file-level attrs. Raises if the
    key is in neither. There is deliberately NO default value: a default that
    silently stands in for a missing value is exactly how theta_i became a
    constant 0.14 across every run with no error raised anywhere.

    The scope is returned because where the generator put the attribute is a
    declaration of intent. Per-run means the value varies. File-level means it
    is fixed for the whole dataset. fit_scalers uses this to tell a legitimate
    single-point range from a wiring failure.

    Returns
    -------
    values : (len(run_ids),) float64 array, one value per run.
    scope  : "run" or "file".
    """
    if key in f[run_ids[0]].attrs:
        vals = np.array([float(f[r].attrs[key]) for r in run_ids], dtype=float)
        return vals, "run"
    if key in f.attrs:
        return np.full(len(run_ids), float(f.attrs[key]), dtype=float), "file"
    raise KeyError(
        f"Soil parameter '{key}' is in neither the run attrs nor the file "
        f"attrs of this h5. Refusing to substitute a default."
    )


def fit_scalers(h5_path: str, train_ids: list[str]) -> dict:
    """
    Compute min and max for each variable over the training runs only.
    Soil parameters are read per-run where the h5 file provides them. A
    parameter the file holds fixed lands as min == max, which apply_scaler
    handles by returning zeros. A parameter the file stores per-run is declared
    to vary; if it does not, this raises rather than let apply_scaler silently
    zero it.

    Fitting on train runs only means val and test theta_i can fall outside
    [min, max] and scale slightly outside [0, 1]. That is not a bug. It is what
    "no leakage" means.

    Returns
    -------
    scalers : dict
        { variable_name: {"min": float, "max": float} }
    """
    R_vals  = []
    q_vals  = []
    _t0 = time.perf_counter()

    with h5py.File(h5_path, "r") as f:
        # --- read soil params per-run over the TRAIN runs only ---
        soil  = {}
        scope = {}
        for k in SOIL_PARAM_KEYS:
            soil[k], scope[k] = read_soil_param(f, k, train_ids)

        # --- accumulate R and q over training runs ---
        for run_id in train_ids:
            R_vals.append(f[run_id]["R_out"][:])
            q_vals.append(f[run_id]["q_out"][:])

    R_all = np.concatenate(R_vals)
    q_all = np.concatenate(q_vals)

    scalers = {
        "R_out":   {"min": float(R_all.min()),  "max": float(R_all.max())},
        "q_out":   {"min": float(q_all.min()),  "max": float(q_all.max())},
    }
    # A parameter that is fixed lands as min == max. One that varies gets a
    # real range. The file decides, not the code.
    for k in SOIL_PARAM_KEYS:
        lo, hi = float(soil[k].min()), float(soil[k].max())
        # A parameter stored per-run is declared to vary. If it does not, the
        # generator failed to wire it through and apply_scaler would return a
        # constant zero column for it with no error anywhere downstream.
        if scope[k] == "run" and hi == lo:
            raise ValueError(
                f"'{k}' is stored as a per-run attribute, which means the "
                f"generator intended it to vary, but all {len(train_ids)} "
                f"training runs share the value {lo}. Either the generator did "
                f"not pass it to the simulator, or it belongs at file level. "
                f"Refusing to scale it to a constant zero column."
            )
        scalers[k] = {"min": lo, "max": hi}

    print(f"[Step 3] Scalers fitted on train set  ({time.perf_counter()-_t0:.2f}s):")
    for var, bounds in scalers.items():
        span = bounds["max"] - bounds["min"]
        tag  = "" if var not in SOIL_PARAM_KEYS else (
            "  (fixed)" if span == 0.0 else f"  (varies over {len(train_ids)} train runs)")
        print(f"         {var:10s}  min={bounds['min']:.6f}  "
              f"max={bounds['max']:.6f}{tag}")
    return scalers


def apply_scaler(x: np.ndarray, var_name: str, scalers: dict) -> np.ndarray:
    """
    Apply min-max scaling to a single variable using the pre-fitted scaler.
    x_scaled = (x - x_min) / (x_max - x_min)

    If min == max the variable is constant across the training set and this
    returns zeros. That is correct for a parameter the dataset genuinely holds
    fixed, and it is silent, so it would also hide a parameter that was supposed
    to vary but never got wired through. fit_scalers is responsible for telling
    those two cases apart before any scaler reaches this function.
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
# One sample: [Se[:,t], R[t+1], Ks, alpha, n, theta_s] -> Se[:,t+1]
#             R[t+1] is the rate applied over the step t -> t+1 (see v2.6).
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
    print(f"         input  : Se[:,t] (221) + R[t+1] (1) + soil (4) = {INPUT_DIM}")
    print(f"         output : Se[:,t+1] = {OUTPUT_DIM}")
    print(f"         train  : {len(train_ids):>3} runs x {SAMPLES_PER_RUN} = {n_train:>6} samples")
    print(f"         val    : {len(val_ids):>3} runs x {SAMPLES_PER_RUN} = {n_val:>6} samples")
    print(f"         test   : {len(test_ids):>3} runs x {SAMPLES_PER_RUN} = {n_test:>6} samples")


# =============================================================================
# STEP 5 · PYTORCH DATASET + DATALOADER
# HeapDataset loads ALL runs into RAM at init (one h5 read per run).
# __getitem__ slices from in-memory arrays — no disk access during training.
# X and Y are moved to _DEVICE (GPU). For 4096 runs that is
# ~2.45M samples: X (N,226) + Y (N,221) ~= 4.4 GB resident on the 8.55 GB card,
# split across train/val/test. Tight but fits with the small MLP; watch it if
# runs grow (drop to CPU-resident or chunk if it OOMs).
# =============================================================================

class HeapDataset(Dataset):
    """
    PyTorch Dataset for the PGML heap leaching prediction model.

    One sample = one hourly state transition from a single simulation run:
        input  : Se[:,t] (221) + R[t+1] (1) + [Ks, alpha, n, theta_s] (4) = 226
        output : Se[:,t+1] (221)

    At init every run is read from the h5 ONCE, the full (N, 226) input matrix
    and (N, 221) output matrix are assembled in one vectorised pass, and both are
    moved to the GPU. There is no per-sample work during training: __getitem__
    and the batch loader just index the resident tensors. This replaces the old
    per-sample np.concatenate (214,442 rebuilds per epoch) that left the GPU
    starving while the CPU assembled batches.

        self.X : (N, 226) float32 on _DEVICE
        self.Y : (N, 221) float32 on _DEVICE
    """

    def __init__(
        self,
        h5_path:  str,
        run_ids:  list[str],
        scalers:  dict,
    ):
        self.run_ids = run_ids
        self.scalers = scalers

        _t0 = time.perf_counter()

        with h5py.File(h5_path, "r") as f:
            # Soil parameters are read here rather than passed in, because this
            # is already the only place that opens the file. One reader means
            # one rule: per-run attr if the file has one, else the file-level
            # attr, else raise. A parameter the file holds fixed gives every run
            # the same value; one stored per-run gives each run its own.
            soil_raw = {k: read_soil_param(f, k, run_ids)[0]
                        for k in SOIL_PARAM_KEYS}
            # (n_runs, 4) scaled soil, row i belonging to run i.
            soil_cache = np.stack(
                [apply_scaler(soil_raw[k], k, scalers) for k in SOIL_PARAM_KEYS],
                axis=1,
            ).astype(np.float32)

            # Assemble each run's block of samples, then stack. All the gluing
            # that used to happen per sample happens once, here, vectorised.
            X_blocks = []
            Y_blocks = []
            for i, run_key in enumerate(run_ids):
                Se = f[run_key]["Se_out"][:].astype(np.float32)        # (221, 600)
                R  = apply_scaler(f[run_key]["R_out"][:], "R_out", scalers)  # (600,)

                Se_x = Se[:, :SAMPLES_PER_RUN].T                       # (599, 221) state at t
                Se_y = Se[:, 1:SAMPLES_PER_RUN + 1].T                  # (599, 221) state at t+1
                # R_out[k] is the rate applied DURING hour k, and Se_out[k] is
                # the state at the END of hour k. The step t -> t+1 is hour
                # t+1, so it is driven by R_out[t+1], not R_out[t]. The two
                # differ only at day boundaries (t mod 24 == 23) because R is
                # constant within a day. See changelog v2.6.
                R_x  = R[1:SAMPLES_PER_RUN + 1].reshape(-1, 1)         # (599, 1) rate over t -> t+1
                soil_x = np.broadcast_to(soil_cache[i], (SAMPLES_PER_RUN, 4))  # (599, 4)

                X_blocks.append(np.concatenate([Se_x, R_x, soil_x], axis=1))  # (599, 226)
                Y_blocks.append(Se_y)                                          # (599, 221)

        X = np.concatenate(X_blocks, axis=0).astype(np.float32)   # (N, 226)
        Y = np.concatenate(Y_blocks, axis=0).astype(np.float32)   # (N, 221)

        self.X = torch.from_numpy(X).to(_DEVICE)
        self.Y = torch.from_numpy(Y).to(_DEVICE)

        vram = (self.X.element_size() * self.X.nelement()
                + self.Y.element_size() * self.Y.nelement()) / 1e6
        print(f"[Step 5] Dataset built and moved to {_DEVICE}: "
              f"{len(run_ids)} runs, {self.X.shape[0]} samples, "
              f"{vram:.0f} MB  ({time.perf_counter()-_t0:.2f}s)")

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        # Single-sample access (e.g. inspection). Training uses GPUBatchLoader,
        # which slices whole batches instead of calling this per sample.
        return self.X[idx], self.Y[idx]


class GPUBatchLoader:
    """
    Minimal loader over a GPU-resident HeapDataset. Yields (x, y) batches that
    are already on the GPU, so the training loop's .to(device) is a no-op and
    nothing is copied per batch.

    Duck-types the parts of a DataLoader that the training and evaluation code
    use: iteration, len (number of batches), and the .dataset / .batch_size
    attributes. shuffle reshuffles the row order each epoch (train only); the
    data itself never moves.
    """

    def __init__(self, dataset: HeapDataset, batch_size: int, shuffle: bool):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.N = len(dataset)

    def __len__(self) -> int:
        return (self.N + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        X, Y = self.dataset.X, self.dataset.Y
        if self.shuffle:
            order = torch.randperm(self.N, device=X.device)
        else:
            order = torch.arange(self.N, device=X.device)
        for i in range(0, self.N, self.batch_size):
            idx = order[i:i + self.batch_size]
            yield X[idx], Y[idx]


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
    h5_path    : path to the dataset h5
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

    # -- Step 4: slice summary --
    slice_summary(train_ids, val_ids, test_ids)

    # -- Step 5: Dataset and loader for each split --
    # Data is built once and held on the GPU (see HeapDataset). The loaders
    # below only slice it, so no batch is assembled on the CPU or copied per
    # step. num_workers is accepted for backward compatibility but unused: there
    # is no CPU-side work left to parallelise.
    _t5 = time.perf_counter()
    train_ds = HeapDataset(h5_path, train_ids, scalers)
    val_ds   = HeapDataset(h5_path, val_ids,   scalers)
    test_ds  = HeapDataset(h5_path, test_ids,  scalers)

    train_loader = GPUBatchLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = GPUBatchLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = GPUBatchLoader(test_ds,  batch_size=batch_size, shuffle=False)

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
    # paths is imported here rather than at module scope so that importing
    # preprocess as a library (train.py, closed_loop.py) does not touch sys.path.
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import paths

    p = argparse.ArgumentParser(description="PGML preprocessing pipeline")
    p.add_argument("--h5", default=None,
                   help="Path to the dataset h5. Default: data/dataset.h5")
    p.add_argument("--out", default=None,
                   help="Output directory for scaler.json and split.json. "
                        "Default: reference/")
    p.add_argument("--batch_size", type=int, default=4096,
                   help="Default 4096, the size the released checkpoints used")
    a = p.parse_args()

    if a.h5 is None:
        a.h5 = paths.require_h5()
    if a.out is None:
        a.out = paths.REFERENCE
        os.makedirs(a.out, exist_ok=True)

    return a


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
