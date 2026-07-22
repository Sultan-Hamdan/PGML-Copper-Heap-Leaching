r"""
preprocess.py
=============
ML preprocessing pipeline for heap leaching surrogate training.
Implements five steps in sequence:

    Step 1 · Load     -- open the dataset h5, read run inventory, check shapes/NaN
    Step 2 · Split    -- 70/15/15 run-level split, fixed seed, no timestep mixing
    Step 3 · Scale    -- fit min-max scalers on train only, save to JSON
    Step 4 · Slice    -- build per-timestep samples (226 inputs, 221 outputs)
    Step 5 · Dataset  -- PyTorch Dataset + DataLoader, all runs cached in RAM

Run from the Windows terminal:
    python preprocess.py --h5 r"path\to\strategy<S>.h5" --out r"path\to\03_Preprocessing"

Outputs written to --out directory:
    scaler.json     fitted min-max scaler (train set only)
    split.json      run IDs for each split (reproducibility record)

The three DataLoaders (train, val, test) are returned by build_dataloaders()
and consumed directly by the training scripts in 04_Training/.

Origin: original (not a MATLAB port)
Status: active development
Depends on: generate_dataset.py (consumes the h5 it writes)

Changelog
---------
2026.07.18  v2.4  Speed: the dataset now builds the full (N, 226) input and
                  (N, 221) output tensors ONCE at init and holds them on the GPU.
                  Training reads each batch as a GPU slice, so the per-sample
                  concatenate (214,442 rebuilds per epoch) and the per-batch
                  host-to-device copy are both gone. The GPU was starving
                  (about 24 W) waiting on the CPU to assemble batches; it is now
                  fed directly from VRAM. Output is unchanged: same samples, same
                  70/15/15 split, same scaling, same input width 226. Only the
                  plumbing moved. num_workers is now unused (kept in the
                  signature for callers that still pass it).
2026.07.17  v2.3  Docstring pass. The file is strategy-agnostic: it reads
                  whatever h5 it is pointed at and lets the file declare which
                  parameters vary, so the header no longer names strategy1.h5
                  as though it were the only input. The usage line is now a raw
                  string; the Windows backslashes in it were being read as
                  escape sequences and raised a SyntaxWarning on every run.
2026.07.17  v2.2  HeapDataset now holds one soil vector per run instead of one
                  shared by every run, so a per-run parameter such as theta_i
                  reaches the network as the run's own value. The h5_attrs
                  argument is gone: the Dataset already opens the h5 file to
                  cache Se and R, so it reads the soil parameters itself through
                  read_soil_param and applies the same per-run-then-file-level
                  rule as the scaler fit. This removes the last place where a
                  missing attribute could be replaced by a hardcoded default.
                  Input width is unchanged at 226 (221 Se + 1 R + 4 soil): the
                  parameters take per-run values, they do not become more
                  numerous.
2026.07.17  v2.1  fit_scalers() now raises when a soil parameter is stored
                  per-run but does not vary across the training runs. Where the
                  h5 file puts an attribute is a declaration of intent: a
                  generator writes a value per-run because it varies and
                  file-level because it does not. A per-run parameter that
                  collapses to min == max means the generator failed to wire it
                  through, and apply_scaler would silently feed the network a
                  constant zero column for it. No strategy flag and no hardcoded
                  list of which parameters vary: the file declares it, the check
                  reads it. Strategy 3 moves Ks, alpha and n to per-run and the
                  rule follows with no edit.
2026.07.17  v2.0  Soil parameters are now read per-run where the h5 file
                  provides them. Strategy 2 stores theta_i as a per-run attr
                  rather than a file-level one, so the previous
                  f.attrs.get("theta_i", 0.14) silently returned 0.14 for every
                  run and the scaler collapsed to min == max with no error.
                  read_soil_param() reads the run's own attrs, falls back to the
                  file-level attr, and raises if neither exists. There is no
                  default value: a plausible fallback standing in for a missing
                  value is what caused the failure. Works unchanged for
                  Strategy 3, where all four parameters become per-run.
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

# Soil parameters. Where they live depends on the strategy:
#   Strategy 1: all four file-level (fixed).
#   Strategy 2: theta_i per-run, the rest file-level.
#   Strategy 3: all four per-run.
# read_soil_param() handles all three without an edit.
SOIL_PARAM_KEYS = ["Ks", "alpha", "n_vgm", "theta_i"]


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
                R_x  = R[:SAMPLES_PER_RUN].reshape(-1, 1)              # (599, 1)
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
    p = argparse.ArgumentParser(description="PGML preprocessing pipeline")
    p.add_argument("--h5",  required=True, help="Path to the dataset h5")
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
