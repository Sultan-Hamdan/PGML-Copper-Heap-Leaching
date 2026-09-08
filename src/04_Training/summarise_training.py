"""
summarise_training.py
=====================
Reduces a directory of training runs to one summary file.

train.py writes a log.json per run, each holding one row per epoch. Reading
thirty-odd of those separately is impractical and most of the content is not
needed to compare runs. This walks the run directory, pulls the configuration
and the per-epoch series that matter, and writes a single JSON.

Kept per run: every configuration field, the best epoch and its monitor value,
and eight series over the epochs. Dropped: grad_p99 and grad_mean, which the
median and the maximum already bracket, and the per-epoch timings, which the
median summarises. Console output and checkpoints are untouched.

Usage, from 04_Training:
  python summarise_training.py
  python summarise_training.py --root runs/sweep --out ../results/summary.json

--root defaults to runs/sweep, which train.py creates. --out defaults to
results/training_runs_summary.json in the repository root. The released summary
covers 34 runs; re-running this after a retrain overwrites it.

Status: stable
Depends on: log.json files written by train.py

Changelog
---------
2026.09.07  v1.2  Repository release. Renamed from collect_sweep.py. --out now
                  defaults through paths.py. Header wording only otherwise.
2026.08.12  v1.1  Added tr_data, tr_phys, grad_max and frac_out to the kept
                  series. The first two carry the free lam 0 control; grad_max
                  cannot be inferred from the median.
2026.08.12  v1.0  Created. Reduces a directory of run logs to one summary.
"""

import argparse
import json
from pathlib import Path


SERIES = [
    ("monitor", lambda r: r["monitor"]),
    ("val_phys", lambda r: r["val"]["phys"]),
    ("tr_data", lambda r: r["train"]["data"]),
    # Passive at lam 0, where the physics term is skipped entirely. That makes
    # it the free control for whether the penalty earns its place: the physics
    # score of a model that never saw the residual.
    ("tr_phys", lambda r: r["train"]["phys"]),
    ("grad_median", lambda r: r["train"]["grad_median"]),
    # The median alone cannot show whether a gradient tail developed, and a
    # tail is the failure mode that would require a gradient cap to return.
    ("grad_max", lambda r: r["train"]["grad_max"]),
    ("frac_out", lambda r: (r["val"]["frac_below_zero"]
                            + r["val"]["frac_above_one"])),
    ("lr", lambda r: r["lr"]),
]

CONFIG = ["script", "version", "lam", "seed", "output_activation",
          "normalisation", "form", "effective_physics_weight", "train_frac",
          "subset", "steps_per_epoch", "batch_size", "lr", "warmup_epochs",
          "patience", "width", "depth", "epochs", "Dz", "Dt",
          "best_epoch", "best_monitor", "wall_s", "epoch_s_median"]


def main(args):
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"not found: {root}")

    runs, missing = {}, []
    for log_path in sorted(root.rglob("log.json")):
        rel = log_path.parent.relative_to(root).as_posix()
        try:
            d = json.loads(log_path.read_text())
        except Exception as e:
            missing.append(f"{rel}: unreadable ({e})")
            continue

        rows = d.get("rows", [])
        if not rows:
            missing.append(f"{rel}: no epoch rows")
            continue

        entry = {k: d.get(k) for k in CONFIG}
        entry["n_epochs_logged"] = len(rows)
        # Flag any run that stopped short, since a truncated run must not be
        # read alongside complete ones without the reader knowing.
        entry["complete"] = (len(rows) == d.get("epochs"))
        for name, get in SERIES:
            entry[name] = [get(r) for r in rows]
        runs[rel] = entry

        flag = "" if entry["complete"] else "  INCOMPLETE"
        print(f"{rel:<34} lam {str(entry['lam']):<5} "
              f"seed {str(entry['seed']):<4} {entry['normalisation']:<6} "
              f"frac {entry['train_frac']}  "
              f"best {entry['best_monitor']:.4e} @ep{entry['best_epoch']}{flag}")

    out = {"root": str(root), "n_runs": len(runs),
           "problems": missing, "runs": runs}
    Path(args.out).write_text(json.dumps(out))
    size_kb = Path(args.out).stat().st_size / 1024

    print("-" * 78)
    print(f"{len(runs)} runs collected -> {args.out}  ({size_kb:.0f} kB)")
    if missing:
        print("PROBLEMS:")
        for m in missing:
            print("  " + m)


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import paths

    p = argparse.ArgumentParser(
        description="Reduce a directory of training runs to one summary file")
    p.add_argument("--root", default="runs/sweep",
                   help="Directory of run folders, each with a log.json")
    p.add_argument("--out", default=None,
                   help="Output JSON. Default: "
                        "results/training_runs_summary.json")
    a = p.parse_args()

    if a.out is None:
        os.makedirs(paths.RESULTS, exist_ok=True)
        a.out = paths.result("training_runs_summary.json")

    main(a)
