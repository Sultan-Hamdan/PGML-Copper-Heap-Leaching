r"""
build_extrapolation_soils.py
============================
Writes the soil parameter tables for the extrapolation study.

Arithmetic over the training domain plus one scrambled Sobol draw. Nothing is
simulated and no dataset file is opened; the closed loop advances the column
live, so these soils need parameters only.

PRODUCES, under --out
   8 single-property tables, 1024 rows each
  16 joint tables,            128 rows each
  extrapolation_manifest.json with a sha256 per table

DESIGN
  Four properties vary: Ks, alpha, n_vgm, theta_s. Se_i is the population
  variable. Distances are 5, 10 and 20 percent of each property's own domain
  range, beyond the bound, both directions. 20 percent is the last level: 25
  percent below the n bound reaches n = 2, where m = 1 - 2/n is zero and
  delta = 3 + 2/(n-2) diverges.

  Single-property cells share one Sobol background of 256 points in 5D,
  overwriting one property each, so cells are comparable to each other. Joint
  cells pin all four at 10 percent and draw 64 Se_i points in 1D. Both
  scrambled at seed 42; counts are powers of two for Owen's balance property.

ORDER IS A CONTRACT
  "ids" and "rows" are written in the same order. closed_loop.py consumes
  r_max positionally after matching the id lists, which is what keeps each
  soil on its own stability limit.

Usage, from 06_Control/extrapolation:
    python -u build_extrapolation_soils.py --out ../../reference/extrapolation/soils

--scaler defaults through paths.py to reference/scaler.json.

Status: stable
Depends on: paths.py, numpy, scipy

Changelog
---------
2026.09.07  v1.2  Repository release. --scaler defaults through paths.py.
                  Header wording only otherwise.
2026.08.24  v1.1  Manifest now carries TWO hashes per table. file_sha256
                  covers the whole document including created_utc, so it
                  changes on every run and can only answer "has this file
                  been altered since it was written". rows_sha256 covers
                  the soil numbers alone and is stable across runs and
                  machines, so it is the one that answers "did this draw
                  reproduce". v1.0 had only the first and it was mistaken
                  for the second.
2026.08.24  v1.0  Created. Eight single-property and sixteen joint tables.
                  Background draw 256 points in 5D, joint draw 64 points in 1D,
                  both scrambled Sobol at seed 42. Cap audit reported per cell.
                  Six construction checks, abort on the first five.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
import paths

# ---------------------------------------------------------------------------
# fixed constants
# ---------------------------------------------------------------------------

# Training domain. Keys are the four soil properties plus Se_i.
BOUNDS = {
    "Ks":      (130.0,   232.0),     # cm/day
    "alpha":   (0.025,   0.042),     # 1/cm
    "n_vgm":   (2.20,    3.00),      # dimensionless
    "theta_s": (0.270,   0.388),     # dimensionless, theta_r = 0
    "Se_i":    (0.40,    0.63),      # dimensionless
}

# The four that reach the network. Order fixes the joint combination letters.
PROPERTIES = ["Ks", "alpha", "n_vgm", "theta_s"]

# Short labels used in cell names and in Chapter 4 prose.
LABEL = {"Ks": "Ks", "alpha": "alpha", "n_vgm": "n", "theta_s": "theta_s"}

THETA_R = 0.0
AREA = 308.0        # m2
IRRIGATION_CAP = 0.30   # R/Ks, eq:qub

DISTANCES = [5, 10, 20]     # percent of the property's domain range
JOINT_DISTANCE = 10         # percent, joint cells only

N_BACKGROUND = 256          # 2^8, single-property background population
N_JOINT = 64                # 2^6, Se_i draws per joint cell
DRAW_SEED = 42

DEC = {"Ks": 2, "alpha": 3, "n_vgm": 2, "theta_s": 3, "Se_i": 3}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def level(prop, direction, pct):
    """Value of `prop` at `pct` percent of its domain range beyond a bound.

    direction "low"  -> lo - (pct/100) * width
    direction "high" -> hi + (pct/100) * width
    pct = 0 returns the bound itself, which is the paired baseline.
    """
    lo, hi = BOUNDS[prop]
    width = hi - lo
    if direction == "low":
        return lo - (pct / 100.0) * width
    if direction == "high":
        return hi + (pct / 100.0) * width
    raise ValueError(f"direction must be low or high, got {direction!r}")


def delta_of_n(n):
    """VGM conductivity exponent, eq:vgm_k. Diverges at n = 2."""
    return 3.0 + 2.0 / (n - 2.0)


def sobol_scaled(dims, n_points, seed):
    """Scrambled Sobol, first n_points, scaled to the bounds of `dims`."""
    pts = qmc.Sobol(len(dims), scramble=True, seed=seed).random(n_points)
    lo = [BOUNDS[d][0] for d in dims]
    hi = [BOUNDS[d][1] for d in dims]
    return qmc.scale(pts, lo, hi)


def rows_digest(rows):
    """sha256 over the soil numbers alone, at full float precision.

    Deliberately excludes created_utc and every other header field, so the
    value is stable across runs and across machines. This is the digest that
    answers whether a draw reproduced. The whole-document hash cannot, since
    it carries the timestamp.
    """
    parts = []
    for r in rows:
        parts.append("|".join([
            r["id"], repr(r["Ks"]), repr(r["alpha"]), repr(r["n_vgm"]),
            repr(r["theta_s"]), repr(r["Se_i"]),
        ]))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def make_row(rid, cell, role, distance, pair_id, params):
    row = {
        "id": rid,
        "cell": cell,
        "role": role,
        "distance": distance,
        "pair_id": int(pair_id),
        "theta_r": THETA_R,
    }
    for k in ("Ks", "alpha", "n_vgm", "theta_s", "Se_i"):
        row[k] = float(params[k])
    return row


# ---------------------------------------------------------------------------
# table construction
# ---------------------------------------------------------------------------

def build_single_property(background):
    """Eight tables. Each reuses the shared background and overwrites one
    property: at its bound for the baseline, beyond it for each distance."""
    tables = {}
    for prop in PROPERTIES:
        for direction in ("low", "high"):
            cell = f"sp_{LABEL[prop]}_{direction}"
            rows = []

            for k in range(N_BACKGROUND):
                p = {key: background[key][k] for key in background}
                p[prop] = level(prop, direction, 0)
                rows.append(make_row(f"{cell}__base__{k:04d}",
                                     cell, "baseline", 0, k, p))

            for pct in DISTANCES:
                value = level(prop, direction, pct)
                for k in range(N_BACKGROUND):
                    p = {key: background[key][k] for key in background}
                    p[prop] = value
                    rows.append(make_row(f"{cell}__test_{pct:02d}__{k:04d}",
                                         cell, "test", pct, k, p))

            tables[cell] = {
                "kind": "single_property",
                "property": prop,
                "direction": direction,
                "rows": rows,
            }
    return tables


def build_joint(se_i_draw):
    """Sixteen tables. Every low/high combination of the four properties,
    all four pinned, Se_i the only variable."""
    tables = {}
    for mask in range(16):
        # bit 3 -> PROPERTIES[0], bit 0 -> PROPERTIES[3]
        dirs = ["H" if (mask >> (3 - i)) & 1 else "L"
                for i in range(len(PROPERTIES))]
        cell = "jt_" + "".join(dirs)
        base_vals, test_vals = {}, {}
        for prop, d in zip(PROPERTIES, dirs):
            direction = "low" if d == "L" else "high"
            base_vals[prop] = level(prop, direction, 0)
            test_vals[prop] = level(prop, direction, JOINT_DISTANCE)

        rows = []
        for k in range(N_JOINT):
            p = dict(base_vals)
            p["Se_i"] = se_i_draw[k]
            rows.append(make_row(f"{cell}__base__{k:04d}",
                                 cell, "baseline", 0, k, p))
        for k in range(N_JOINT):
            p = dict(test_vals)
            p["Se_i"] = se_i_draw[k]
            rows.append(make_row(f"{cell}__test_{JOINT_DISTANCE:02d}__{k:04d}",
                                 cell, "test", JOINT_DISTANCE, k, p))

        tables[cell] = {
            "kind": "joint",
            "property": "+".join(PROPERTIES),
            "direction": "".join(dirs),
            "rows": rows,
        }
    return tables


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def run_checks(tables, scaler):
    """Five aborting checks and one reporting check. Returns (ok, cap_audit)."""
    ok = True
    r_min = float(scaler["R_out"]["min"])
    r_max = float(scaler["R_out"]["max"])
    all_ids = []
    cap_audit = {}

    for cell, t in sorted(tables.items()):
        rows = t["rows"]

        bad_n = [r["id"] for r in rows if not r["n_vgm"] > 2.0]
        if bad_n:
            print(f"ABORT check 1  {cell}: n_vgm <= 2.0 on {len(bad_n)} rows, "
                  f"first {bad_n[0]}")
            ok = False

        bad_p = [r["id"] for r in rows
                 if not (r["theta_s"] > 0.0 and 0.0 < r["Se_i"] < 1.0)]
        if bad_p:
            print(f"ABORT check 2  {cell}: theta_s <= 0 or Se_i outside (0,1) "
                  f"on {len(bad_p)} rows, first {bad_p[0]}")
            ok = False

        bad_r = [r["id"] for r in rows
                 if not IRRIGATION_CAP * r["Ks"] > r_min]
        if bad_r:
            print(f"ABORT check 3  {cell}: 0.30*Ks below the scaler R_out "
                  f"minimum on {len(bad_r)} rows, first {bad_r[0]}")
            ok = False

        all_ids.extend(r["id"] for r in rows)

        base_pairs = {r["pair_id"] for r in rows if r["role"] == "baseline"}
        orphan = [r["id"] for r in rows
                  if r["role"] == "test" and r["pair_id"] not in base_pairs]
        if orphan:
            print(f"ABORT check 5  {cell}: {len(orphan)} test rows without a "
                  f"baseline partner, first {orphan[0]}")
            ok = False

        # check 6, the cap audit. Reported, never aborting.
        binding, shortfall = 0, []
        for r in rows:
            phys = IRRIGATION_CAP * r["Ks"]
            if phys > r_max:
                binding += 1
                shortfall.append(phys - r_max)
        cap_audit[cell] = {
            "rows": len(rows),
            "rows_harness_capped": binding,
            "share_harness_capped": round(binding / len(rows), 4),
            "mean_shortfall_cm_per_day":
                round(float(np.mean(shortfall)), 4) if shortfall else 0.0,
            "max_shortfall_cm_per_day":
                round(float(np.max(shortfall)), 4) if shortfall else 0.0,
        }

    if len(all_ids) != len(set(all_ids)):
        print(f"ABORT check 4  ids are not unique across tables: "
              f"{len(all_ids)} written, {len(set(all_ids))} distinct")
        ok = False

    return ok, cap_audit


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def report_levels():
    print("LEVELS, value = lo - d*width  or  hi + d*width")
    head = (f"  {'property':<9}" + "".join(f"{-p:>10}%" for p in DISTANCES[::-1])
            + f"{'lo':>11}{'hi':>11}"
            + "".join(f"{'+' + str(p):>10}%" for p in DISTANCES))
    print(head)
    for prop in PROPERTIES:
        d = DEC[prop]
        lo, hi = BOUNDS[prop]
        cells = [f"{level(prop, 'low', p):>11.{d}f}" for p in DISTANCES[::-1]]
        cells += [f"{lo:>11.{d}f}", f"{hi:>11.{d}f}"]
        cells += [f"{level(prop, 'high', p):>11.{d}f}" for p in DISTANCES]
        print(f"  {LABEL[prop]:<9}" + "".join(cells))
    print()
    print("  conductivity exponent delta = 3 + 2/(n-2) at the n levels")
    ns = [level("n_vgm", "low", p) for p in DISTANCES[::-1]]
    ns += [BOUNDS["n_vgm"][0], BOUNDS["n_vgm"][1]]
    ns += [level("n_vgm", "high", p) for p in DISTANCES]
    print("    " + "".join(f"n={n:.2f} d={delta_of_n(n):>6.2f}   " for n in ns))
    print()


def report_cap_audit(cap_audit, scaler):
    r_max = float(scaler["R_out"]["max"])
    print(f"CAP AUDIT  against scaler R_out max = "
          f"{r_max:.4f} cm/day")
    print("  a row is harness capped when 0.30*Ks exceeds that value, so")
    print("  r_max is set by the scaler rather than by eq:qub. Shortfall is")
    print("  in cm/day. This is NOT a probability.")
    print(f"  {'cell':<16}{'rows':>7}{'capped':>9}{'share':>9}"
          f"{'mean short':>12}{'max short':>11}")
    for cell, a in sorted(cap_audit.items()):
        if a["rows_harness_capped"] == 0:
            continue
        print(f"  {cell:<16}{a['rows']:>7}{a['rows_harness_capped']:>9}"
              f"{a['share_harness_capped']:>9.3f}"
              f"{a['mean_shortfall_cm_per_day']:>12.3f}"
              f"{a['max_shortfall_cm_per_day']:>11.3f}")
    clean = [c for c, a in cap_audit.items() if a["rows_harness_capped"] == 0]
    print(f"  {len(clean)} of {len(cap_audit)} cells have no harness capped rows")
    print()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="soils",
                   help="directory for the soil tables")
    p.add_argument("--scaler", default=None,
                   help="scaler.json, read for the R_out bounds. "
                        "Default: reference/scaler.json")
    p.add_argument("--draw_seed", type=int, default=DRAW_SEED)
    a = p.parse_args()

    if a.scaler is None:
        a.scaler = paths.reference("scaler.json")

    if not os.path.isfile(a.scaler):
        print(f"ABORT: no scaler at {a.scaler}")
        return 1
    with open(a.scaler, "r") as fh:
        scaler = json.load(fh)

    print("=" * 70)
    print("EXTRAPOLATION SOIL TABLES")
    print("=" * 70)
    print(f"scaler        : {a.scaler}")
    print(f"                R_out min {scaler['R_out']['min']:.4f}  "
          f"max {scaler['R_out']['max']:.4f}  cm/day")
    print(f"draw seed     : {a.draw_seed}")
    print(f"background    : {N_BACKGROUND} soils, 5D scrambled Sobol")
    print(f"joint Se_i    : {N_JOINT} draws, 1D scrambled Sobol")
    print(f"distances     : {DISTANCES} % of the domain range, both directions")
    print(f"joint distance: {JOINT_DISTANCE} %")
    print()

    report_levels()

    dims = ["Ks", "alpha", "n_vgm", "theta_s", "Se_i"]
    s = sobol_scaled(dims, N_BACKGROUND, a.draw_seed)
    background = {d: s[:, i] for i, d in enumerate(dims)}
    print(f"background draw, {N_BACKGROUND} soils in {len(dims)}D")
    for d in dims:
        k = DEC[d]
        print(f"  {d:<9} [{background[d].min():.{k}f}, "
              f"{background[d].max():.{k}f}]")
    print()

    se_i_draw = sobol_scaled(["Se_i"], N_JOINT, a.draw_seed)[:, 0]
    print(f"joint Se_i draw, {N_JOINT} points  "
          f"[{se_i_draw.min():.3f}, {se_i_draw.max():.3f}]")
    print()

    tables = build_single_property(background)
    tables.update(build_joint(se_i_draw))

    ok, cap_audit = run_checks(tables, scaler)
    total = sum(len(t["rows"]) for t in tables.values())
    print(f"CHECKS         {'PASS' if ok else 'FAIL'}   "
          f"{len(tables)} tables, {total} rows")
    print()
    if not ok:
        print("nothing written")
        return 1

    report_cap_audit(cap_audit, scaler)

    os.makedirs(a.out, exist_ok=True)
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest = {"created_utc": created, "draw_seed": a.draw_seed,
                "n_background": N_BACKGROUND, "n_joint": N_JOINT,
                "distances_pct": DISTANCES,
                "joint_distance_pct": JOINT_DISTANCE,
                "bounds": {k: list(v) for k, v in BOUNDS.items()},
                "scaler_R_out": scaler["R_out"],
                "cap_audit": cap_audit, "tables": []}

    for cell, t in sorted(tables.items()):
        rows = t["rows"]
        rdig = rows_digest(rows)
        doc = {"cell": cell, "kind": t["kind"], "property": t["property"],
               "direction": t["direction"], "created_utc": created,
               "draw_seed": a.draw_seed, "n_rows": len(rows),
               "rows_sha256": rdig,
               "area": AREA, "theta_r": THETA_R,
               "bounds": {k: list(v) for k, v in BOUNDS.items()},
               "distances_pct": sorted({r["distance"] for r in rows}),
               "ids": [r["id"] for r in rows],
               "rows": rows}
        path = os.path.join(a.out, f"{cell}.json")
        blob = json.dumps(doc, indent=1)
        with open(path, "w") as fh:
            fh.write(blob)
        manifest["tables"].append({
            "cell": cell, "file": f"{cell}.json", "n_rows": len(rows),
            "rows_sha256": rdig,
            "file_sha256": hashlib.sha256(blob.encode()).hexdigest()[:16]})

    with open(os.path.join(a.out, "extrapolation_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)

    print(f"written to {a.out}")
    print(f"  {len(tables)} tables, {total} rows, "
          f"plus extrapolation_manifest.json")
    print()
    print("ROWS DIGEST, soil numbers only, stable across runs and machines")
    for t in manifest["tables"]:
        print(f"  {t['cell']:<18}{t['rows_sha256']}")
    print("  file_sha256 also written, but it covers created_utc and so")
    print("  changes on every run. Compare rows_sha256 to check a draw.")
    print()
    print("NEXT  one ceiling per table, then one closed-loop run per table")
    print("      per seed. The ceiling file name must carry the cell name.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
