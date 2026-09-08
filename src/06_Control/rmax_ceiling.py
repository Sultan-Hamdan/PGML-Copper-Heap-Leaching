r"""
rmax_ceiling.py
===============
Measures the per-soil irrigation ceiling and writes it out for the controller.

DEFINITION
  R_stable(soil)  the largest rate for which one day from the driest state the
                  controller will ever see produces zero pre-clip excursions
                  outside [theta_r, theta_s], as reported by
                  run_batch(diag=True). It is the last stable rate, not the
                  first that floods.

  r_max = min(0.30 * Ks, safety * R_stable)       safety default 0.95

WHY IT IS NEEDED
  Above the surface's infiltration capacity the top boundary condition stops
  holding. It is a prescribed flux, so it assumes everything applied
  infiltrates; a real heap ponds and sheds the excess, and this scheme has no
  ponding term and no runoff term. Above R_stable the model does not represent
  the state badly, it does not represent it at all.

  The solver shows this as a Picard iteration that returns finite but
  unconverged values, which the clip then flattens into a profile that looks
  physical. Finite is not converged. Capping keeps the solver inside the
  model's stated domain.

  The cap is a property of the soil, not of the model, so every lambda case
  gets an identical action set and the closed-loop comparison stays clean.

WHY DAY 1
  The column starts uniform at Se_i and irrigation only adds water, so surface
  infiltration capacity only rises from there. Day 1 is the worst case.
  --se_floor overrides this with a fixed uniform Se if a stricter test is
  wanted.

FLAGS
  --split           val | test. Default val.
  --soil_table      JSON of constructed soils, for the extrapolation cells.
  --reference_soil  use the Cariaga 2015 soil, which has no dataset entry.
  --se_grid         initial saturations to test. Use "own" to match the
                    ceiling in force for the validation and test soils, which
                    was measured from each soil's own Se_i over one day. A
                    ceiling measured on any other grid gives a different action
                    set, and comparisons would then mix a network effect with a
                    difference in the moves available.
  --safety          margin on R_stable. Default 0.95.
  --n_R, --days, --div_tol, --res_tol   sweep resolution and thresholds.

Writes rmax_<split>.json: per-soil R_stable, r_max, and the metadata the
controller needs to confirm which ceiling it used.

Usage, from 06_Control:
  python -u rmax_ceiling.py
  python -u rmax_ceiling.py --soil_table ../reference/extrapolation/soils/sp_n_low.json --se_grid own

--root and --h5 default through paths.py.

Status: stable
Depends on: column_solver.py, paths.py, numpy, h5py

Changelog
---------
2026.09.07  v2.5  Repository release. Renamed from probe_flood_ceiling_v2.py.
                  R_flood renamed to R_stable, since it is the last stable rate
                  rather than the first that floods. Hardcoded absolute paths
                  replaced by paths.py; output renamed to rmax_<split>.json;
                  import updated to column_solver.
2026.08.24  v2.4  NaN now counts as a failure. A comparison against NaN returns
                  False, so an unevaluable rate registered as converged: mean
                  r_max 54.3 cm/day at n = 2.04 against 9.3 at n = 2.12.
                  --r_floor extends the grid below 0.30*Ks/n_R, where 19 rows
                  had recorded R_stable = 0. Additive, so soils above the old
                  floor are bit-identical and default 0 reproduces v2.3.
2026.08.24  v2.3  --soil_table reads constructed soils from a table instead of
                  the dataset. Output is named per cell so ceilings cannot be
                  crossed. Pass --se_grid own to match the val and test
                  ceilings, which were measured on a single state.
2026.08.17  v2.2  Banner printed the v1.2 form of r_max, understating it by 5
                  percent for non-binding soils. Print only.
2026.08.13  v2.1  Guarded the gap statistic, which crashed on a single soil.
2026.08.13  v2.0  --reference_soil for the Cariaga 2015 soil, which has no
                  dataset entry and so no measured ceiling. Uncapped, the
                  controller reached 50.23 of a permitted 51 cm/day and the
                  plant stopped converging.
2026.08.01  v1.4  Ceiling is the per-soil minimum over several initial
                  saturations, not day 1. A dry surface is worst for
                  conductivity, a wet column worst for storage, so the limit is
                  not monotone in wetness; the day-1 ceiling let 4 of 614 soils
                  diverge. Still one constant per soil, deliberately not
                  state-dependent, or comparability across lam would go.
2026.08.01  v1.3  Safety margin applies only where the stability limit binds. v1.2
                  shaved 5 percent off the cap for all 64 soils.
2026.08.01  v1.2  Thresholds computed after the sweep rather than baked into it.
                  v1.1 used res_tol = 1e-4, which mis-binned 55 of 64 soils. The
                  distribution is bimodal with a six order of magnitude gap, so
                  any threshold inside it gives the same answer.
2026.08.01  v1.1  Criterion is the final Picard pass plus its residual. v1.0
                  used any-pass excursion, counting normal early overshoot as
                  divergence.
2026.08.01  v1.0  Initial. Defines the ceiling on convergence rather than on
                  NaN, using the solver diag counters.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths

DEF_ROOT = paths.REPO_ROOT
DEF_H5 = paths.H5

SOIL_KEYS = ["Ks", "alpha", "n_vgm", "theta_s"]


def _sweep(states_ctx, theta0, R_cap, frac, Ks, ts, tr, soil, area, a,
           run_batch, ON_GPU, S):
    """Sweep the rate grid from one initial state, accumulating the per-rate
    worst excursion and residual into EXC / RES / ANY by elementwise maximum.
    Taking the max across states is what makes the resulting ceiling the
    minimum over states."""
    EXC, RES, ANY = states_ctx
    for j, fr in enumerate(frac):
        R = R_cap * fr
        Q = (R / 100.0 * area).reshape(-1, 1)
        Q = np.repeat(Q, a.days, axis=1)
        out = run_batch(Q, Ks=Ks, theta_s=ts, theta_r=tr,
                        alpha=soil["alpha"], n=soil["n_vgm"],
                        H=a.H, Area=area, tf=a.days, Dz=a.Dz, Dt=a.Dt,
                        D_H=a.D_H, n_iter=a.n_iter, theta0=theta0,
                        diag=True, div_tol=a.div_tol)
        dg = out[-1]
        EXC[j] = np.maximum(EXC[j], dg["max_excursion_final"])
        RES[j] = np.maximum(RES[j], dg["max_residual"])
        ANY[j] = np.maximum(ANY[j], dg["max_excursion"])
        if (j + 1) % 20 == 0 or j == len(frac) - 1:
            nb = int(((EXC > a.div_tol) | (RES > a.res_tol)).any(axis=0).sum())
            print(f"   rate {j+1:>3}/{len(frac)}  cumulative binding "
                  f"{nb:>4}/{S}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=DEF_ROOT)
    p.add_argument("--h5", default=DEF_H5)
    p.add_argument("--split", default="val")
    p.add_argument("--reference_soil", action="store_true",
                   help="Measure the ceiling for the Cariaga 2015 reference "
                        "soil instead of the split. That soil is not in the "
                        "h5, so it has no ceiling, and without one the "
                        "controller is capped only by 0.30*Ks. Use "
                        "--se_grid own to match how the 614 split soils were "
                        "measured, and --out to avoid overwriting their file.")
    p.add_argument("--soil_table", default="",
                   help="a constructed soil table from "
                        "build_extrapolation_soils.py. Parameters are read "
                        "from the table itself, so no h5 is opened and "
                        "--split, --soils_json and --n_soils are ignored. "
                        "Pass --se_grid own to match the ceiling in force "
                        "for the validation and test soils.")
    p.add_argument("--soils_json",
                   default=paths.reference("soils_val.json"))
    p.add_argument("--n_soils", type=int, default=0)
    p.add_argument("--n_R", type=int, default=80)
    p.add_argument("--r_floor", type=float, default=0.0,
                   help="absolute lowest irrigation rate to sweep, cm/day. "
                        "0 disables and reproduces the v2.2 grid exactly. "
                        "Set it to the controller's r_min (the scaler R_out "
                        "minimum, 0.1623) so the sweep covers the whole range "
                        "the controller can command. Rungs are ADDED below "
                        "the old floor; every original rung is still swept.")
    p.add_argument("--n_R_low", type=int, default=40,
                   help="rungs added below 0.30*Ks/n_R when --r_floor is set, "
                        "geometrically spaced")
    p.add_argument("--safety", type=float, default=0.95)
    p.add_argument("--se_floor", type=float, default=0.0,
                   help="if >0, test from this uniform Se instead of each "
                        "soil's own Se_i")
    p.add_argument("--se_grid", default="own,0.40,0.55,0.70,0.85,0.95",
                   help="initial states to test. The ceiling is the per-soil "
                        "MINIMUM over all of them. 'own' means each soil's own "
                        "Se_i. Empty string = single state from --se_floor.")
    p.add_argument("--days", type=int, default=1)
    p.add_argument("--div_tol", type=float, default=1e-3)
    p.add_argument("--res_tol", type=float, default=1.0,
                   help="max final-pass Picard residual, relative to the theta "
                        "span, still counted as converged")
    p.add_argument("--Dz", type=float, default=2.5)
    p.add_argument("--Dt", type=float, default=1.0 / 24.0)
    p.add_argument("--D_H", type=int, default=24)
    p.add_argument("--H", type=float, default=5.50)
    p.add_argument("--n_iter", type=int, default=20)
    p.add_argument("--out", default="")
    a = p.parse_args()

    if a.soil_table and a.reference_soil:
        print("ABORT: --soil_table and --reference_soil are mutually "
              "exclusive. Each supplies its own soils.")
        return
    if a.soil_table and not os.path.isfile(a.soil_table):
        print(f"ABORT: no soil table at {a.soil_table}")
        return

    paths.add_stage_paths()
    import h5py
    from column_solver import run_batch, ON_GPU
    import inspect
    if "diag" not in inspect.signature(run_batch).parameters:
        print("ABORT: column_solver has no diag= keyword.")
        return

    # ------------------------------------------------------------ soils
    cell = None
    if a.soil_table:
        # Constructed soils. The extrapolation tables carry
        # their parameters inline, so no h5 is opened. This is the reference
        # soil branch below generalised from one hardcoded soil to a table.
        #
        # ORDER IS A CONTRACT. build_extrapolation_soils.py writes "ids" and
        # "rows" in the same order, this file preserves it, and
        # closed_loop_pr8.py compares its ceiling's id list against the soil
        # list element by element before consuming r_max positionally. That
        # chain is the only thing keeping each soil on its own stability limit.
        with open(a.soil_table, "r") as fh:
            tab = json.load(fh)
        rows = tab["rows"]
        ids = [r["id"] for r in rows]
        if ids != list(tab["ids"]):
            print("ABORT: the soil table's ids do not match its rows in "
                  "order. Rebuild the table. The order is what keeps each "
                  "soil on its own ceiling.")
            return
        soil = {k: np.asarray([float(r[k]) for r in rows], dtype=np.float64)
                for k in SOIL_KEYS + ["Se_i", "theta_r"]}
        area = float(tab["area"])
        cell = tab["cell"]
        src = a.soil_table
        S = len(ids)
    elif a.reference_soil:
        # The Cariaga 2015 soil the baseline study uses. Its parameters sit
        # inside the Sobol envelope but it is not one of the 4096 samples, so
        # it has no entry in the h5 and no measured ceiling. Values are taken
        # from pgml_closed_loop.py, which hardcodes the same four.
        ids = ["reference_cariaga2015"]
        soil = {"Ks": np.array([170.0]), "alpha": np.array([0.035]),
                "n_vgm": np.array([2.267]), "theta_s": np.array([0.33]),
                "Se_i": np.array([0.14 / 0.33]), "theta_r": np.array([0.0])}
        area = 308.0
        src = "hardcoded reference soil (Cariaga 2015)"
        S = 1
    else:
        if os.path.isfile(a.soils_json):
            with open(a.soils_json, "r") as fh:
                ids = json.load(fh)["ids"]
            src = a.soils_json
        else:
            with open(os.path.join(a.root, "03_Preprocessing", "split.json")) as fh:
                pool = json.load(fh)[a.split]
            ids = pool if not a.n_soils else pool[:a.n_soils]
            src = f"split.json[{a.split}]"

        soil = {k: [] for k in SOIL_KEYS + ["Se_i", "theta_r"]}
        with h5py.File(a.h5, "r") as f:
            area = float(f.attrs["area"])
            for rid in ids:
                g = f[rid]
                for k in SOIL_KEYS + ["Se_i", "theta_r"]:
                    soil[k].append(float(g.attrs[k]))
        soil = {k: np.asarray(v, dtype=np.float64) for k, v in soil.items()}
        S = len(ids)

    Ks, ts, tr = soil["Ks"], soil["theta_s"], soil["theta_r"]
    Nz1 = int(a.H * 100 / a.Dz) + 1
    # Initial states to test. The day-1 state is NOT the worst case: a dry
    # surface is worst for conductivity, but a nearly saturated column is worst
    # for storage, so the stability limit is not monotone in wetness. A ceiling
    # measured only from Se_i let 4 of 614 soils diverge partway through. The
    # ceiling is therefore the per-soil MINIMUM over a spread of states.
    #
    # The ceiling stays ONE CONSTANT PER SOIL and is deliberately not made
    # state-dependent: a ceiling tracking the live column state would depend on
    # that lambda case's own past actions, so each lambda would get a different
    # action set and the comparison would stop being like for like.
    states = []
    if a.se_grid.strip():
        for tok in a.se_grid.split(","):
            tok = tok.strip()
            if not tok:
                continue
            states.append("own" if tok == "own" else float(tok))
    else:
        states = ["own" if a.se_floor <= 0 else a.se_floor]
    Se0_own = soil["Se_i"]

    print("=" * 78)
    print("FLOOD CEILING  --  convergence-defined, not NaN-defined")
    print("=" * 78)
    print(f"soils      : {S}  from {src}")
    print(f"start states: {states}  (ceiling = per-soil minimum over these)")
    print(f"R grid     : {a.n_R} points to 0.30*Ks per soil"
          + (f", plus {a.n_R_low} below, floor {a.r_floor:g} cm/day"
             if a.r_floor > 0 else ""))
    print(f"criterion  : zero FINAL-pass excursions (div_tol={a.div_tol:g}) "
          f"AND final Picard residual <= {a.res_tol:g}, over {a.days} day(s)")
    print("             any-pass excursions are reported but do NOT bind:")
    print("             early overshoot that settles is normal Picard behaviour")
    print(f"safety     : r_max = 0.30*Ks where the soil never fails; "
          f"{a.safety} * R_stable where it does\n")

    R_cap = 0.30 * Ks
    # THE GRID'S LOWER REACH.
    # v2.2 and earlier swept fractions of 0.30*Ks starting at 1/n_R, so the
    # lowest rate examined was 0.00375*Ks. For the n-low extrapolation soils
    # that is 0.51 to 0.87 cm/day, against a controller floor r_min of 0.1623.
    # The bottom three quarters of the controller's own range was never looked
    # at. In domain that was harmless: no soil of the 614 failed at the
    # first rung, the tightest sitting at 0.20 of 0.30*Ks, sixteen times the
    # old floor. Out of domain it is not harmless: 19 rows of the n-low cell at
    # 10 percent failed at the FIRST rate tested and were recorded as
    # R_stable = 0. That is not a measurement, it is the grid failing to reach.
    #
    # Farthing 2017 sec 5, on poor convergence in Richards solvers: "a
    # significant portion of this poor convergence and instability arises from
    # under resolution in space of sharp fronts". The response in that
    # literature is to refine and retry rather than to record the failure as a
    # result. Extending the sweep downward is that move applied to a rate grid.
    #
    # The extension is ADDITIVE. Every original rung is still swept, in the
    # same place, so any soil whose stability limit already sat above the old floor
    # returns a bit-identical answer. Only soils that failed at the old floor
    # see anything new. That is what makes this an instrument range rather than
    # a change to what is measured, and it is exact, not approximate.
    frac_hi = np.linspace(1.0 / a.n_R, 1.0, a.n_R)
    if a.r_floor > 0:
        # Lowest fraction that puts EVERY soil's first rung at or below
        # r_floor. Soils with a smaller R_cap are covered further down still.
        f_lo = a.r_floor / float(R_cap.max())
        if f_lo >= 1.0 / a.n_R:
            print(f"NOTE: --r_floor {a.r_floor:g} is already above the grid "
                  f"floor 0.30*Ks/{a.n_R}. No rungs added.")
            frac = frac_hi
        else:
            # Geometric below the old floor: the resolution that matters is
            # relative, and a linear extension would put one rung at r_floor
            # and the next back near the old floor.
            frac_lo = np.geomspace(f_lo, 1.0 / a.n_R, a.n_R_low, endpoint=False)
            frac = np.concatenate([frac_lo, frac_hi])
            print(f"grid floor : extended to r_floor {a.r_floor:g} cm/day, "
                  f"{a.n_R_low} geometric rungs added below 0.30*Ks/{a.n_R}")
            print(f"             lowest tested rate per soil now "
                  f"{(R_cap.min()*f_lo):.4f} to {(R_cap.max()*f_lo):.4f} cm/day")
    else:
        frac = frac_hi
    n_rates = len(frac)
    # Per-rate records, so R_stable can be recomputed at ANY tolerance after the
    # sweep rather than baking one in. The threshold then has to be justified
    # against the observed distribution instead of asserted up front.
    EXC = np.zeros((n_rates, S))   # final-pass excursion / span
    RES = np.zeros((n_rates, S))   # final Picard residual / span
    ANY = np.zeros((n_rates, S))   # any-pass excursion / span, reported only
    ok_upto = np.zeros(S)          # largest R with zero divergence so far
    diverged = np.zeros(S, dtype=bool)
    worst_exc = np.zeros(S)        # final-pass, the criterion
    worst_any = np.zeros(S)        # any-pass, reported only
    worst_res = np.zeros(S)        # final Picard residual
    first_div_R = np.full(S, np.nan)

    t0 = time.perf_counter()
    for si, st in enumerate(states):
        Se0 = Se0_own if st == "own" else np.full(S, float(st))
        theta0 = np.repeat((Se0 * (ts - tr) + tr)[:, None], Nz1, axis=1)
        label = "own Se_i" if st == "own" else f"Se={st}"
        print(f"\n-- state {si+1}/{len(states)}: {label}")
        _sweep(states_ctx=(EXC, RES, ANY), theta0=theta0, R_cap=R_cap,
               frac=frac, Ks=Ks, ts=ts, tr=tr, soil=soil, area=area, a=a,
               run_batch=run_batch, ON_GPU=ON_GPU, S=S)
    # ---------------------------------------------------------- distribution
    print("\n" + "=" * 78)
    print("DISTRIBUTION  --  read the threshold off the data, do not invent it")
    print("=" * 78)
    wexc = EXC.max(axis=0)
    wres = RES.max(axis=0)
    print("worst final-pass excursion per soil, sorted:")
    se = np.sort(wexc)
    print("  " + "  ".join(f"{v:.3g}" for v in se[:8]) + "   ...   "
          + "  ".join(f"{v:.3g}" for v in se[-6:]))
    print("worst final Picard residual per soil, sorted:")
    sr = np.sort(wres)
    print("  " + "  ".join(f"{v:.3g}" for v in sr[:8]) + "   ...   "
          + "  ".join(f"{v:.3g}" for v in sr[-6:]))
    # The gap statistic reads a threshold off the spread ACROSS soils, so it
    # needs at least two. A single-soil run has no spread and nothing to read.
    if S < 2:
        print("largest gap: not computed, a single soil has no spread")
    else:
        for nm, v in (("excursion", se), ("residual", sr)):
            gaps = v[1:] / np.maximum(v[:-1], 1e-300)
            k = int(np.argmax(gaps))
            print(f"largest {nm} gap: {v[k]:.4g} -> {v[k+1]:.4g}  "
                  f"({gaps[k]:.3g}x)  splits {k+1} low / {len(v)-k-1} high")

    print("\n" + "=" * 78)
    print("THRESHOLD SENSITIVITY  --  n soils whose ceiling binds below 0.30*Ks")
    print("=" * 78)
    print(f"{'div_tol':>12}{'res_tol':>12}{'n binding':>12}")
    for dt in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e2, 1e4):
        bad_any = ((EXC > dt) | (RES > max(dt, 1.0))
                   | np.isnan(EXC) | np.isnan(RES)).any(axis=0)
        print(f"{dt:>12.0e}{max(dt,1.0):>12.0e}{int(bad_any.sum()):>12}")
    print("If this column is flat across several decades, the threshold choice")
    print("does not affect the result and cannot have been tuned to it.")

    # ------------------------------------------------------- final ceiling
    # NaN is a FAILURE, not a pass. A comparison against NaN returns False, so
    # a rate that returned NaN would otherwise register as converged and hand
    # the soil its full 0.30*Ks. The n-low extrapolation cell at 20 percent
    # (n = 2.04, conductivity exponent 53) returned NaN on all 256 rows and
    # came out with a HIGHER mean ceiling than the same soils at 10 percent,
    # which is physically backwards. A rate the simulator could not evaluate is
    # a rate the controller must not be given.
    nan_mask = np.isnan(EXC) | np.isnan(RES)
    n_nan_rates = int(nan_mask.any(axis=1).sum())
    n_nan_soils = int(nan_mask.any(axis=0).sum())
    fail = (EXC > a.div_tol) | (RES > a.res_tol) | nan_mask   # (n_R, S)
    if n_nan_soils:
        print(f"\nNaN GUARD: the simulator returned NaN for {n_nan_soils} of "
              f"{S} soils on {n_nan_rates} of {a.n_R} rates.")
        print("           Those rates count as failures. Without this guard "
              "they read as converged.")
    R_stable = np.empty(S)
    for i in range(S):
        f = np.where(fail[:, i])[0]
        R_stable[i] = R_cap[i] if f.size == 0 else R_cap[i] * frac[f[0] - 1] \
            if f[0] > 0 else 0.0
    diverged = fail.any(axis=0)
    worst_exc = wexc
    worst_res = wres
    worst_any = ANY.max(axis=0)
    for i in range(S):
        f = np.where(fail[:, i])[0]
        first_div_R[i] = np.nan if f.size == 0 else R_cap[i] * frac[f[0]]
    # Safety margin applies ONLY where the stability limit binds. R_stable is
    # measured on a finite rate grid and carries grid uncertainty, so it earns
    # a margin. The 0.30*Ks cap is an exact declared bound with no
    # uncertainty to hedge, so shaving it would narrow the controller's action
    # set for no reason. v1.2 applied the margin to both and gave every soil
    # 0.95*0.30*Ks; caught by verify T3 reporting 64 of 64 soils as binding.
    r_max = np.where(diverged, np.minimum(R_cap, a.safety * R_stable), R_cap)

    print(f"\nswept in {time.perf_counter()-t0:.1f}s")
    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"soils failing the criterion at or below 0.30*Ks : "
          f"{int(diverged.sum())} of {S}")
    print(f"  worst final-pass excursion / span : max {worst_exc.max():.6g}")
    print(f"  worst any-pass  excursion / span : max {worst_any.max():.6g}"
          f"   (reported only)")
    print(f"  worst final Picard residual      : max {worst_res.max():.6g}"
          f"   (tol {a.res_tol:g})")
    n_exc_only = int(((worst_exc > a.div_tol) & (worst_res <= a.res_tol)).sum())
    n_res_only = int(((worst_exc <= a.div_tol) & (worst_res > a.res_tol)).sum())
    n_both = int(((worst_exc > a.div_tol) & (worst_res > a.res_tol)).sum())
    print(f"  failed on bounds only {n_exc_only}, residual only {n_res_only}, "
          f"both {n_both}")
    print(f"binding ceiling is the stability limit, not 0.30*Ks, for "
          f"{int((r_max < R_cap - 1e-12).sum())} of {S} soils")
    rel = R_stable / R_cap
    print(f"R_stable / 0.30Ks : min {rel.min():.4f}  median "
          f"{np.median(rel):.4f}  max {rel.max():.4f}")

    # A constructed table runs 1024 soils and printing every one buries the
    # log. Only the tightest matter, since the ordering is by R_stable/0.30Ks.
    # The split path is untouched, so val and test console output is unchanged.
    order = np.argsort(rel)
    n_show = len(order)
    if cell is not None and len(order) > 60:
        n_show = 30
        print(f"\nshowing the {n_show} tightest of {len(order)} rows, "
              f"sorted by R_stable / 0.30Ks. Full arrays are in the json.")
    idw = max(10, max(len(str(x)) for x in ids)) if cell is not None else 10
    print(f"\n{'run':>{idw}}{'Ks':>9}{'n':>7}{'Se_i':>8}{'0.30Ks':>10}"
          f"{'R_stable':>10}{'r_max':>10}{'binding':>10}")
    for i in order[:n_show]:
        print(f"{ids[i]:>{idw}}{Ks[i]:>9.2f}{soil['n_vgm'][i]:>7.3f}"
              f"{Se0[i]:>8.4f}{R_cap[i]:>10.3f}{R_stable[i]:>10.3f}"
              f"{r_max[i]:>10.3f}"
              f"{('stable' if r_max[i] < R_cap[i] - 1e-12 else '0.30Ks'):>10}")

    print("\nassociation of R_stable/0.30Ks with soil parameters")
    for k, v in [("Se_i", Se0), ("Ks", Ks), ("alpha", soil["alpha"]),
                 ("n_vgm", soil["n_vgm"]), ("theta_s", ts)]:
        print(f"  {k:9s} pearson r = {np.corrcoef(v, rel)[0,1]:+.4f}")

    # A reference-soil or constructed run must not overwrite the split ceiling
    # in force. A constructed ceiling is named for its cell, which is what
    # stops one cell's ceiling reaching another cell's run.
    out_path = a.out or (
        f"rmax_{cell}.json" if cell is not None
        else "rmax_reference.json" if a.reference_soil
        else f"rmax_{a.split}.json")
    with open(out_path, "w") as fh:
        json.dump({
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_soils": src,
            "cell": cell,
            "split": None if cell is not None else a.split,
            "safety": a.safety,
            "div_tol": a.div_tol,
            "days_tested": a.days,
            "start_states": [str(x) for x in states],
            "n_R": a.n_R,
            "n_rates_swept": int(n_rates),
            "r_floor": a.r_floor,
            "n_R_low": a.n_R_low if a.r_floor > 0 else 0,
            "ids": ids,
            "Ks": Ks.tolist(),
            "R_cap_F16": R_cap.tolist(),
            "R_stable": R_stable.tolist(),
            "r_max": r_max.tolist(),
            "diverged_at_or_below_cap": diverged.tolist(),
            "first_divergent_R": first_div_R.tolist(),
            "res_tol": a.res_tol,
            "criterion": "final_pass_excursion_and_picard_residual",
            "worst_relative_excursion_final": worst_exc.tolist(),
            "worst_relative_excursion_anypass": worst_any.tolist(),
            "worst_final_residual": worst_res.tolist(),
        }, fh, indent=2)
    print(f"\nwrote {out_path}")
    print("The harness must load this and use r_max per soil. The same ceiling")
    print("applies to every lambda case, so the action set is identical across")
    print("the comparison.")


if __name__ == "__main__":
    main()
