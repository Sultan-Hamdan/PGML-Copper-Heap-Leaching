"""
fit_alpha_g.py
==============
Per-run sweep of the Gardner alpha_g in the physics residual, and the fit that
produces the rule train.py uses.

Context
    Of the three Gardner quantities, two are fixed by closed forms. beta is
    delta(n) = 3 + 2/(n - 2) per run, which makes the Gardner conductivity
    identical to the plant's since both write K = Ks * Se**exponent; Olivares
    2025 section 4 identified 10.49 for the Cariaga soil where delta(2.267) =
    10.4906. D is Olivares 2025 Eq. (14), carried by physics_loss.py. alpha_g
    has no published rule in either paper, so it is measured here.

What this does
    Within one run the soil is fixed and beta is determined, so alpha_g is the
    only free number. For each run it is swept over a log grid, scored on that
    run's own hourly pairs, and the optimum refined below grid resolution by a
    three-point log-parabola.

Candidate closed form tested
    alpha_g = delta(n) * alpha_vgm, i.e. Gardner and van Genuchten sharing an
    inverse capillary length. For each run the predicted value is computed and
    the ratio alpha_g_optimum / prediction is reported, with the rank
    correlation against the prediction and a log-log slope. That form would give
    ratio near 1 and slope near 1. It does not hold: the measured slope is
    negative, so the relation is fitted from the data instead. The result is
    alpha_g = 2.736 * (n - 2)**0.711 * alpha_vgm, which is what train.py
    evaluates.

Two scores, kept separate
    resid   mean squared residual on the TRUE next step. This SELECTS.
    stasis  residual of a no-change prediction divided by residual of the true
            step. This is a GATE: it must exceed 1, or the penalty scores
            standing still better than the truth and is unusable however small
            its residual. It is not a score and ranks nothing above 1. The
            margin at each run's residual optimum is checked and any run
            failing the gate is flagged.

SAMPLING. Every hourly pair, stride 1. An even stride censors every day
boundary, because they sit at t = 23 + 24k, always odd: stride 4 samples only
indices divisible by 4 and contains not a single irrigation change. That defect
voided two earlier sweeps. The stride argument remains for diagnostics but the
default is 1 and any other value prints a warning.

Usage, from 04_Training
    python -u fit_alpha_g.py --n_runs 512
    python -u fit_alpha_g.py --n_runs 2048 --run_offset 2048
    python -u fit_alpha_g.py --check --run_offset 2048

--h5 defaults through paths.py to data/dataset.h5. Writes its own ASCII log
alongside itself, so no shell redirection is needed.

Status: stable. Produces the alpha_g rule that train.py evaluates.
Depends on: physics_loss.py, paths.py, h5py, torch, numpy

Check mode (--check)
    Out-of-sample comparison of two candidate policies for setting alpha_g,
    on runs SEPARATE from the ones the rule was fitted on (use --run_offset).
    For each run, the residual is scored at three values:
        own      the run's own refined optimum (the floor)
        frozen   a single fixed alpha_g for all runs (--frozen)
        rule     A * (n - 2)**p * alpha_vgm    (--rule_A, --rule_p)
        rucker   the published capillary-drive closed form, nothing fitted
    and the excess res(x)/res(own) - 1 is reported for frozen and rule, with
    the stasis gate checked at the rule's value on every run. Adoption
    criterion: adopt the rule iff the gate passes on every run AND the rule's
    median excess is at most half the frozen median excess. Otherwise freeze.

Changelog
---------
2026.09.07  v1.5  Repository release. Renamed from probe_alpha_g.py. --h5
                  defaults through paths.py; output CSV and log names follow
                  the new file name. Header wording only otherwise.
2026.07.31  v1.4  Added Rucker's published capillary-drive closed form as a
                  third policy in --check. It had been cited as support for the
                  fitted rule's direction but never measured. Adds three columns
                  to the check CSV and a second verdict block.
2026.07.31  v1.3  Added --check mode: out-of-sample comparison of frozen against
                  rule, per-run excess over each run's own optimum, gate check
                  at the rule's value, and the adoption criterion printed with
                  the result. Defaults --frozen 0.0682, --rule_A 2.736,
                  --rule_p 0.711 from the 512-run fit at offset 0.
2026.07.31  v1.2  Stride default 4 -> 1. An even stride samples no day boundary
                  at all, since t = 23 + 24k is always odd, the censoring that
                  voided two earlier sweeps; any stride other than 1 now warns.
                  Removed a local diffusivity override that duplicated the
                  module. Added the per-run test of the candidate closed form,
                  three-point log-parabola refinement of each optimum, the gate
                  check at the optimum, and a self-written ASCII log. Grid
                  default 1e-3..1e3 (25) -> 0.003..30 (31).
2026.07.30  v1.1  Added --run_offset so separate halves of the dataset can be
                  run independently. Added a high-Se node breakdown.
2026.07.30  v1.0  Initial. Per-run alpha_g sweep with beta = delta(n) and the
                  corrected D prefactor wired in locally. Scores on residual
                  and stasis margin. Writes per-run optima to CSV.
"""

import argparse
import os
import sys
import time

import h5py
import math

import numpy as np
import torch

# probes live one level below the module they test
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(_HERE))))
import paths
paths.add_stage_paths()

import physics_loss as PL


BAR = "=" * 74


class Tee(object):
    def __init__(self, path):
        self.stream = open(path, "w", encoding="ascii", errors="replace")
        self.stdout = sys.stdout
        self.path = path

    def write(self, s):
        self.stdout.write(s)
        self.stream.write(s)

    def flush(self):
        self.stdout.flush()
        self.stream.flush()

    def close(self):
        try:
            self.stream.close()
        except Exception:
            pass


def alpha_capillary_drive(n, alpha_vgm):
    """
    Gardner alpha from the capillary-drive equivalence of Rucker, Warrick and
    Ferre (2005), method 1: preserve H = integral of kr(h) dh over the full
    suction range, and set alpha_G = 1/H. For the van Genuchten conductivity
    with the Burdine constraint m = 1 - 2/n, the exponent product is
    m*delta = (3n - 4)/n and the integral closes in Gamma functions:

        alpha_G = alpha_vgm * n * Gamma((3n-4)/n)
                  / [ Gamma(1/n) * Gamma((3n-5)/n) ]

    Computed through lgamma so the three factors never overflow independently.
    All three arguments are strictly positive for n above 5/3, so the sampled
    box is safely inside the domain.

    This is a PUBLISHED closed form, not a fit. It is scored here on the same
    held-out runs as the fitted rule so the two are compared on one axis
    rather than argued about.
    """
    n = np.asarray(n, dtype=float)
    a1 = (3.0 * n - 4.0) / n
    a2 = 1.0 / n
    a3 = (3.0 * n - 5.0) / n
    if np.any(a1 <= 0) or np.any(a2 <= 0) or np.any(a3 <= 0):
        raise ValueError("Gamma argument outside the domain; n too small")
    lg = np.vectorize(math.lgamma)
    ratio = n * np.exp(lg(a1) - lg(a2) - lg(a3))
    return ratio * np.asarray(alpha_vgm, dtype=float)


def D_zero(Se, Ks, m, alpha_g, dtheta):
    """The alpha_g to infinity limit: diffusion off."""
    return torch.zeros_like(Se)


def residual(theta_t, theta_pred, R, ts, tr, Ks, beta, alpha_g, Dz, Dt,
             no_diffusion=False):
    original = PL.gardner_D
    if no_diffusion:
        PL.gardner_D = D_zero
    try:
        return PL.physics_residual_gardner_torch(
            theta_t, theta_pred, R, ts, tr, Ks, beta, alpha_g, Dz, Dt)
    finally:
        PL.gardner_D = original


def load_tile(f, names, stride):
    Se_list, R_list, soil = [], [], []
    for nm in names:
        g = f[nm]
        Se_list.append(np.asarray(g["Se_out"][:], dtype=np.float64))
        R_list.append(np.asarray(g["R_out"][:], dtype=np.float64).ravel())
        soil.append((float(g.attrs["Ks"]),
                     float(g.attrs["n_vgm"]),
                     float(g.attrs["alpha"]),
                     float(g.attrs["theta_s"]),
                     float(g.attrs["theta_r"])))
    Se = np.stack(Se_list)
    Rr = np.stack(R_list)
    nr, nz1, nt = Se.shape

    idx = np.arange(0, nt - 1, stride)
    npair = len(idx)

    Se_t = Se[:, :, idx].transpose(0, 2, 1).reshape(-1, nz1)
    Se_p = Se[:, :, idx + 1].transpose(0, 2, 1).reshape(-1, nz1)
    R_b = Rr[:, idx + 1].reshape(-1, 1)          # R[t+1] drives the step

    soil = np.array(soil)
    rep = lambda col: np.repeat(soil[:, col], npair).reshape(-1, 1)
    Ks_b, n_b, ts_b, tr_b = rep(0), rep(1), rep(3), rep(4)
    dth_b = ts_b - tr_b

    th_t = tr_b + Se_t * dth_b
    th_p = tr_b + Se_p * dth_b
    beta_b = 3.0 + 2.0 / (n_b - 2.0)

    return th_t, th_p, R_b, Ks_b, ts_b, tr_b, beta_b, soil, npair


def log_parabola_refine(grid, curve, j):
    """Refine argmin j below grid resolution. Fit a parabola in log(alpha_g)
    through the three points around j; return refined alpha, value, edge flag.
    At an edge, return the grid point unchanged."""
    if j <= 0 or j >= len(grid) - 1:
        return float(grid[j]), float(curve[j]), True
    x = np.log(grid[j - 1:j + 2])
    y = curve[j - 1:j + 2]
    denom = (y[0] - 2.0 * y[1] + y[2])
    if denom <= 0:
        return float(grid[j]), float(curve[j]), False
    dx = 0.5 * (y[0] - y[2]) / denom * (x[1] - x[0])
    xr = x[1] + np.clip(dx, x[0] - x[1], x[2] - x[1])
    a2 = denom / (2.0 * (x[1] - x[0]) ** 2)
    yr = y[1] - a2 * (x[1] - xr) ** 2
    return float(np.exp(xr)), float(yr), False


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / max(d, 1e-300))


def main():
    ap = argparse.ArgumentParser(description="per-run Gardner alpha_g sweep")
    ap.add_argument("--h5", default=None,
                    help="Path to the dataset h5. Default: data/dataset.h5")
    ap.add_argument("--n_runs", type=int, default=512)
    ap.add_argument("--run_offset", type=int, default=0,
                    help="skip this many runs before taking n_runs")
    ap.add_argument("--se_hi", type=float, default=0.80,
                    help="node Se threshold for the wet-node breakdown")
    ap.add_argument("--tile", type=int, default=32)
    ap.add_argument("--stride", type=int, default=1,
                    help="use every Nth hourly pair. Any even value censors "
                         "every day boundary; leave at 1")
    ap.add_argument("--a_lo", type=float, default=0.003)
    ap.add_argument("--a_hi", type=float, default=30.0)
    ap.add_argument("--n_grid", type=int, default=31)
    ap.add_argument("--csv", type=str, default="alpha_g_perrun.csv")
    ap.add_argument("--Dz", type=float, default=2.5)
    ap.add_argument("--Dt", type=float, default=1.0 / 24.0)
    ap.add_argument("--out", default=None,
                    help="log filename; 'none' to skip; default from offset")
    ap.add_argument("--check", action="store_true",
                    help="out-of-sample frozen-vs-rule comparison instead of "
                         "the sweep report")
    ap.add_argument("--frozen", type=float, default=0.0682,
                    help="frozen candidate alpha_g")
    ap.add_argument("--rule_A", type=float, default=2.736,
                    help="rule prefactor A in A*(n-2)**p*alpha_vgm")
    ap.add_argument("--rule_p", type=float, default=0.711,
                    help="rule exponent p in A*(n-2)**p*alpha_vgm")
    args = ap.parse_args()

    if args.h5 is None:
        args.h5 = paths.require_h5()

    out_name = args.out
    if out_name is None:
        suf = f"_off{args.run_offset}" if args.run_offset else ""
        mode = "_check" if args.check else ""
        out_name = f"fit_alpha_g{mode}{suf}_out.txt"

    tee = None
    if str(out_name).lower() != "none":
        tee = Tee(out_name)
        sys.stdout = tee
    try:
        if args.check:
            run_check(args)
        else:
            run(args)
    finally:
        if tee is not None:
            sys.stdout = tee.stdout
            tee.close()
            print(f"\nlog written: {tee.path}")


def run_check(args):
    """Out-of-sample frozen-vs-rule comparison. Runs the same sweep per tile
    (needed for each run's own optimum, the floor) plus two extra residual
    evaluations per run at the frozen and rule values."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)

    print(BAR)
    print("fit_alpha_g --check  --  out-of-sample frozen vs rule")
    print(BAR)
    print(f"device : {device}")
    if device.type == "cuda":
        print(f"gpu    : {torch.cuda.get_device_name(0)}")
    print(f"frozen : alpha_g = {args.frozen}")
    print(f"rule   : alpha_g = {args.rule_A} * (n - 2)**{args.rule_p}"
          f" * alpha_vgm")
    print("rucker : alpha_g = alpha_vgm * n * G((3n-4)/n)"
          " / [G(1/n) * G((3n-5)/n)]")
    print("         capillary-drive equivalence, published closed form,"
          " nothing fitted")
    print("criterion: adopt the rule iff the stasis")
    print("gate passes at the rule's value on EVERY run AND the rule's median")
    print("excess is at most half the frozen median excess. Otherwise freeze.")
    if args.run_offset == 0:
        print("WARNING: run_offset = 0. These are the FITTING runs; this is")
        print("         not an out-of-sample check. Use a separate offset.")

    grid = np.logspace(np.log10(args.a_lo), np.log10(args.a_hi), args.n_grid)
    nG = len(grid)
    if args.stride != 1:
        print(f"WARNING: stride = {args.stride} censors day boundaries.")

    f = h5py.File(args.h5, "r")
    all_names = sorted([k for k in f.keys() if k.startswith("run_")])
    names = all_names[args.run_offset:args.run_offset + args.n_runs]
    print(f"runs   : {len(names)} of {len(all_names)}   "
          f"offset {args.run_offset}")
    print()

    rows = []
    t0 = time.time()
    for start in range(0, len(names), args.tile):
        chunk = names[start:start + args.tile]
        (th_t, th_p, R_b, Ks_b, ts_b, tr_b,
         beta_b, soil, npair) = load_tile(f, chunk, args.stride)
        nr = len(chunk)

        T = lambda a: torch.tensor(a, device=device)
        th_t_g, th_p_g = T(th_t), T(th_p)
        R_g, Ks_g = T(R_b), T(Ks_b)
        ts_g, tr_g, be_g = T(ts_b), T(tr_b), T(beta_b)

        res_run = np.zeros((nr, nG))
        with torch.no_grad():
            for gi, ag in enumerate(grid):
                r_true = residual(th_t_g, th_p_g, R_g, ts_g, tr_g, Ks_g,
                                  be_g, float(ag), args.Dz, args.Dt)
                res_run[:, gi] = ((r_true ** 2).mean(dim=1)
                                  .reshape(nr, npair).mean(dim=1)
                                  .cpu().numpy())

            # frozen: one shared value
            r_fr = residual(th_t_g, th_p_g, R_g, ts_g, tr_g, Ks_g,
                            be_g, args.frozen, args.Dz, args.Dt)
            res_fr = ((r_fr ** 2).mean(dim=1).reshape(nr, npair)
                      .mean(dim=1).cpu().numpy())

            # rule: per-run value, passed as a (rows, 1) tensor
            n_i = soil[:, 1]
            av_i = soil[:, 2]
            a_rule = args.rule_A * (n_i - 2.0) ** args.rule_p * av_i
            ag_rows = np.repeat(a_rule, npair).reshape(-1, 1)
            ag_g = T(ag_rows)
            r_ru = residual(th_t_g, th_p_g, R_g, ts_g, tr_g, Ks_g,
                            be_g, ag_g, args.Dz, args.Dt)
            res_ru = ((r_ru ** 2).mean(dim=1).reshape(nr, npair)
                      .mean(dim=1).cpu().numpy())
            r_ru_nc = residual(th_t_g, th_t_g, R_g, ts_g, tr_g, Ks_g,
                               be_g, ag_g, args.Dz, args.Dt)
            sta_ru = ((r_ru_nc ** 2).mean(dim=1).reshape(nr, npair)
                      .mean(dim=1).cpu().numpy()) / res_ru

            # rucker: the published closed form, same per-run treatment
            a_ruck = alpha_capillary_drive(n_i, av_i)
            ak_rows = np.repeat(a_ruck, npair).reshape(-1, 1)
            ak_g = T(ak_rows)
            r_rk = residual(th_t_g, th_p_g, R_g, ts_g, tr_g, Ks_g,
                            be_g, ak_g, args.Dz, args.Dt)
            res_rk = ((r_rk ** 2).mean(dim=1).reshape(nr, npair)
                      .mean(dim=1).cpu().numpy())
            r_rk_nc = residual(th_t_g, th_t_g, R_g, ts_g, tr_g, Ks_g,
                               be_g, ak_g, args.Dz, args.Dt)
            sta_rk = ((r_rk_nc ** 2).mean(dim=1).reshape(nr, npair)
                      .mean(dim=1).cpu().numpy()) / res_rk

        for i, nm in enumerate(chunk):
            j = int(np.argmin(res_run[i]))
            a_own, r_own, _ = log_parabola_refine(grid, res_run[i], j)
            rows.append((nm, soil[i, 1], soil[i, 2], a_own, r_own,
                         args.frozen, res_fr[i], float(a_rule[i]),
                         res_ru[i], float(sta_ru[i]),
                         float(a_ruck[i]), res_rk[i], float(sta_rk[i])))

        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  {start + nr:>5} / {len(names)} runs   "
              f"{time.time() - t0:>7.1f} s", flush=True)
    f.close()

    r_own = np.array([r[4] for r in rows])
    r_fr = np.array([r[6] for r in rows])
    r_ru = np.array([r[8] for r in rows])
    sta = np.array([r[9] for r in rows])
    r_rk = np.array([r[11] for r in rows])
    sta_rk_all = np.array([r[12] for r in rows])
    ex_fr = r_fr / r_own - 1.0
    ex_ru = r_ru / r_own - 1.0
    ex_rk = r_rk / r_own - 1.0

    def q(a):
        return np.percentile(a, [10, 50, 90])

    print()
    print("EXCESS over each run's own optimum, res(x)/res(own) - 1")
    print(f"{'policy':>10}{'p10':>12}{'median':>12}{'p90':>12}{'max':>12}")
    for nmn, ex in (("frozen", ex_fr), ("rule", ex_ru), ("rucker", ex_rk)):
        p = q(ex)
        print(f"{nmn:>10}{p[0]:>12.4f}{p[1]:>12.4f}{p[2]:>12.4f}"
              f"{ex.max():>12.4f}")

    nveto = int(np.sum(sta <= 1.0))
    nveto_rk = int(np.sum(sta_rk_all <= 1.0))
    print()
    print(f"stasis gate at the rule's value  : "
          f"{'all pass' if nveto == 0 else f'{nveto} runs VETO'}"
          f"   worst margin {sta.min():.4f}")
    print(f"stasis gate at rucker's value    : "
          f"{'all pass' if nveto_rk == 0 else f'{nveto_rk} runs VETO'}"
          f"   worst margin {sta_rk_all.min():.4f}")

    med_fr = float(np.median(ex_fr))
    med_ru = float(np.median(ex_ru))
    med_rk = float(np.median(ex_rk))
    gate_ok = nveto == 0
    halved = med_ru <= 0.5 * med_fr
    print()
    print("CRITERION as declared for the fitted rule")
    print(f"  gate all pass          : {gate_ok}")
    print(f"  median excess, frozen  : {med_fr:.4f}")
    print(f"  median excess, rule    : {med_ru:.4f}"
          f"   (needs <= {0.5 * med_fr:.4f})")
    print(f"  VERDICT: {'ADOPT THE RULE' if (gate_ok and halved) else 'FREEZE'}")

    # The published closed form, scored on the same held-out runs. This asks a
    # question the original criterion did not: whether an established
    # equivalence beats a fit made here. Criterion for it: prefer the
    # published form if its median excess is at or below the
    # fitted rule's, since a published closed form with equal performance is
    # the stronger position. Otherwise the fitted rule stands and the published
    # form is reported as the corroborating comparison it already was.
    print()
    print("PUBLISHED CLOSED FORM, same held-out runs")
    print(f"  gate all pass          : {nveto_rk == 0}")
    print(f"  median excess, rucker  : {med_rk:.4f}")
    print(f"  median excess, rule    : {med_ru:.4f}")
    if nveto_rk == 0 and med_rk <= med_ru:
        pref = "PREFER THE PUBLISHED FORM"
    elif nveto_rk != 0:
        pref = "published form VETOED by the stasis gate; fitted rule stands"
    else:
        pref = (f"fitted rule stands; published form costs "
                f"{med_rk / max(med_ru, 1e-30):.2f}x its median excess")
    print(f"  VERDICT: {pref}")

    with open("alpha_g_check.csv", "w") as fh:
        fh.write("run,n,alpha_vgm,a_own,res_own,a_frozen,res_frozen,"
                 "a_rule,res_rule,stasis_rule,"
                 "a_rucker,res_rucker,stasis_rucker\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")
    print()
    print("per-run comparison written to alpha_g_check.csv")
    print(BAR)
    print("done")


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)

    print(BAR)
    print("fit_alpha_g  --  per-run sweep of the Gardner alpha_g")
    print(BAR)
    print(f"device : {device}")
    if device.type == "cuda":
        print(f"gpu    : {torch.cuda.get_device_name(0)}")

    grid = np.logspace(np.log10(args.a_lo), np.log10(args.a_hi), args.n_grid)
    nG = len(grid)
    print(f"grid   : {nG} points, {args.a_lo:g} to {args.a_hi:g}, log spaced")
    if args.stride != 1:
        print(f"WARNING: stride = {args.stride}. Any even stride samples NO")
        print("         day boundary (t = 23 + 24k is always odd). This exact")
        print("         censoring voided two earlier sweeps.")
    print(f"stride : {args.stride}")

    f = h5py.File(args.h5, "r")
    all_names = sorted([k for k in f.keys() if k.startswith("run_")])
    names = all_names[args.run_offset:args.run_offset + args.n_runs]
    print(f"runs   : {len(names)} of {len(all_names)}   "
          f"offset {args.run_offset}")
    print(f"wet    : node breakdown at Se > {args.se_hi}")
    print()

    best_rows = []
    curve_res = np.zeros(nG + 1)          # last slot is the D = 0 anchor
    curve_sta = np.zeros(nG + 1)
    wet_sum = np.zeros(nG + 1)
    wet_n = 0.0
    n_done = 0

    t0 = time.time()
    for start in range(0, len(names), args.tile):
        chunk = names[start:start + args.tile]
        (th_t, th_p, R_b, Ks_b, ts_b, tr_b,
         beta_b, soil, npair) = load_tile(f, chunk, args.stride)
        nr = len(chunk)

        T = lambda a: torch.tensor(a, device=device)
        th_t_g, th_p_g = T(th_t), T(th_p)
        R_g, Ks_g = T(R_b), T(Ks_b)
        ts_g, tr_g, be_g = T(ts_b), T(tr_b), T(beta_b)

        Se_t_g = (th_t_g - tr_g) / (ts_g - tr_g)
        wet = (Se_t_g > args.se_hi)
        wet_n += float(wet.sum().item())

        res_run = np.zeros((nr, nG + 1))
        sta_run = np.zeros((nr, nG + 1))

        with torch.no_grad():
            for gi, ag in enumerate(list(grid) + [None]):
                nd = ag is None
                agv = 1.0 if nd else float(ag)
                r_true = residual(th_t_g, th_p_g, R_g, ts_g, tr_g, Ks_g,
                                  be_g, agv, args.Dz, args.Dt,
                                  no_diffusion=nd)
                r_nc = residual(th_t_g, th_t_g, R_g, ts_g, tr_g, Ks_g,
                                be_g, agv, args.Dz, args.Dt,
                                no_diffusion=nd)
                st = (r_true ** 2).mean(dim=1).reshape(nr, npair).mean(dim=1)
                sn = (r_nc ** 2).mean(dim=1).reshape(nr, npair).mean(dim=1)
                res_run[:, gi] = st.cpu().numpy()
                sta_run[:, gi] = (sn / st).cpu().numpy()
                wet_sum[gi] += float(((r_true ** 2) * wet).sum().item())

        curve_res += res_run.sum(axis=0)
        curve_sta += sta_run.sum(axis=0)
        n_done += nr

        for i, nm in enumerate(chunk):
            j = int(np.argmin(res_run[i, :nG]))
            a_ref, r_ref, at_edge = log_parabola_refine(grid,
                                                        res_run[i, :nG], j)
            margin_at_opt = float(sta_run[i, j])
            Ks_i, n_i, av_i = soil[i, 0], soil[i, 1], soil[i, 2]
            dth_i = soil[i, 3] - soil[i, 4]
            beta_i = 3.0 + 2.0 / (n_i - 2.0)
            a_pred = beta_i * av_i               # candidate closed form
            best_rows.append((nm, Ks_i, n_i, av_i, dth_i, beta_i,
                              a_ref, r_ref, int(at_edge), margin_at_opt,
                              a_pred, a_ref / a_pred,
                              res_run[i, -1], sta_run[i, -1]))

        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  {start + nr:>5} / {len(names)} runs   "
              f"{time.time() - t0:>7.1f} s", flush=True)
    f.close()

    curve_res /= n_done
    curve_sta /= n_done
    wet_curve = wet_sum / max(wet_n, 1.0)

    # ------------------------------------------------------------ aggregate
    print()
    print("AGGREGATE over all runs")
    print(f"{'alpha_g':>12}{'mean sq res':>18}{'stasis margin':>18}")
    print("-" * 48)
    for gi, ag in enumerate(grid):
        print(f"{ag:>12.4g}{curve_res[gi]:>18.6e}{curve_sta[gi]:>18.4f}")
    print(f"{'D = 0':>12}{curve_res[-1]:>18.6e}{curve_sta[-1]:>18.4f}")

    j = int(np.argmin(curve_res[:nG]))
    a_agg, r_agg, _ = log_parabola_refine(grid, curve_res[:nG], j)
    print()
    print(f"aggregate argmin on the grid : alpha_g = {grid[j]:.4g}")
    print(f"aggregate refined optimum    : alpha_g = {a_agg:.4g}"
          f"   res = {r_agg:.6e}")
    print(f"stasis margin at the optimum : {curve_sta[j]:.4f}"
          f"   {'gate PASS' if curve_sta[j] > 1 else 'gate VETO'}")
    print("  (stasis is a gate, not a score; it selects nothing above 1)")

    # ------------------------------------------------------------ wet nodes
    print()
    print(f"WET-NODE BREAKDOWN  (Se > {args.se_hi} at start of step)")
    print(f"  wet node-samples : {int(wet_n)}")
    if wet_n > 0:
        jw = int(np.argmin(wet_curve[:nG]))
        gain_all = 100.0 * (curve_res[-1] - curve_res[j]) / curve_res[j]
        gain_wet = 100.0 * (wet_curve[-1] - wet_curve[jw]) / wet_curve[jw]
        print(f"{'':>14}{'best D':>18}{'D = 0':>18}{'D worth':>12}")
        print("-" * 62)
        print(f"{'all nodes':>14}{curve_res[j]:>18.6e}"
              f"{curve_res[-1]:>18.6e}{gain_all:>11.2f}%")
        print(f"{'wet nodes':>14}{wet_curve[jw]:>18.6e}"
              f"{wet_curve[-1]:>18.6e}{gain_wet:>11.2f}%")
        print(f"  wet-node argmin : alpha_g = {grid[jw]:.4g}")

    # ------------------------------------------------------------ per-run
    arr = np.array([r[6] for r in best_rows])          # refined optima
    edge = np.array([r[8] for r in best_rows])
    marg = np.array([r[9] for r in best_rows])
    print()
    print("PER-RUN OPTIMA  (log-parabola refined)")
    q = np.percentile(arr, [10, 25, 50, 75, 90])
    print(f"  median {q[2]:.4g}   IQR [{q[1]:.4g}, {q[3]:.4g}]   "
          f"10-90 [{q[0]:.4g}, {q[4]:.4g}]")
    print(f"  min {arr.min():.4g}   max {arr.max():.4g}   "
          f"on grid edge {int(edge.sum())} of {len(arr)}")
    print(f"  spread, log10 IQR = {np.log10(q[3] / q[1]):.3f} decades")
    ngate = int(np.sum(marg <= 1.0))
    print(f"  stasis gate at the per-run optimum: "
          f"{'all pass' if ngate == 0 else f'{ngate} runs VETO'}"
          f"   worst margin {marg.min():.4f}")

    # ------------------------------------------------------------ soil dep
    Ks_a = np.array([r[1] for r in best_rows])
    n_a = np.array([r[2] for r in best_rows])
    av_a = np.array([r[3] for r in best_rows])
    dt_a = np.array([r[4] for r in best_rows])
    be_a = np.array([r[5] for r in best_rows])
    print()
    print("SOIL DEPENDENCE of the per-run optimum (Spearman rank)")
    for nm, v in (("Ks", Ks_a), ("n", n_a), ("alpha_vgm", av_a),
                  ("dtheta", dt_a), ("beta", be_a)):
        print(f"  {nm:<12} vs alpha_g*  : {spearman(v, arr):+.4f}")

    # -------------------------------------------------------- candidate form
    pred = np.array([r[10] for r in best_rows])
    ratio = np.array([r[11] for r in best_rows])
    qr = np.percentile(ratio, [10, 25, 50, 75, 90])
    lx, ly = np.log(pred), np.log(arr)
    lx0, ly0 = lx - lx.mean(), ly - ly.mean()
    slope = float((lx0 * ly0).sum() / max((lx0 ** 2).sum(), 1e-300))
    r_pear = float((lx0 * ly0).sum() /
                   max(np.sqrt((lx0 ** 2).sum() * (ly0 ** 2).sum()), 1e-300))
    print()
    print("CANDIDATE FORM  alpha_g = delta(n) * alpha_vgm")
    print(f"  ratio alpha_g* / prediction:")
    print(f"    median {qr[2]:.4g}   IQR [{qr[1]:.4g}, {qr[3]:.4g}]   "
          f"10-90 [{qr[0]:.4g}, {qr[4]:.4g}]")
    print(f"  Spearman alpha_g* vs prediction : {spearman(pred, arr):+.4f}")
    print(f"  log-log slope (predicts 1)      : {slope:+.4f}"
          f"   pearson r in log space {r_pear:+.4f}")
    print("  reading: ratio near 1 with slope near 1 supports the rule;")
    print("  a stable ratio far from 1 with slope near 1 supports a rescaled")
    print("  rule; slope near 0 kills soil-tracking through this group.")

    # ------------------------------------------------------------ CSV
    with open(args.csv, "w") as fh:
        fh.write("run,Ks,n,alpha_vgm,dtheta,beta,alpha_g_opt,res_at_opt,"
                 "at_edge,stasis_at_opt,alpha_pred,ratio,res_D0,sta_D0\n")
        for r in best_rows:
            fh.write(",".join(str(x) for x in r) + "\n")
    print()
    print(f"per-run optima written to {args.csv}")
    print(BAR)
    print("done")


if __name__ == "__main__":
    main()
