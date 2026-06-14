"""
Balmer_transient_regime_heatmap.py
==================================

Regime-map sweep for the corrected v3 Balmer transient analysis.

Purpose
-------
Replace sparse 3-point robustness plots with real regime maps.

The script imports the corrected local module:

    src/rates/Balmer_transient_ratio.py

and reuses its v3 physics:

    - Hoang-Binh radiative A-values loaded from radiative_rates.csv
    - only type == "res_to_res" rows
    - resolved Halpha/Hbeta transitions only
    - time-dependent CR propagation after a Te step

Default sweep
-------------
    initial Te grid values between 1.5 and 6.0 eV
    all available ne grid values
    requested DeltaTe = +0.6 eV
    t_max = 1e-3 s
    n_time = 500

Outputs
-------
CSV:
    data/processed/sensitivity/regime/Balmer_transient_regime_heatmap_DTe_p0p60.csv

Figures:
    figures/regime/Balmer_regime_peak_ratio_error_DTe_p0p60.png/.pdf
    figures/regime/Balmer_regime_last_gt10_DTe_p0p60.png/.pdf
    figures/regime/Balmer_regime_tau_slow_DTe_p0p60.png/.pdf
    figures/regime/Balmer_regime_M_DTe_p0p60.png/.pdf

Run from repo rates directory:
    cd /Users/phi/Desktop/non_markovian_cr/src/rates
    python Balmer_transient_regime_heatmap.py

Smoke test:
    python Balmer_transient_regime_heatmap.py --max-cases 8 --n-time 200 --t-max 1e-4

Full recommended run:
    python Balmer_transient_regime_heatmap.py --delta 0.6 --te-min 1.5 --te-max 6.0 --n-time 500 --t-max 1e-3
"""

from __future__ import annotations

import argparse
import math
import pathlib
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import Balmer_transient_ratio as btr


HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT_DIR = REPO / "data/processed/sensitivity/regime"
FIG_DIR = REPO / "figures/regime"


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def peak_abs(times: np.ndarray, err: np.ndarray) -> Tuple[float, float, float]:
    err = np.asarray(err, dtype=float)
    times = np.asarray(times, dtype=float)
    valid = np.isfinite(err)
    if not np.any(valid):
        return np.nan, np.nan, np.nan
    idx_valid = np.where(valid)[0]
    idx = idx_valid[int(np.nanargmax(np.abs(err[valid])))]
    return float(err[idx]), float(abs(err[idx])), float(times[idx])


def last_time_abs_above(times: np.ndarray, err: np.ndarray, threshold: float) -> Tuple[float, bool]:
    """
    Returns (last_time, censored_at_tmax).

    If the final time sample is still above the threshold, the result is censored:
    the true crossing time is later than t_max.
    """
    err = np.asarray(err, dtype=float)
    times = np.asarray(times, dtype=float)
    valid = np.isfinite(err)
    above = valid & (np.abs(err) >= threshold)
    if not np.any(above):
        return 0.0, False
    last_idx = int(np.where(above)[0][-1])
    censored = bool(last_idx == len(times) - 1)
    return float(times[last_idx]), censored


def timescale_metrics(eig_times_s: np.ndarray) -> Tuple[float, float, float]:
    eig_times_s = np.asarray(eig_times_s, dtype=float)
    eig_times_s = eig_times_s[np.isfinite(eig_times_s) & (eig_times_s > 0)]
    eig_times_s = np.sort(eig_times_s)[::-1]
    if eig_times_s.size == 0:
        return np.nan, np.nan, np.nan
    tau_slow = float(eig_times_s[0])
    tau_fast = float(eig_times_s[1]) if eig_times_s.size > 1 else np.nan
    M = tau_slow / tau_fast if np.isfinite(tau_fast) and tau_fast > 0 else np.nan
    return tau_slow, tau_fast, M


def summarize(res: btr.TransientResult, te_requested: float, ne_requested: float, delta_requested: float) -> Dict[str, float | int | bool]:
    peak_R_signed, peak_R_abs, t_peak_R = peak_abs(res.times, res.err_ratio)
    peak_Ha_signed, peak_Ha_abs, t_peak_Ha = peak_abs(res.times, res.err_Halpha)
    peak_Hb_signed, peak_Hb_abs, t_peak_Hb = peak_abs(res.times, res.err_Hbeta)

    last_1, cens_1 = last_time_abs_above(res.times, res.err_ratio, 0.01)
    last_5, cens_5 = last_time_abs_above(res.times, res.err_ratio, 0.05)
    last_10, cens_10 = last_time_abs_above(res.times, res.err_ratio, 0.10)

    tau_slow, tau_fast, M = timescale_metrics(res.eig_times_s)

    return {
        "Te_requested_eV": float(te_requested),
        "ne_requested_cm-3": float(ne_requested),
        "DeltaTe_requested_eV": float(delta_requested),
        "Te_old_grid_eV": float(res.te_old),
        "Te_new_grid_eV": float(res.te_new),
        "DeltaTe_actual_eV": float(res.delta_actual),
        "ne_grid_cm-3": float(res.ne),
        "ti_old": int(res.ti_old),
        "ti_new": int(res.ti_new),
        "ni": int(res.ni),
        "ratio_QSS_old": float(res.ratio_qss_old),
        "ratio_QSS_new": float(res.ratio_qss_new),
        "initial_ratio_error_percent_signed": 100.0 * float(res.err_ratio[0]),
        "peak_ratio_error_percent_abs": 100.0 * peak_R_abs,
        "peak_ratio_error_percent_signed": 100.0 * peak_R_signed,
        "t_peak_ratio_ns": 1e9 * t_peak_R,
        "peak_Halpha_error_percent_abs": 100.0 * peak_Ha_abs,
        "peak_Hbeta_error_percent_abs": 100.0 * peak_Hb_abs,
        "last_t_ratio_abs_error_gt_1pct_us": 1e6 * last_1,
        "last_t_ratio_abs_error_gt_5pct_us": 1e6 * last_5,
        "last_t_ratio_abs_error_gt_10pct_us": 1e6 * last_10,
        "censored_gt_1pct": cens_1,
        "censored_gt_5pct": cens_5,
        "censored_gt_10pct": cens_10,
        "tau_slow_us": 1e6 * tau_slow,
        "tau_fast_ns": 1e9 * tau_fast,
        "M_tau_slow_over_tau_fast": M,
    }


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def setup_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.grid": False,
    })


def slug_delta(delta: float) -> str:
    return (f"{delta:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p"))


def cell_edges(x: np.ndarray, log: bool = False) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 1:
        return np.array([x[0] * 0.8, x[0] * 1.2])
    if log:
        lx = np.log10(x)
        mids = 0.5 * (lx[:-1] + lx[1:])
        first = lx[0] - (mids[0] - lx[0])
        last = lx[-1] + (lx[-1] - mids[-1])
        return 10 ** np.concatenate([[first], mids, [last]])
    mids = 0.5 * (x[:-1] + x[1:])
    first = x[0] - (mids[0] - x[0])
    last = x[-1] + (x[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def pivot_grid(df: pd.DataFrame, value_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Use actual grid coordinates.
    te_vals = np.array(sorted(df["Te_old_grid_eV"].unique()), dtype=float)
    ne_vals = np.array(sorted(df["ne_grid_cm-3"].unique()), dtype=float)
    Z = np.full((te_vals.size, ne_vals.size), np.nan, dtype=float)
    for i, te in enumerate(te_vals):
        for j, ne in enumerate(ne_vals):
            sub = df[(np.isclose(df["Te_old_grid_eV"], te)) & (np.isclose(df["ne_grid_cm-3"], ne))]
            if not sub.empty:
                Z[i, j] = float(sub.iloc[0][value_col])
    return te_vals, ne_vals, Z


def plot_heatmap(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    cbar_label: str,
    out_stem: str,
    log_color: bool = False,
    cmap: str = "viridis",
    overlay_benchmark: bool = True,
) -> None:
    setup_plot_style()
    te_vals, ne_vals, Z = pivot_grid(df, value_col)
    ne_edges = cell_edges(ne_vals, log=True)
    te_edges = cell_edges(te_vals, log=False)

    fig, ax = plt.subplots(figsize=(7.4, 5.4))

    plot_kwargs = dict(cmap=cmap, shading="auto")
    if log_color:
        positive = Z[np.isfinite(Z) & (Z > 0)]
        if positive.size:
            vmin = max(np.nanmin(positive), 1e-6)
            vmax = np.nanmax(positive)
            plot_kwargs["norm"] = LogNorm(vmin=vmin, vmax=vmax)

    im = ax.pcolormesh(ne_edges, te_edges, Z, **plot_kwargs)
    ax.set_xscale("log")
    ax.set_xlabel(r"$n_e$ grid value [cm$^{-3}$]")
    ax.set_ylabel(r"initial $T_e$ grid value [eV]")
    ax.set_title(title)

    if overlay_benchmark:
        # nearest grid point to nominal 3 eV, 1e14 cm^-3 usually used in thesis analysis
        te_b = float(btr.TE_GRID[int(np.argmin(np.abs(btr.TE_GRID - 3.0)))])
        ne_b = float(btr.NE_GRID[int(np.argmin(np.abs(btr.NE_GRID - 1e14)))])
        ax.plot(ne_b, te_b, marker="*", ms=14, color="cyan", mec="black", mew=0.4, label="benchmark grid point")
        ax.legend(fontsize=8, loc="best")

    cb = fig.colorbar(im, ax=ax)
    cb.set_label(cbar_label)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"{out_stem}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_all_plots(df: pd.DataFrame, delta_slug: str) -> None:
    plot_heatmap(
        df,
        value_col="peak_ratio_error_percent_abs",
        title=r"Peak transient $H\alpha/H\beta$ ratio error, $\Delta T_e\approx0.6$ eV",
        cbar_label=r"peak $|H\alpha/H\beta$ error| [%]",
        out_stem=f"Balmer_regime_peak_ratio_error_DTe_{delta_slug}",
        log_color=False,
        cmap="magma",
    )
    plot_heatmap(
        df,
        value_col="last_t_ratio_abs_error_gt_10pct_us",
        title=r"Persistence of $H\alpha/H\beta$ bias above 10%, $\Delta T_e\approx0.6$ eV",
        cbar_label=r"last time $|$ratio error$|>10\%$ [$\mu$s]",
        out_stem=f"Balmer_regime_last_gt10_DTe_{delta_slug}",
        log_color=True,
        cmap="viridis",
    )
    plot_heatmap(
        df,
        value_col="tau_slow_us",
        title=r"Slow CR decay time after step, $\Delta T_e\approx0.6$ eV",
        cbar_label=r"$\tau_{slow}$ [$\mu$s]",
        out_stem=f"Balmer_regime_tau_slow_DTe_{delta_slug}",
        log_color=True,
        cmap="plasma",
    )
    plot_heatmap(
        df,
        value_col="M_tau_slow_over_tau_fast",
        title=r"Timescale separation $M=\tau_{slow}/\tau_{fast}$, $\Delta T_e\approx0.6$ eV",
        cbar_label=r"$M$",
        out_stem=f"Balmer_regime_M_DTe_{delta_slug}",
        log_color=True,
        cmap="cividis",
    )


# -----------------------------------------------------------------------------
# Sweep logic
# -----------------------------------------------------------------------------

def build_cases(te_min: float, te_max: float, delta: float) -> List[Tuple[float, float, float]]:
    cases: List[Tuple[float, float, float]] = []
    for te in btr.TE_GRID:
        te = float(te)
        if te < te_min or te > te_max:
            continue
        # Skip if post-step target would be outside grid.
        if te + delta > float(np.max(btr.TE_GRID)):
            continue
        for ne in btr.NE_GRID:
            cases.append((te, float(ne), float(delta)))
    return cases


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regime heatmap for transient Balmer Halpha/Hbeta ratio bias.")
    p.add_argument("--delta", type=float, default=0.6, help="Requested DeltaTe [eV].")
    p.add_argument("--te-min", type=float, default=1.5, help="Minimum initial Te grid value [eV].")
    p.add_argument("--te-max", type=float, default=6.0, help="Maximum initial Te grid value [eV].")
    p.add_argument("--n-time", type=int, default=500, help="Number of time samples per transient.")
    p.add_argument("--t-min", type=float, default=None, help="Minimum positive time [s]. Default auto from eigenvalues.")
    p.add_argument("--t-max", type=float, default=1e-3, help="Maximum time [s].")
    p.add_argument("--n-ion", type=float, default=btr.N_ION_DEFAULT, help="Ion density/source scaling.")
    p.add_argument("--use-photon-energy", action="store_true", help="Use energy emissivities rather than photon proxies.")
    p.add_argument("--max-cases", type=int, default=None, help="For smoke testing: run only first N cases.")
    p.add_argument("--resume", action="store_true", help="Resume from existing output CSV and skip completed ti_old/ni cases.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    delta_slug = slug_delta(args.delta)
    out_csv = OUT_DIR / f"Balmer_transient_regime_heatmap_DTe_{delta_slug}.csv"
    out_tex = OUT_DIR / f"Balmer_transient_regime_heatmap_DTe_{delta_slug}.tex"

    print("Loading corrected v3 CR transient machinery...")
    print(f"Using driver module: {btr.__file__}")
    L_grid, S_grid = btr.load_cr_grids()
    line_weights = btr.load_radiative_weights(use_photon_energy=args.use_photon_energy)
    btr.print_weights(line_weights, use_photon_energy=args.use_photon_energy)

    cases = build_cases(args.te_min, args.te_max, args.delta)
    if args.max_cases is not None:
        cases = cases[: int(args.max_cases)]

    existing_rows: List[Dict] = []
    done_keys = set()
    if args.resume and out_csv.exists():
        prev = pd.read_csv(out_csv)
        existing_rows = prev.to_dict("records")
        for _, r in prev.iterrows():
            done_keys.add((int(r["ti_old"]), int(r["ni"]), int(r["ti_new"])))
        print(f"Resume mode: loaded {len(existing_rows)} previous rows from {out_csv}")

    rows = list(existing_rows)

    print(f"Running {len(cases)} grid cases for requested ΔTe={args.delta:g} eV...")
    print(f"Initial Te range: {args.te_min:g} to {args.te_max:g} eV")
    print(f"Densities: all {len(btr.NE_GRID)} ne grid values")
    print(f"t_max={args.t_max:g} s, n_time={args.n_time}")

    for k, (te_req, ne_req, dte_req) in enumerate(cases, start=1):
        # Determine key quickly using actual nearest indices via btr helper.
        ti_old, ti_new, _, _, _ = btr.nearest_step_indices(te_req, dte_req)
        ni = btr.nearest_index(btr.NE_GRID, ne_req)
        key = (int(ti_old), int(ni), int(ti_new))
        if key in done_keys:
            print(f"[{k:03d}/{len(cases):03d}] skip completed Te={te_req:.4g}, ne={ne_req:.3e}")
            continue

        print(f"[{k:03d}/{len(cases):03d}] Te={te_req:.4g} eV, ne={ne_req:.3e}, ΔTe={dte_req:g} eV")
        res = btr.run_single_transient(
            L_grid=L_grid,
            S_grid=S_grid,
            line_weights=line_weights,
            te_old_requested=te_req,
            ne_requested=ne_req,
            delta_nominal=dte_req,
            n_ion=float(args.n_ion),
            t_min=args.t_min,
            t_max=args.t_max,
            n_time=int(args.n_time),
        )
        row = summarize(res, te_req, ne_req, dte_req)
        rows.append(row)
        done_keys.add(key)

        # Checkpoint after every case.
        pd.DataFrame(rows).sort_values(["Te_old_grid_eV", "ne_grid_cm-3"]).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows).sort_values(["Te_old_grid_eV", "ne_grid_cm-3"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)

    table_cols = [
        "Te_old_grid_eV", "Te_new_grid_eV", "DeltaTe_actual_eV", "ne_grid_cm-3",
        "peak_ratio_error_percent_abs", "t_peak_ratio_ns",
        "last_t_ratio_abs_error_gt_10pct_us", "censored_gt_10pct",
        "tau_fast_ns", "tau_slow_us", "M_tau_slow_over_tau_fast",
    ]
    df[table_cols].to_latex(
        out_tex,
        index=False,
        float_format=lambda x: f"{x:.3g}",
        escape=False,
        caption=(
            r"Regime heatmap data for transient $H\alpha/H\beta$ ratio bias at fixed requested "
            rf"$\Delta T_e={args.delta:g}$ eV. Censored entries remain above threshold at $t_{{max}}={args.t_max:g}$ s."
        ),
        label="tab:balmer_regime_heatmap",
    )

    make_all_plots(df, delta_slug)

    print()
    print("=" * 100)
    print("Regime heatmap summary")
    print("=" * 100)
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(df[table_cols].to_string(index=False))

    print()
    n_censored = int(df["censored_gt_10pct"].sum()) if "censored_gt_10pct" in df else 0
    print(f"Censored >10% rows at t_max: {n_censored}/{len(df)}")
    print("Saved:")
    print(f"  {out_csv.relative_to(REPO)}")
    print(f"  {out_tex.relative_to(REPO)}")
    print(f"  {FIG_DIR.relative_to(REPO)}/Balmer_regime_*.png/.pdf")
    print()
    print("Interpretation rule:")
    print("  Peak-error heatmap = where diagnostic bias becomes large.")
    print("  Last-time >10% heatmap = where the bias persists.")
    print("  Censored rows mean the persistence is at least t_max, not exactly t_max.")


if __name__ == "__main__":
    main()
