"""
Balmer_transient_robustness_sweep.py
====================================

Multi-point robustness sweep for the corrected Balmer transient analysis.

This script is intentionally a DRIVER around the corrected v3 physics in
Balmer_transient_ratio.py. It does not reimplement radiative loading or the
CR propagator. It imports and uses:

    run_single_transient(...)
    load_cr_grids(...)
    load_radiative_weights(...)

Therefore, before running this script, make sure src/rates/Balmer_transient_ratio.py
is the corrected v3 version that loads radiative A-values using:

    type == "res_to_res"
    n_upper,l_upper,n_lower,l_lower matching the transition

Default sweep
-------------
1. Temperature sweep at reference density:
       ne_requested = 1.39e14 cm^-3
       Te_old_requested = 2, 3, 5 eV
       DeltaTe_requested = 0.3, 0.6, 1.0 eV

2. Density sweep at reference temperature:
       Te_old_requested = 3 eV
       ne_requested = 1e13, 1e14, 1e15 cm^-3
       DeltaTe_requested = 0.6 eV

Outputs
-------
data/processed/sensitivity/robustness/Balmer_transient_robustness_summary.csv
data/processed/sensitivity/robustness/Balmer_transient_robustness_summary.tex

figures/robustness/Balmer_robustness_peak_ratio_error_vs_Te.png/.pdf
figures/robustness/Balmer_robustness_duration_vs_Te.png/.pdf
figures/robustness/Balmer_robustness_density_sweep.png/.pdf
figures/robustness/Balmer_robustness_peak_ratio_scatter.png/.pdf

Run
---
cd /Users/phi/Desktop/non_markovian_cr/src/rates
python Balmer_transient_robustness_sweep.py

Optional faster/smaller run:
python Balmer_transient_robustness_sweep.py --n-time 250 --t-max 1e-4

Optional custom grids:
python Balmer_transient_robustness_sweep.py \
  --te-sweep 2 3 5 \
  --ne-sweep 1e13 1e14 1e15 \
  --deltas-temp 0.3 0.6 1.0 \
  --delta-density 0.6
"""

from __future__ import annotations

import argparse
import pathlib
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import corrected v3 physics from the local script.
import Balmer_transient_ratio as btr


HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT_DIR = REPO / "data/processed/sensitivity/robustness"
FIG_DIR = REPO / "figures/robustness"


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def _peak_abs(times: np.ndarray, err: np.ndarray) -> Tuple[float, float, float]:
    """
    Return signed peak error, absolute peak error, and t_peak.
    Peak is selected by maximum absolute error.
    """
    err = np.asarray(err, dtype=float)
    times = np.asarray(times, dtype=float)
    valid = np.isfinite(err)
    if not np.any(valid):
        return np.nan, np.nan, np.nan
    idx_valid = np.where(valid)[0]
    idx = idx_valid[int(np.nanargmax(np.abs(err[valid])))]
    return float(err[idx]), float(abs(err[idx])), float(times[idx])


def _last_time_abs_above(times: np.ndarray, err: np.ndarray, threshold: float) -> float:
    err = np.asarray(err, dtype=float)
    times = np.asarray(times, dtype=float)
    valid = np.isfinite(err)
    above = valid & (np.abs(err) >= threshold)
    if not np.any(above):
        return 0.0
    return float(times[np.where(above)[0][-1]])


def _timescale_metrics(eig_times_s: np.ndarray) -> Tuple[float, float, float]:
    """
    eig_times_s is sorted descending by corrected v3 script.

    Definition used here:
      tau_slow = largest decay time = eig_times_s[0]
      tau_fast = next largest decay time = eig_times_s[1]
      M = tau_slow / tau_fast

    This matches the observed current runs where the first mode is the slow
    ionisation/source-balance relaxation and the second mode is the fast
    excited-state relaxation relevant to differential Balmer response.
    """
    eig_times_s = np.asarray(eig_times_s, dtype=float)
    eig_times_s = eig_times_s[np.isfinite(eig_times_s) & (eig_times_s > 0)]
    eig_times_s = np.sort(eig_times_s)[::-1]
    if eig_times_s.size == 0:
        return np.nan, np.nan, np.nan
    tau_slow = float(eig_times_s[0])
    tau_fast = float(eig_times_s[1]) if eig_times_s.size > 1 else np.nan
    M = tau_slow / tau_fast if np.isfinite(tau_fast) and tau_fast > 0 else np.nan
    return tau_slow, tau_fast, M


def summarize_run(res: btr.TransientResult, sweep_type: str) -> Dict[str, float | str]:
    peak_Ha_signed, peak_Ha_abs, t_peak_Ha = _peak_abs(res.times, res.err_Halpha)
    peak_Hb_signed, peak_Hb_abs, t_peak_Hb = _peak_abs(res.times, res.err_Hbeta)
    peak_R_signed, peak_R_abs, t_peak_R = _peak_abs(res.times, res.err_ratio)

    tau_slow, tau_fast, M = _timescale_metrics(res.eig_times_s)

    return {
        "sweep_type": sweep_type,
        "Te_old_requested_eV": np.nan,  # filled by caller
        "ne_requested_cm-3": np.nan,    # filled by caller
        "DeltaTe_requested_eV": res.delta_nominal,
        "Te_old_grid_eV": res.te_old,
        "Te_new_grid_eV": res.te_new,
        "DeltaTe_actual_eV": res.delta_actual,
        "ne_grid_cm-3": res.ne,
        "ti_old": res.ti_old,
        "ti_new": res.ti_new,
        "ni": res.ni,
        "ratio_QSS_old": res.ratio_qss_old,
        "ratio_QSS_new": res.ratio_qss_new,
        "initial_ratio_error_percent": 100.0 * float(res.err_ratio[0]),
        "peak_Halpha_error_percent_abs": 100.0 * peak_Ha_abs,
        "peak_Hbeta_error_percent_abs": 100.0 * peak_Hb_abs,
        "peak_ratio_error_percent_abs": 100.0 * peak_R_abs,
        "peak_ratio_error_percent_signed": 100.0 * peak_R_signed,
        "t_peak_ratio_ns": 1e9 * t_peak_R,
        "last_t_ratio_abs_error_gt_1pct_us": 1e6 * _last_time_abs_above(res.times, res.err_ratio, 0.01),
        "last_t_ratio_abs_error_gt_5pct_us": 1e6 * _last_time_abs_above(res.times, res.err_ratio, 0.05),
        "last_t_ratio_abs_error_gt_10pct_us": 1e6 * _last_time_abs_above(res.times, res.err_ratio, 0.10),
        "last_t_Halpha_abs_error_gt_10pct_us": 1e6 * _last_time_abs_above(res.times, res.err_Halpha, 0.10),
        "last_t_Hbeta_abs_error_gt_10pct_us": 1e6 * _last_time_abs_above(res.times, res.err_Hbeta, 0.10),
        "tau_slow_us": 1e6 * tau_slow,
        "tau_fast_ns": 1e9 * tau_fast,
        "M_tau_slow_over_tau_fast": M,
    }


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def setup_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.28,
    })


def plot_temperature_sweep(df: pd.DataFrame) -> None:
    setup_plot_style()
    temp = df[df["sweep_type"] == "temperature_sweep"].copy()
    if temp.empty:
        return

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for dte, sub in temp.groupby("DeltaTe_requested_eV"):
        sub = sub.sort_values("Te_old_grid_eV")
        ax.plot(
            sub["Te_old_grid_eV"],
            sub["peak_ratio_error_percent_abs"],
            marker="o",
            lw=2,
            label=f"requested ΔTe={dte:g} eV",
        )
    ax.set_xlabel("initial $T_e$ grid value [eV]")
    ax.set_ylabel("peak |Hα/Hβ ratio error| [%]")
    ax.set_title("Temperature robustness at reference density")
    ax.legend(fontsize=8)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"Balmer_robustness_peak_ratio_error_vs_Te.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for dte, sub in temp.groupby("DeltaTe_requested_eV"):
        sub = sub.sort_values("Te_old_grid_eV")
        ax.plot(
            sub["Te_old_grid_eV"],
            sub["last_t_ratio_abs_error_gt_10pct_us"],
            marker="o",
            lw=2,
            label=f"requested ΔTe={dte:g} eV",
        )
    ax.set_xlabel("initial $T_e$ grid value [eV]")
    ax.set_ylabel("last time |ratio error| > 10% [µs]")
    ax.set_title("Persistence of ratio bias at reference density")
    ax.legend(fontsize=8)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"Balmer_robustness_duration_vs_Te.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_density_sweep(df: pd.DataFrame) -> None:
    setup_plot_style()
    den = df[df["sweep_type"] == "density_sweep"].copy()
    if den.empty:
        return
    den = den.sort_values("ne_grid_cm-3")

    fig, ax1 = plt.subplots(figsize=(7.5, 5.0))
    ax1.semilogx(
        den["ne_grid_cm-3"],
        den["peak_ratio_error_percent_abs"],
        marker="o",
        lw=2,
        label="peak |ratio error| [%]",
    )
    ax1.set_xlabel("$n_e$ grid value [cm$^{-3}$]")
    ax1.set_ylabel("peak |Hα/Hβ ratio error| [%]")

    ax2 = ax1.twinx()
    ax2.semilogx(
        den["ne_grid_cm-3"],
        den["last_t_ratio_abs_error_gt_10pct_us"],
        marker="s",
        lw=2,
        linestyle="--",
        label="last time |ratio error| > 10% [µs]",
    )
    ax2.set_ylabel("last time |ratio error| > 10% [µs]")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")
    ax1.set_title("Density robustness at reference temperature")
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"Balmer_robustness_density_sweep.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_peak_scatter(df: pd.DataFrame) -> None:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sc = ax.scatter(
        df["Te_old_grid_eV"],
        df["ne_grid_cm-3"],
        c=df["peak_ratio_error_percent_abs"],
        s=80,
        cmap="viridis",
        edgecolor="k",
        linewidth=0.4,
    )
    ax.set_yscale("log")
    ax.set_xlabel("initial $T_e$ grid value [eV]")
    ax.set_ylabel("$n_e$ grid value [cm$^{-3}$]")
    ax.set_title("Sparse robustness sweep: peak Hα/Hβ transient ratio error")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("peak |ratio error| [%]")
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"Balmer_robustness_peak_ratio_scatter.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Sweep construction and CLI
# -----------------------------------------------------------------------------

def make_cases(args: argparse.Namespace) -> List[Tuple[str, float, float, float]]:
    """
    Return list of unique cases:
       (sweep_type, te_old_requested, ne_requested, delta_requested)
    """
    cases: List[Tuple[str, float, float, float]] = []

    for te in args.te_sweep:
        for dte in args.deltas_temp:
            cases.append(("temperature_sweep", float(te), float(args.ne_ref), float(dte)))

    for ne in args.ne_sweep:
        cases.append(("density_sweep", float(args.te_ref), float(ne), float(args.delta_density)))

    # De-duplicate exact requested cases but preserve first sweep label.
    seen = set()
    unique: List[Tuple[str, float, float, float]] = []
    for case in cases:
        key = (round(case[1], 12), round(case[2], 6), round(case[3], 12))
        if key not in seen:
            seen.add(key)
            unique.append(case)
    return unique


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-point Balmer transient robustness sweep.")
    p.add_argument("--te-sweep", type=float, nargs="+", default=[2.0, 3.0, 5.0], help="Requested Te_old values for temperature sweep [eV].")
    p.add_argument("--ne-ref", type=float, default=1.39e14, help="Requested reference ne for temperature sweep [cm^-3].")
    p.add_argument("--deltas-temp", type=float, nargs="+", default=[0.3, 0.6, 1.0], help="Requested DeltaTe values for temperature sweep [eV].")
    p.add_argument("--te-ref", type=float, default=3.0, help="Requested reference Te_old for density sweep [eV].")
    p.add_argument("--ne-sweep", type=float, nargs="+", default=[1e13, 1e14, 1e15], help="Requested ne values for density sweep [cm^-3].")
    p.add_argument("--delta-density", type=float, default=0.6, help="Requested DeltaTe for density sweep [eV].")
    p.add_argument("--n-time", type=int, default=350, help="Number of positive log time samples per transient.")
    p.add_argument("--t-min", type=float, default=None, help="Minimum positive time [s]. Default auto.")
    p.add_argument("--t-max", type=float, default=1e-4, help="Maximum time [s].")
    p.add_argument("--n-ion", type=float, default=btr.N_ION_DEFAULT, help="Ion density/source scaling.")
    p.add_argument("--use-photon-energy", action="store_true", help="Use energy emissivities instead of photon proxies.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading corrected v3 CR transient machinery...")
    print(f"Using driver module: {btr.__file__}")

    L_grid, S_grid = btr.load_cr_grids()
    line_weights = btr.load_radiative_weights(use_photon_energy=args.use_photon_energy)
    btr.print_weights(line_weights, use_photon_energy=args.use_photon_energy)

    cases = make_cases(args)
    print(f"Running {len(cases)} unique robustness cases...")

    rows: List[Dict[str, float | str]] = []

    for idx, (sweep_type, te_req, ne_req, dte_req) in enumerate(cases, start=1):
        print(
            f"[{idx:02d}/{len(cases):02d}] {sweep_type}: "
            f"Te_req={te_req:g} eV, ne_req={ne_req:.3e} cm^-3, ΔTe_req={dte_req:g} eV"
        )
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
        row = summarize_run(res, sweep_type=sweep_type)
        row["Te_old_requested_eV"] = float(te_req)
        row["ne_requested_cm-3"] = float(ne_req)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["sweep_type", "ne_grid_cm-3", "Te_old_grid_eV", "DeltaTe_actual_eV"]).reset_index(drop=True)

    out_csv = OUT_DIR / "Balmer_transient_robustness_summary.csv"
    out_tex = OUT_DIR / "Balmer_transient_robustness_summary.tex"
    df.to_csv(out_csv, index=False)

    table_cols = [
        "sweep_type",
        "Te_old_grid_eV",
        "ne_grid_cm-3",
        "DeltaTe_actual_eV",
        "Te_new_grid_eV",
        "peak_ratio_error_percent_abs",
        "t_peak_ratio_ns",
        "last_t_ratio_abs_error_gt_5pct_us",
        "last_t_ratio_abs_error_gt_10pct_us",
        "tau_fast_ns",
        "tau_slow_us",
        "M_tau_slow_over_tau_fast",
    ]
    df[table_cols].to_latex(
        out_tex,
        index=False,
        float_format=lambda x: f"{x:.3g}",
        escape=False,
        caption="Multi-point robustness sweep of transient Hα/Hβ ratio error.",
        label="tab:balmer_transient_robustness",
    )

    plot_temperature_sweep(df)
    plot_density_sweep(df)
    plot_peak_scatter(df)

    print()
    print("=" * 120)
    print("Robustness summary")
    print("=" * 120)
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(df[table_cols].to_string(index=False))

    print()
    print("Saved:")
    print(f"  {out_csv.relative_to(REPO)}")
    print(f"  {out_tex.relative_to(REPO)}")
    print(f"  {FIG_DIR.relative_to(REPO)}/Balmer_robustness_*.png/.pdf")

    print()
    print("Interpretation guide:")
    print("  1. If peak ratio error persists across Te and ne, the mechanism is robust.")
    print("  2. If peak is strongest near 2-4 eV, the mechanism is localized to the detachment-relevant window.")
    print("  3. If duration shrinks with ne, density mainly controls memory duration through faster relaxation.")
    print("  4. Use the tau_fast/tau_slow/M columns to connect the diagnostic bias to eigenvalue timescales.")


if __name__ == "__main__":
    main()
