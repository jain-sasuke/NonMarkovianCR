"""
Balmer_timescale_audit.py
=========================

Priority-2 audit script for the corrected v3 Balmer transient analysis.

Purpose
-------
Create clean, reproducible timescale tables connecting:

    peak Halpha/Hbeta transient ratio error
    last time |ratio error| > 10%
    tau_fast
    tau_slow
    M = tau_slow / tau_fast

This is meant to eliminate stale/inconsistent thesis numbers and define exactly
which timescale is being used.

It reuses the corrected local module:

    src/rates/Balmer_transient_ratio.py

so it uses the same v3 physics:

    - Hoang-Binh radiative A-values
    - only res_to_res rows
    - resolved Balmer transitions
    - analytic CR transient propagation

Run from repo rates directory:

    cd /Users/phi/Desktop/non_markovian_cr/src/rates
    python Balmer_timescale_audit.py

Outputs
-------
CSV/TEX:
    data/processed/sensitivity/timescale_audit/Balmer_timescale_audit_benchmark_deltas.csv
    data/processed/sensitivity/timescale_audit/Balmer_timescale_audit_benchmark_deltas.tex
    data/processed/sensitivity/timescale_audit/Balmer_timescale_audit_regime_selected.csv
    data/processed/sensitivity/timescale_audit/Balmer_timescale_audit_regime_summary.txt

Figures:
    figures/timescale_audit/Balmer_timescale_last_gt10_vs_tau_slow.png/.pdf
    figures/timescale_audit/Balmer_timescale_peak_error_vs_tau_slow.png/.pdf
    figures/timescale_audit/Balmer_timescale_peak_error_vs_M.png/.pdf
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import Balmer_transient_ratio as btr

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT_DIR = REPO / "data/processed/sensitivity/timescale_audit"
FIG_DIR = REPO / "figures/timescale_audit"
REGIME_CSV = REPO / "data/processed/sensitivity/regime/Balmer_transient_regime_heatmap_DTe_p0p60.csv"

# Old/stale thesis numbers to explicitly compare against if desired.
# These are not used as truth. They are included to make inconsistencies visible.
OLD_TAU_RELAX_NS = 25.0
OLD_TAU_QSS_US = 15.3
OLD_M = 611.0


def setup_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


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
    err = np.asarray(err, dtype=float)
    times = np.asarray(times, dtype=float)
    valid = np.isfinite(err)
    above = valid & (np.abs(err) >= threshold)
    if not np.any(above):
        return 0.0, False
    last_idx = int(np.where(above)[0][-1])
    return float(times[last_idx]), bool(last_idx == len(times) - 1)


def timescale_metrics(eig_times_s: np.ndarray) -> Tuple[float, float, float]:
    """
    Definition used in this audit:

        tau_slow = largest positive decay time from post-step L_new eigenvalues
        tau_fast = second-largest positive decay time from post-step L_new eigenvalues
        M        = tau_slow / tau_fast

    This is deliberately explicit. Do not mix this with older thesis definitions
    unless those definitions are rederived and shown to be equivalent.
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


def summarize_transient(res: btr.TransientResult, te_requested: float, ne_requested: float) -> Dict[str, float | bool]:
    peak_R_signed, peak_R_abs, t_peak_R = peak_abs(res.times, res.err_ratio)
    peak_Ha_signed, peak_Ha_abs, t_peak_Ha = peak_abs(res.times, res.err_Halpha)
    peak_Hb_signed, peak_Hb_abs, t_peak_Hb = peak_abs(res.times, res.err_Hbeta)
    last_5, cens_5 = last_time_abs_above(res.times, res.err_ratio, 0.05)
    last_10, cens_10 = last_time_abs_above(res.times, res.err_ratio, 0.10)
    tau_slow, tau_fast, M = timescale_metrics(res.eig_times_s)

    return {
        "Te_requested_eV": float(te_requested),
        "ne_requested_cm-3": float(ne_requested),
        "DeltaTe_nominal_eV": float(res.delta_nominal),
        "DeltaTe_actual_eV": float(res.delta_actual),
        "Te_old_grid_eV": float(res.te_old),
        "Te_new_grid_eV": float(res.te_new),
        "ne_grid_cm-3": float(res.ne),
        "tau_fast_ns": 1e9 * tau_fast,
        "tau_slow_us": 1e6 * tau_slow,
        "M_tau_slow_over_tau_fast": M,
        "peak_ratio_error_percent_abs": 100.0 * peak_R_abs,
        "peak_ratio_error_percent_signed": 100.0 * peak_R_signed,
        "t_peak_ratio_ns": 1e9 * t_peak_R,
        "last_t_abs_ratio_error_gt_5pct_us": 1e6 * last_5,
        "last_t_abs_ratio_error_gt_10pct_us": 1e6 * last_10,
        "censored_gt_5pct": cens_5,
        "censored_gt_10pct": cens_10,
        "peak_Halpha_error_percent_abs": 100.0 * peak_Ha_abs,
        "peak_Hbeta_error_percent_abs": 100.0 * peak_Hb_abs,
    }


def run_benchmark_delta_audit(args: argparse.Namespace) -> pd.DataFrame:
    print("Loading corrected v3 transient machinery...")
    L_grid, S_grid = btr.load_cr_grids()
    line_weights = btr.load_radiative_weights(use_photon_energy=False)

    rows = []
    print("\nRunning benchmark delta timescale audit...")
    for dte in args.deltas:
        print(f"  requested DeltaTe={dte:g} eV")
        res = btr.run_single_transient(
            L_grid=L_grid,
            S_grid=S_grid,
            line_weights=line_weights,
            te_old_requested=args.te_old,
            ne_requested=args.ne,
            delta_nominal=dte,
            n_ion=args.n_ion,
            t_min=args.t_min,
            t_max=args.t_max,
            n_time=args.n_time,
        )
        rows.append(summarize_transient(res, args.te_old, args.ne))

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "Balmer_timescale_audit_benchmark_deltas.csv"
    out_tex = OUT_DIR / "Balmer_timescale_audit_benchmark_deltas.tex"
    df.to_csv(out_csv, index=False)

    paper_cols = [
        "DeltaTe_actual_eV",
        "Te_old_grid_eV",
        "Te_new_grid_eV",
        "ne_grid_cm-3",
        "tau_fast_ns",
        "tau_slow_us",
        "M_tau_slow_over_tau_fast",
        "peak_ratio_error_percent_abs",
        "last_t_abs_ratio_error_gt_10pct_us",
        "censored_gt_10pct",
    ]
    df[paper_cols].to_latex(
        out_tex,
        index=False,
        float_format=lambda x: f"{x:.4g}",
        escape=False,
        caption=(
            "Timescale audit for H$\\alpha$/H$\\beta$ transients at the nominal "
            "benchmark condition. $\\tau_{\\rm slow}$ is the largest post-step "
            "decay time; $\\tau_{\\rm fast}$ is the second-largest post-step decay time."
        ),
        label="tab:balmer_timescale_audit_benchmark",
    )

    print("\nSaved benchmark audit:")
    print(out_csv)
    print(out_tex)
    return df


def load_regime_audit() -> pd.DataFrame | None:
    if not REGIME_CSV.exists():
        print(f"\nRegime CSV not found, skipping regime audit: {REGIME_CSV}")
        return None
    df = pd.read_csv(REGIME_CSV)
    return df


def write_regime_selected_and_summary(regime: pd.DataFrame) -> None:
    selected_cols = [
        "Te_old_grid_eV",
        "Te_new_grid_eV",
        "ne_grid_cm-3",
        "DeltaTe_actual_eV",
        "peak_ratio_error_percent_abs",
        "last_t_ratio_abs_error_gt_10pct_us",
        "censored_gt_10pct",
        "tau_fast_ns",
        "tau_slow_us",
        "M_tau_slow_over_tau_fast",
    ]
    available_cols = [c for c in selected_cols if c in regime.columns]
    selected = regime[available_cols].copy()
    selected = selected.sort_values(["Te_old_grid_eV", "ne_grid_cm-3"])

    out_csv = OUT_DIR / "Balmer_timescale_audit_regime_selected.csv"
    selected.to_csv(out_csv, index=False)

    n = len(regime)
    gt10 = int((regime["peak_ratio_error_percent_abs"] >= 10).sum()) if "peak_ratio_error_percent_abs" in regime else -1
    gt20 = int((regime["peak_ratio_error_percent_abs"] >= 20).sum()) if "peak_ratio_error_percent_abs" in regime else -1
    gt50 = int((regime["peak_ratio_error_percent_abs"] >= 50).sum()) if "peak_ratio_error_percent_abs" in regime else -1
    cens10 = int(regime["censored_gt_10pct"].sum()) if "censored_gt_10pct" in regime else -1

    benchmark = regime.iloc[((regime["Te_old_grid_eV"] - 2.947052).abs() + np.abs(np.log10(regime["ne_grid_cm-3"] / 1.389495e14))).argmin()]

    txt = []
    txt.append("Balmer timescale audit: regime summary")
    txt.append("=" * 52)
    txt.append("")
    txt.append("Definition used:")
    txt.append("  tau_slow = largest positive post-step decay time from eigenvalues of L_new")
    txt.append("  tau_fast = second-largest positive post-step decay time from eigenvalues of L_new")
    txt.append("  M        = tau_slow / tau_fast")
    txt.append("  last_t   = last sampled time at which |Halpha/Hbeta error| exceeds threshold")
    txt.append("             not integrated duration")
    txt.append("")
    txt.append(f"Regime cases: {n}")
    txt.append(f"Peak ratio error >= 10%: {gt10}/{n}")
    txt.append(f"Peak ratio error >= 20%: {gt20}/{n}")
    txt.append(f"Peak ratio error >= 50%: {gt50}/{n}")
    txt.append(f"Cases censored at >10% threshold: {cens10}/{n}")
    txt.append("")
    txt.append("Nearest benchmark-grid row:")
    for key in available_cols:
        val = benchmark[key]
        txt.append(f"  {key}: {val}")
    txt.append("")
    txt.append("Stale-number warning:")
    txt.append(f"  Old thesis/example tau_relax: {OLD_TAU_RELAX_NS:g} ns")
    txt.append(f"  Old thesis/example tau_QSS:   {OLD_TAU_QSS_US:g} us")
    txt.append(f"  Old thesis/example M:         {OLD_M:g}")
    txt.append("  Do not reuse these unless their definition is rederived and matched to the current matrix/grid.")

    out_txt = OUT_DIR / "Balmer_timescale_audit_regime_summary.txt"
    out_txt.write_text("\n".join(txt) + "\n")

    print("\nSaved regime audit:")
    print(out_csv)
    print(out_txt)


def plot_regime_correlations(regime: pd.DataFrame) -> None:
    if regime is None or regime.empty:
        return

    plt.rcParams.update({"font.family": "serif", "font.size": 12})

    # Only positive values for log plots.
    df = regime.copy()
    df = df[np.isfinite(df["tau_slow_us"]) & (df["tau_slow_us"] > 0)]

    # 1. Last >10% vs tau_slow; nonzero last times only.
    df_last = df[np.isfinite(df["last_t_ratio_abs_error_gt_10pct_us"]) & (df["last_t_ratio_abs_error_gt_10pct_us"] > 0)].copy()
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    sc = ax.scatter(
        df_last["tau_slow_us"],
        df_last["last_t_ratio_abs_error_gt_10pct_us"],
        c=df_last["peak_ratio_error_percent_abs"],
        s=36,
        edgecolor="k",
        linewidth=0.25,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\tau_{\rm slow}$ [$\mu$s]")
    ax.set_ylabel(r"last time $|H\alpha/H\beta\ \mathrm{error}| > 10\%$ [$\mu$s]")
    ax.set_title(r"Persistence of Balmer-ratio bias vs slow CR time")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"peak $|H\alpha/H\beta|$ error [%]")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Balmer_timescale_last_gt10_vs_tau_slow.png", dpi=300)
    fig.savefig(FIG_DIR / "Balmer_timescale_last_gt10_vs_tau_slow.pdf")
    plt.close(fig)

    # 2. Peak error vs tau_slow.
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    sc = ax.scatter(
        df["tau_slow_us"],
        df["peak_ratio_error_percent_abs"],
        c=np.log10(df["ne_grid_cm-3"]),
        s=36,
        edgecolor="k",
        linewidth=0.25,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\tau_{\rm slow}$ [$\mu$s]")
    ax.set_ylabel(r"peak $|H\alpha/H\beta|$ error [%]")
    ax.set_title(r"Peak Balmer-ratio bias vs slow CR time")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"$\log_{10}(n_e\,[\mathrm{cm}^{-3}])$")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Balmer_timescale_peak_error_vs_tau_slow.png", dpi=300)
    fig.savefig(FIG_DIR / "Balmer_timescale_peak_error_vs_tau_slow.pdf")
    plt.close(fig)

    # 3. Peak error vs M.
    dfM = df[np.isfinite(df["M_tau_slow_over_tau_fast"]) & (df["M_tau_slow_over_tau_fast"] > 0)].copy()
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    sc = ax.scatter(
        dfM["M_tau_slow_over_tau_fast"],
        dfM["peak_ratio_error_percent_abs"],
        c=dfM["Te_old_grid_eV"],
        s=36,
        edgecolor="k",
        linewidth=0.25,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$M=\tau_{\rm slow}/\tau_{\rm fast}$")
    ax.set_ylabel(r"peak $|H\alpha/H\beta|$ error [%]")
    ax.set_title(r"Peak Balmer-ratio bias vs timescale separation")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"initial $T_e$ [eV]")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Balmer_timescale_peak_error_vs_M.png", dpi=300)
    fig.savefig(FIG_DIR / "Balmer_timescale_peak_error_vs_M.pdf")
    plt.close(fig)

    print("\nSaved timescale figures:")
    print(FIG_DIR / "Balmer_timescale_last_gt10_vs_tau_slow.png")
    print(FIG_DIR / "Balmer_timescale_peak_error_vs_tau_slow.png")
    print(FIG_DIR / "Balmer_timescale_peak_error_vs_M.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--te-old", type=float, default=3.0)
    parser.add_argument("--ne", type=float, default=1.389495e14)
    parser.add_argument("--deltas", type=float, nargs="+", default=[0.3, 0.6, 1.0, 2.0, 3.0])
    parser.add_argument("--n-ion", type=float, default=1e14)
    parser.add_argument("--t-min", type=float, default=None)
    parser.add_argument("--t-max", type=float, default=1e-4)
    parser.add_argument("--n-time", type=int, default=300)
    args = parser.parse_args()

    setup_dirs()
    bench = run_benchmark_delta_audit(args)

    regime = load_regime_audit()
    if regime is not None:
        write_regime_selected_and_summary(regime)
        plot_regime_correlations(regime)

    print("\n" + "=" * 100)
    print("Benchmark delta audit")
    print("=" * 100)
    cols = [
        "DeltaTe_actual_eV", "Te_old_grid_eV", "Te_new_grid_eV", "ne_grid_cm-3",
        "tau_fast_ns", "tau_slow_us", "M_tau_slow_over_tau_fast",
        "peak_ratio_error_percent_abs", "last_t_abs_ratio_error_gt_10pct_us", "censored_gt_10pct",
    ]
    print(bench[cols].to_string(index=False))

    print("\nInterpretation warning:")
    print("  These tau values are post-step eigenvalue decay times with the explicit definition in the output files.")
    print("  Do not mix them with older thesis numbers unless those older numbers use the same matrix, grid point, and definition.")


if __name__ == "__main__":
    main()
