"""
Balmer_transient_ratio_fast.py
============================

Full time-dependent CR transient for Hα, Hβ, and Hα/Hβ after a Te step.

Purpose
-------
This is the next analysis after the static QSS target-mismatch sensitivity tests.
It answers the PPCF-style diagnostic question:

    Does the time-dependent CR evolution produce a transient Hα/Hβ ratio error,
    not just an Hα absolute-intensity error?

Model used
----------
At fixed ne, start from the old QSS state at Te_old:

    n(0) = n_ss(Te_old, ne)

At t = 0, switch to the post-step matrix/source at Te_new = Te_old + DeltaTe:

    dn/dt = L_new n + S_new * N_ION

The solution is computed analytically using matrix exponential action:

    n(t) = n_ss_new + exp(L_new t) [n_ss_old - n_ss_new]

Line proxies
------------
Uses actual Hoang-Binh A-values from:

    data/processed/Radiative/radiative_rates.csv

Hα = sum_{3l -> 2l'} A_ul n_u
Hβ = sum_{4l -> 2l'} A_ul n_u

The common photon-energy factor is disabled by default. It changes absolute
Hα/Hβ ratio values if enabled, but not logarithmic Te sensitivities. For
transient fractional ratio error, a constant photon-energy factor also cancels
between CR and QSS target comparisons.

Outputs
-------
CSV summaries:
    data/processed/sensitivity/Balmer_transient_ratio_summary.csv
    data/processed/sensitivity/Balmer_transient_ratio_timeseries_DTe_*.csv

Figures:
    figures/Balmer_transient_ratio_DTe_*.png/.pdf
    figures/Balmer_transient_absolute_errors_DTe_*.png/.pdf
    figures/Balmer_transient_ratio_errors_overlay.png/.pdf

Run
---
cd /Users/phi/Desktop/non_markovian_cr/src/rates
python Balmer_transient_ratio_fast.py

Optional examples:
python Balmer_transient_ratio_fast.py --te-old 3.0 --ne 1.389495e14 --deltas 0.3 0.6 1.0
python Balmer_transient_ratio_fast.py --t-max 1e-3 --n-time 700
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.linalg import eig, solve as scipy_solve
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "This script requires scipy.linalg.eig and scipy.linalg.solve. "
        "Install scipy in your CR conda environment."
    ) from exc


# -----------------------------------------------------------------------------
# Paths and imports
# -----------------------------------------------------------------------------

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATA_CR = REPO / "data/processed/cr_matrix"
DATA_RAD = REPO / "data/processed/Radiative/radiative_rates.csv"
OUT_DIR = REPO / "data/processed/sensitivity"
FIG_DIR = REPO / "figures"

sys.path.insert(0, str(HERE))
from assemble_cr_matrix import TE_GRID, NE_GRID  # noqa: E402


# -----------------------------------------------------------------------------
# Constants and line definitions
# -----------------------------------------------------------------------------

N_ION_DEFAULT = 1e14
MIN_POSITIVE = 1e-300

# Assumed state indices from your 43-state ordering.
IDX_2S = 1
IDX_2P = 2
IDX_3S = 3
IDX_3P = 4
IDX_3D = 5
IDX_4S = 6
IDX_4P = 7
IDX_4D = 8

# Required radiative channels. These are read from radiative_rates.csv by idx.
LINE_CHANNELS = {
    "Halpha": [
        (IDX_3S, IDX_2P, "3S_to_2P"),
        (IDX_3P, IDX_2S, "3P_to_2S"),
        (IDX_3D, IDX_2P, "3D_to_2P"),
    ],
    "Hbeta": [
        (IDX_4S, IDX_2P, "4S_to_2P"),
        (IDX_4P, IDX_2S, "4P_to_2S"),
        (IDX_4D, IDX_2P, "4D_to_2P"),
    ],
}

# Photon wavelengths in meters. Only used if --use-photon-energy is passed.
LINE_WAVELENGTH_M = {
    "Halpha": 656.281e-9,
    "Hbeta": 486.135e-9,
}
H_PLANCK = 6.62607015e-34
C_LIGHT = 299792458.0


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------

@dataclass
class LineWeights:
    weights: Dict[str, Dict[str, float]]


@dataclass
class TransientResult:
    delta_nominal: float
    delta_actual: float
    te_old: float
    te_new: float
    ne: float
    ti_old: int
    ti_new: int
    ni: int
    times: np.ndarray
    Halpha_cr: np.ndarray
    Hbeta_cr: np.ndarray
    ratio_cr: np.ndarray
    Halpha_qss_new: float
    Hbeta_qss_new: float
    ratio_qss_new: float
    Halpha_qss_old: float
    Hbeta_qss_old: float
    ratio_qss_old: float
    err_Halpha: np.ndarray
    err_Hbeta: np.ndarray
    err_ratio: np.ndarray
    eig_times_s: np.ndarray


# -----------------------------------------------------------------------------
# Loaders and helpers
# -----------------------------------------------------------------------------

def load_cr_grids() -> Tuple[np.ndarray, np.ndarray]:
    L_path = DATA_CR / "L_grid.npy"
    S_path = DATA_CR / "S_grid.npy"

    if not L_path.exists():
        raise FileNotFoundError(f"Missing L_grid: {L_path}")
    if not S_path.exists():
        raise FileNotFoundError(f"Missing S_grid: {S_path}")

    L_grid = np.load(str(L_path))
    S_grid = np.load(str(S_path))

    if L_grid.ndim != 4:
        raise ValueError(f"Expected L_grid 4D, got {L_grid.shape}")
    if S_grid.ndim != 3:
        raise ValueError(f"Expected S_grid 3D, got {S_grid.shape}")

    n_te, n_ne, n_state, n_state_2 = L_grid.shape
    if n_state != n_state_2:
        raise ValueError("L_grid matrices are not square")
    if S_grid.shape != (n_te, n_ne, n_state):
        raise ValueError(f"S_grid shape mismatch: {S_grid.shape} vs {L_grid.shape}")
    if len(TE_GRID) != n_te or len(NE_GRID) != n_ne:
        raise ValueError("TE_GRID/NE_GRID lengths do not match L_grid")

    return L_grid, S_grid


def load_radiative_weights(use_photon_energy: bool = False) -> LineWeights:
    if not DATA_RAD.exists():
        raise FileNotFoundError(
            f"Missing radiative rates CSV: {DATA_RAD}\n"
            "Expected relative path from repo root: data/processed/Radiative/radiative_rates.csv"
        )

    df = pd.read_csv(DATA_RAD)
    required_cols = {"idx_upper", "idx_lower", "A_s-1"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"radiative_rates.csv missing columns: {sorted(missing)}")

    weights: Dict[str, Dict[str, float]] = {}

    for line, channels in LINE_CHANNELS.items():
        weights[line] = {}
        photon_factor = 1.0
        if use_photon_energy:
            photon_factor = H_PLANCK * C_LIGHT / LINE_WAVELENGTH_M[line]

        for upper, lower, label in channels:
            rows = df[(df["idx_upper"] == upper) & (df["idx_lower"] == lower)]
            if rows.empty:
                raise ValueError(
                    f"No radiative row found for {label}: idx_upper={upper}, idx_lower={lower}"
                )
            if len(rows) > 1:
                # Sum would be dangerous if duplicates are accidental. Use sum only if exact same transition split.
                A_val = float(rows["A_s-1"].sum())
                print(f"WARNING: multiple rows for {label}; using summed A = {A_val:.6e}")
            else:
                A_val = float(rows.iloc[0]["A_s-1"])
            weights[line][label] = A_val * photon_factor

    return LineWeights(weights=weights)


def print_weights(line_weights: LineWeights, use_photon_energy: bool) -> None:
    print()
    print("=" * 88)
    print(f"Radiative A-values loaded from {DATA_RAD.relative_to(REPO)}")
    print("=" * 88)
    print(f"USE_PHOTON_ENERGY = {use_photon_energy}")
    print("If False, line intensities are photon-emissivity proxies. Fractional errors are unaffected.")
    for line, channels in LINE_CHANNELS.items():
        print(line.replace("Halpha", "Hα").replace("Hbeta", "Hβ"))
        for _, _, label in channels:
            print(f"  {label:8s}: weight={line_weights.weights[line][label]:.6e}")
    print()


def nearest_index(grid: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(np.asarray(grid, dtype=float) - value)))


def nearest_step_indices(te_old_requested: float, delta_nominal: float) -> Tuple[int, int, float, float, float]:
    ti_old = nearest_index(TE_GRID, te_old_requested)
    te_old = float(TE_GRID[ti_old])
    te_target = te_old + float(delta_nominal)
    if te_target < float(TE_GRID[0]) or te_target > float(TE_GRID[-1]):
        raise ValueError(
            f"Requested Te_new={te_target:.6g} eV outside grid [{TE_GRID[0]}, {TE_GRID[-1]}]"
        )
    ti_new = nearest_index(TE_GRID, te_target)
    if ti_new == ti_old:
        raise ValueError(
            f"DeltaTe={delta_nominal} snaps to same Te grid point. Use a larger step or finer grid."
        )
    te_new = float(TE_GRID[ti_new])
    delta_actual = te_new - te_old
    return ti_old, ti_new, te_old, te_new, delta_actual


def steady_state(L: np.ndarray, S_src: np.ndarray, n_ion: float) -> np.ndarray:
    n_ss = np.linalg.solve(L, -S_src * n_ion)
    if not np.all(np.isfinite(n_ss)):
        raise FloatingPointError("Non-finite steady-state population")

    scale = max(float(np.max(np.abs(n_ss))), 1.0)
    tol = 1e-10 * scale
    min_val = float(np.min(n_ss))
    if min_val < -tol:
        print(
            f"WARNING: significant negative steady-state population: min={min_val:.3e}, "
            f"tol={tol:.3e}; clipping for diagnostics."
        )
    return np.where(n_ss < 0.0, 0.0, n_ss)


def line_intensity(n: np.ndarray, line: str, line_weights: LineWeights) -> float:
    total = 0.0
    for upper, _lower, label in LINE_CHANNELS[line]:
        total += line_weights.weights[line][label] * n[upper]
    return float(total)


def compute_observables(n: np.ndarray, line_weights: LineWeights) -> Tuple[float, float, float]:
    Ha = line_intensity(n, "Halpha", line_weights)
    Hb = line_intensity(n, "Hbeta", line_weights)
    ratio = Ha / Hb if Hb > MIN_POSITIVE else np.nan
    return Ha, Hb, ratio


def decay_times_from_L(L: np.ndarray) -> np.ndarray:
    eig = np.linalg.eigvals(L)
    real = np.real(eig)
    valid = real < -1e-30
    if not np.any(valid):
        return np.array([], dtype=float)
    times = -1.0 / real[valid]
    times = times[np.isfinite(times) & (times > 0)]
    return np.sort(times)[::-1]


def make_time_grid(eig_times: np.ndarray, t_min: float | None, t_max: float | None, n_time: int) -> np.ndarray:
    if eig_times.size == 0:
        tau_slow = 1e-5
        tau_fast = 1e-9
    else:
        tau_slow = float(np.nanmax(eig_times))
        tau_fast = float(np.nanmin(eig_times))

    if t_min is None:
        t_min_use = max(1e-13, min(1e-11, tau_fast / 100.0))
    else:
        t_min_use = float(t_min)

    if t_max is None:
        t_max_use = max(1e-6, min(1e-2, 10.0 * tau_slow))
    else:
        t_max_use = float(t_max)

    if t_max_use <= t_min_use:
        t_max_use = t_min_use * 1e6

    positive = np.logspace(np.log10(t_min_use), np.log10(t_max_use), int(n_time))
    return np.concatenate(([0.0], positive))


def safe_fractional_error(actual: np.ndarray | float, target: float) -> np.ndarray:
    if not np.isfinite(target) or abs(target) <= MIN_POSITIVE:
        return np.full_like(np.asarray(actual, dtype=float), np.nan, dtype=float)
    return (np.asarray(actual, dtype=float) - target) / target


def threshold_duration(times: np.ndarray, err: np.ndarray, threshold: float) -> float:
    """
    Duration from t=0 until |err| drops below threshold for good.
    Uses the last sampled time with |err| >= threshold.
    """
    mag = np.abs(err)
    valid = np.isfinite(mag)
    above = valid & (mag >= threshold)
    if not np.any(above):
        return 0.0
    return float(times[np.where(above)[0][-1]])


def peak_error(times: np.ndarray, err: np.ndarray) -> Tuple[float, float]:
    mag = np.abs(err)
    valid = np.isfinite(mag)
    if not np.any(valid):
        return np.nan, np.nan
    idx_valid = np.where(valid)[0]
    idx = idx_valid[int(np.nanargmax(mag[valid]))]
    return float(err[idx]), float(times[idx])


# -----------------------------------------------------------------------------
# Core transient calculation
# -----------------------------------------------------------------------------

def run_single_transient(
    L_grid: np.ndarray,
    S_grid: np.ndarray,
    line_weights: LineWeights,
    te_old_requested: float,
    ne_requested: float,
    delta_nominal: float,
    n_ion: float,
    t_min: float | None,
    t_max: float | None,
    n_time: int,
) -> TransientResult:
    ti_old, ti_new, te_old, te_new, delta_actual = nearest_step_indices(te_old_requested, delta_nominal)
    ni = nearest_index(NE_GRID, ne_requested)
    ne = float(NE_GRID[ni])

    L_old = L_grid[ti_old, ni]
    S_old = S_grid[ti_old, ni]
    L_new = L_grid[ti_new, ni]
    S_new = S_grid[ti_new, ni]

    n_ss_old = steady_state(L_old, S_old, n_ion=n_ion)
    n_ss_new = steady_state(L_new, S_new, n_ion=n_ion)

    Ha_old, Hb_old, R_old = compute_observables(n_ss_old, line_weights)
    Ha_new, Hb_new, R_new = compute_observables(n_ss_new, line_weights)

    eig_times = decay_times_from_L(L_new)
    times = make_time_grid(eig_times, t_min=t_min, t_max=t_max, n_time=n_time)

    deviation0 = n_ss_old - n_ss_new

    # Fast analytic solution for the linear constant-coefficient system:
    #   n(t) = n_ss_new + exp(L_new t) (n_ss_old - n_ss_new)
    # The previous v1 script called expm_multiply once per time point, which can be
    # very slow for stiff matrices and hundreds of log-spaced samples. Here we
    # diagonalize the 43x43 matrix once per DeltaTe and evaluate all times vectorized.
    # This is much faster. If the eigenvector matrix is very ill-conditioned, the
    # terminal output will warn you and you should cross-check one case with the
    # old expm_multiply version.
    eigvals, eigvecs = eig(L_new)
    cond_v = np.linalg.cond(eigvecs)
    if cond_v > 1e12:
        print(
            f"WARNING: eigenvector matrix is ill-conditioned, cond(V)={cond_v:.3e}. "
            "Use this fast result as a diagnostic and cross-check with expm_multiply if needed."
        )

    coeff = scipy_solve(eigvecs, deviation0, assume_a="gen")
    exp_wt = np.exp(np.outer(times, eigvals))
    deviations = (exp_wt * coeff[None, :]) @ eigvecs.T
    N = n_ss_new[None, :] + np.real_if_close(deviations, tol=1000).real

    # Clip tiny negative numerical noise for line diagnostics.
    N = np.where(N < 0.0, 0.0, N)

    n_time_steps = len(times)

    Ha_cr = np.empty(n_time_steps, dtype=float)
    Hb_cr = np.empty(n_time_steps, dtype=float)
    R_cr = np.empty(n_time_steps, dtype=float)

    for k in range(n_time_steps):
        Ha_cr[k], Hb_cr[k], R_cr[k] = compute_observables(N[k, :], line_weights)

    err_Ha = safe_fractional_error(Ha_cr, Ha_new)
    err_Hb = safe_fractional_error(Hb_cr, Hb_new)
    err_R = safe_fractional_error(R_cr, R_new)

    return TransientResult(
        delta_nominal=float(delta_nominal),
        delta_actual=float(delta_actual),
        te_old=float(te_old),
        te_new=float(te_new),
        ne=float(ne),
        ti_old=int(ti_old),
        ti_new=int(ti_new),
        ni=int(ni),
        times=times,
        Halpha_cr=Ha_cr,
        Hbeta_cr=Hb_cr,
        ratio_cr=R_cr,
        Halpha_qss_new=float(Ha_new),
        Hbeta_qss_new=float(Hb_new),
        ratio_qss_new=float(R_new),
        Halpha_qss_old=float(Ha_old),
        Hbeta_qss_old=float(Hb_old),
        ratio_qss_old=float(R_old),
        err_Halpha=err_Ha,
        err_Hbeta=err_Hb,
        err_ratio=err_R,
        eig_times_s=eig_times,
    )


# -----------------------------------------------------------------------------
# Saving and plotting
# -----------------------------------------------------------------------------

def slug_delta(delta: float) -> str:
    return (f"{delta:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p"))


def result_to_timeseries_df(res: TransientResult) -> pd.DataFrame:
    return pd.DataFrame({
        "t_s": res.times,
        "Halpha_CR": res.Halpha_cr,
        "Hbeta_CR": res.Hbeta_cr,
        "Halpha_over_Hbeta_CR": res.ratio_cr,
        "Halpha_QSS_new": res.Halpha_qss_new,
        "Hbeta_QSS_new": res.Hbeta_qss_new,
        "Halpha_over_Hbeta_QSS_new": res.ratio_qss_new,
        "err_Halpha_frac": res.err_Halpha,
        "err_Hbeta_frac": res.err_Hbeta,
        "err_Halpha_over_Hbeta_frac": res.err_ratio,
    })


def summarize_result(res: TransientResult) -> Dict[str, float | str]:
    peak_Ha, t_peak_Ha = peak_error(res.times, res.err_Halpha)
    peak_Hb, t_peak_Hb = peak_error(res.times, res.err_Hbeta)
    peak_R, t_peak_R = peak_error(res.times, res.err_ratio)

    # Print largest few eigenvalue times for traceability.
    eig_top = res.eig_times_s[:5]
    eig_top_str = ";".join(f"{x:.6e}" for x in eig_top)

    row: Dict[str, float | str] = {
        "DeltaTe_nominal_eV": res.delta_nominal,
        "DeltaTe_actual_eV": res.delta_actual,
        "Te_old_grid_eV": res.te_old,
        "Te_new_grid_eV": res.te_new,
        "ne_grid_cm-3": res.ne,
        "ti_old": res.ti_old,
        "ti_new": res.ti_new,
        "ni": res.ni,
        "Halpha_QSS_old": res.Halpha_qss_old,
        "Halpha_QSS_new": res.Halpha_qss_new,
        "Hbeta_QSS_old": res.Hbeta_qss_old,
        "Hbeta_QSS_new": res.Hbeta_qss_new,
        "ratio_QSS_old": res.ratio_qss_old,
        "ratio_QSS_new": res.ratio_qss_new,
        "initial_err_Halpha_frac": float(res.err_Halpha[0]),
        "initial_err_Hbeta_frac": float(res.err_Hbeta[0]),
        "initial_err_ratio_frac": float(res.err_ratio[0]),
        "peak_err_Halpha_frac": peak_Ha,
        "t_peak_Halpha_s": t_peak_Ha,
        "peak_err_Hbeta_frac": peak_Hb,
        "t_peak_Hbeta_s": t_peak_Hb,
        "peak_err_ratio_frac": peak_R,
        "t_peak_ratio_s": t_peak_R,
        "duration_abs_ratio_err_gt_1pct_s": threshold_duration(res.times, res.err_ratio, 0.01),
        "duration_abs_ratio_err_gt_5pct_s": threshold_duration(res.times, res.err_ratio, 0.05),
        "duration_abs_ratio_err_gt_10pct_s": threshold_duration(res.times, res.err_ratio, 0.10),
        "duration_abs_Halpha_err_gt_10pct_s": threshold_duration(res.times, res.err_Halpha, 0.10),
        "duration_abs_Hbeta_err_gt_10pct_s": threshold_duration(res.times, res.err_Hbeta, 0.10),
        "eig_decay_times_top5_s": eig_top_str,
    }
    return row


def setup_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.28,
    })


def plot_single_ratio_transient(res: TransientResult) -> None:
    setup_plot_style()
    slug = slug_delta(res.delta_nominal)

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True)

    ax = axes[0]
    ax.semilogx(res.times[1:], res.ratio_cr[1:], lw=2, label="CR transient")
    ax.axhline(res.ratio_qss_new, color="k", ls="--", lw=1.4, label="post-step QSS target")
    ax.axhline(res.ratio_qss_old, color="0.5", ls=":", lw=1.2, label="pre-step QSS")
    ax.set_ylabel("Hα/Hβ")
    ax.set_title(
        f"Hα/Hβ transient after Te step: {res.te_old:.3g}→{res.te_new:.3g} eV, "
        f"ne={res.ne:.2e} cm$^{{-3}}$"
    )
    ax.legend(fontsize=9)

    ax2 = axes[1]
    ax2.semilogx(res.times[1:], 100.0 * res.err_ratio[1:], lw=2, label="ratio error")
    ax2.axhline(0.0, color="k", lw=1)
    ax2.axhline(5.0, color="0.55", ls="--", lw=1, label="±5%")
    ax2.axhline(-5.0, color="0.55", ls="--", lw=1)
    ax2.axhline(10.0, color="0.75", ls=":", lw=1, label="±10%")
    ax2.axhline(-10.0, color="0.75", ls=":", lw=1)
    ax2.set_xlabel("time after step [s]")
    ax2.set_ylabel("ratio error [%]")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"Balmer_transient_ratio_DTe_{slug}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_single_absolute_errors(res: TransientResult) -> None:
    setup_plot_style()
    slug = slug_delta(res.delta_nominal)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.semilogx(res.times[1:], 100.0 * res.err_Halpha[1:], lw=2, label="Hα error")
    ax.semilogx(res.times[1:], 100.0 * res.err_Hbeta[1:], lw=2, label="Hβ error")
    ax.semilogx(res.times[1:], 100.0 * res.err_ratio[1:], lw=2, label="Hα/Hβ error")
    ax.axhline(0.0, color="k", lw=1)
    ax.axhline(10.0, color="0.65", ls="--", lw=1, label="±10%")
    ax.axhline(-10.0, color="0.65", ls="--", lw=1)
    ax.set_xlabel("time after step [s]")
    ax.set_ylabel("fractional error vs post-step QSS [%]")
    ax.set_title(
        f"Balmer transient errors: Te {res.te_old:.3g}→{res.te_new:.3g} eV, "
        f"ne={res.ne:.2e} cm$^{{-3}}$"
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"Balmer_transient_absolute_errors_DTe_{slug}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_overlay_ratio_errors(results: List[TransientResult]) -> None:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for res in results:
        ax.semilogx(
            res.times[1:],
            100.0 * res.err_ratio[1:],
            lw=2,
            label=f"ΔTe={res.delta_actual:+.3g} eV",
        )
    ax.axhline(0.0, color="k", lw=1)
    ax.axhline(5.0, color="0.65", ls="--", lw=1, label="±5%")
    ax.axhline(-5.0, color="0.65", ls="--", lw=1)
    ax.axhline(10.0, color="0.8", ls=":", lw=1, label="±10%")
    ax.axhline(-10.0, color="0.8", ls=":", lw=1)
    ax.set_xlabel("time after step [s]")
    ax.set_ylabel("Hα/Hβ error vs post-step QSS [%]")
    if results:
        ax.set_title(f"Hα/Hβ transient ratio error at ne={results[0].ne:.2e} cm$^{{-3}}$")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"Balmer_transient_ratio_errors_overlay.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run time-dependent CR Hα/Hβ transient after Te step."
    )
    parser.add_argument("--te-old", type=float, default=3.0, help="Requested initial Te [eV].")
    parser.add_argument("--ne", type=float, default=1.39e14, help="Requested ne [cm^-3].")
    parser.add_argument("--deltas", type=float, nargs="+", default=[0.3, 0.6, 1.0, 2.0, 3.0])
    parser.add_argument("--n-ion", type=float, default=N_ION_DEFAULT, help="Ion density/source scaling used in steady-state solve.")
    parser.add_argument("--t-min", type=float, default=None, help="Minimum positive time [s]. Default auto.")
    parser.add_argument("--t-max", type=float, default=None, help="Maximum time [s]. Default auto from eigenvalue times.")
    parser.add_argument("--n-time", type=int, default=600, help="Number of positive log-spaced time samples.")
    parser.add_argument("--use-photon-energy", action="store_true", help="Use energy emissivities instead of photon proxies.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading CR grids...")
    L_grid, S_grid = load_cr_grids()
    print(f"L_grid shape: {L_grid.shape}")
    print(f"S_grid shape: {S_grid.shape}")

    line_weights = load_radiative_weights(use_photon_energy=args.use_photon_energy)
    print_weights(line_weights, use_photon_energy=args.use_photon_energy)

    results: List[TransientResult] = []
    summary_rows: List[Dict[str, float | str]] = []

    print("Running transients...")
    for delta in args.deltas:
        print(f"  ΔTe nominal = {delta:+.3g} eV")
        res = run_single_transient(
            L_grid=L_grid,
            S_grid=S_grid,
            line_weights=line_weights,
            te_old_requested=args.te_old,
            ne_requested=args.ne,
            delta_nominal=float(delta),
            n_ion=float(args.n_ion),
            t_min=args.t_min,
            t_max=args.t_max,
            n_time=int(args.n_time),
        )
        results.append(res)
        row = summarize_result(res)
        summary_rows.append(row)

        ts = result_to_timeseries_df(res)
        slug = slug_delta(res.delta_nominal)
        ts_path = OUT_DIR / f"Balmer_transient_ratio_timeseries_DTe_{slug}.csv"
        ts.to_csv(ts_path, index=False)

        plot_single_ratio_transient(res)
        plot_single_absolute_errors(res)

    plot_overlay_ratio_errors(results)

    summary = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "Balmer_transient_ratio_summary.csv"
    summary.to_csv(summary_path, index=False)

    print()
    print("=" * 100)
    print("Transient summary")
    print("=" * 100)

    cols = [
        "DeltaTe_nominal_eV",
        "DeltaTe_actual_eV",
        "Te_old_grid_eV",
        "Te_new_grid_eV",
        "ne_grid_cm-3",
        "initial_err_Halpha_frac",
        "initial_err_Hbeta_frac",
        "initial_err_ratio_frac",
        "peak_err_Halpha_frac",
        "t_peak_Halpha_s",
        "peak_err_Hbeta_frac",
        "t_peak_Hbeta_s",
        "peak_err_ratio_frac",
        "t_peak_ratio_s",
        "duration_abs_ratio_err_gt_1pct_s",
        "duration_abs_ratio_err_gt_5pct_s",
        "duration_abs_ratio_err_gt_10pct_s",
        "duration_abs_Halpha_err_gt_10pct_s",
        "duration_abs_Hbeta_err_gt_10pct_s",
    ]
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(summary[cols].to_string(index=False))

    print()
    print("Top eigenvalue decay times for first run [s]:")
    if results:
        print(results[0].eig_times_s[:10])

    print()
    print("Saved:")
    print(f"  {summary_path.relative_to(REPO)}")
    print(f"  {OUT_DIR.relative_to(REPO)}/Balmer_transient_ratio_timeseries_DTe_*.csv")
    print("  figures/Balmer_transient_ratio_DTe_*.png/.pdf")
    print("  figures/Balmer_transient_absolute_errors_DTe_*.png/.pdf")
    print("  figures/Balmer_transient_ratio_errors_overlay.png/.pdf")

    print()
    print("Interpretation check:")
    print("  If peak_err_ratio_frac and duration_abs_ratio_err_gt_5pct_s are small,")
    print("  then Hα/Hβ line-ratio diagnostics are more robust than Hα-only intensity.")
    print("  If they are large and long-lived, this supports a PPCF-style diagnostic-bias claim.")


if __name__ == "__main__":
    main()
