"""
Balmer_ratio_sensitivity_v2.py
==============================

Balmer observable sensitivity analysis using the actual Hoang-Binh radiative
A-values stored in:

    data/processed/Radiative/radiative_rates.csv

This replaces the older exploratory version that estimated A-values from the
CR matrix L. Put this file in:

    <repo>/src/rates/Balmer_ratio_sensitivity.py

and run:

    cd <repo>/src/rates
    python Balmer_ratio_sensitivity.py

What it computes
----------------
For each observable O:

    1. Halpha emissivity proxy
    2. Hbeta emissivity proxy
    3. Halpha / Hbeta line ratio

it computes the local QSS logarithmic sensitivity:

    a_O(Te, ne) = d ln(O_QSS) / dTe

and compares the finite-step QSS target mismatch:

    eps_O = |O(Te) - O(Te + DeltaTe)| / O(Te + DeltaTe)

against:

    linear predictor:       |a_O DeltaTe_actual|
    local exp predictor:    |exp(-a_O DeltaTe_actual) - 1|

Notes
-----
- The script uses actual grid-snapped DeltaTe, not only nominal DeltaTe.
- The common photon-energy factor cancels in fractional mismatch for a single
  line. For Halpha/Hbeta, including h nu changes the displayed absolute ratio
  by a constant factor but does not change sensitivity or fractional mismatch.
  USE_PHOTON_ENERGY is therefore False by default to match photon-emissivity
  style ratios. Set it True if you want power-emissivity ratios.
- Output figures are intentionally cleaner than the previous version, but the
  terminal summary is the main thing to inspect.
"""

from __future__ import annotations

import os
import sys
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Paths and imports
# -----------------------------------------------------------------------------

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_SENS = _REPO / "data/processed/sensitivity"
_FIG = _REPO / "figures"
_RAD_CSV = _REPO / "data/processed/Radiative/radiative_rates.csv"

sys.path.insert(0, str(_HERE))
from assemble_cr_matrix import TE_GRID, NE_GRID  # noqa: E402


# -----------------------------------------------------------------------------
# Constants and settings
# -----------------------------------------------------------------------------

N_ION = 1e14
MIN_POSITIVE = 1e-300
USE_PHOTON_ENERGY = False

# Vacuum wavelengths in nm. Only used if USE_PHOTON_ENERGY=True.
LINE_WAVELENGTH_NM = {
    "Halpha": 656.281,
    "Hbeta": 486.133,
}

# Each channel is specified as:
#   (n_upper, l_upper, n_lower, l_lower, label)
# Hydrogen E1: Delta l = +/- 1.
LINE_CHANNELS = {
    "Halpha": [
        (3, 0, 2, 1, "3S_to_2P"),
        (3, 1, 2, 0, "3P_to_2S"),
        (3, 2, 2, 1, "3D_to_2P"),
    ],
    "Hbeta": [
        (4, 0, 2, 1, "4S_to_2P"),
        (4, 1, 2, 0, "4P_to_2S"),
        (4, 2, 2, 1, "4D_to_2P"),
        # 4F -> 2D is absent because n=2 has no D state.
    ],
}

OBSERVABLE_ORDER = ["Halpha", "Hbeta", "Halpha_over_Hbeta"]
DELTA_TE_LIST = [0.3, 0.6, 1.0, 2.0, 3.0]


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------

@dataclass
class LineChannel:
    line_name: str
    label: str
    n_upper: int
    l_upper: int
    n_lower: int
    l_lower: int
    idx_upper: int
    idx_lower: int
    A_s: float
    weight: float


@dataclass
class ObservableSensitivity:
    name: str
    O_grid: np.ndarray
    a_grid: np.ndarray
    S_grid: np.ndarray


@dataclass
class ObservableStepResult:
    delta_nominal: float
    eps_actual: np.ndarray
    pred_linear: np.ndarray
    pred_exp: np.ndarray
    actual_delta_te: np.ndarray


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def pretty_name(name: str) -> str:
    mapping = {
        "Halpha": "Hα",
        "Hbeta": "Hβ",
        "Halpha_over_Hbeta": "Hα/Hβ",
    }
    return mapping.get(name, name)


def safe_name(name: str) -> str:
    return name.replace("/", "_over_").replace(" ", "_")


def load_grids() -> Tuple[np.ndarray, np.ndarray]:
    L_grid = np.load(str(_REPO / "data/processed/cr_matrix/L_grid.npy"))
    S_grid_src = np.load(str(_REPO / "data/processed/cr_matrix/S_grid.npy"))

    if L_grid.ndim != 4:
        raise ValueError(f"L_grid should be 4D, got shape {L_grid.shape}")
    if S_grid_src.ndim != 3:
        raise ValueError(f"S_grid should be 3D, got shape {S_grid_src.shape}")

    n_te, n_ne, n_states, n_states_2 = L_grid.shape
    if n_states != n_states_2:
        raise ValueError("L_grid matrices are not square.")
    if S_grid_src.shape != (n_te, n_ne, n_states):
        raise ValueError(
            f"S_grid shape mismatch: expected {(n_te, n_ne, n_states)}, "
            f"got {S_grid_src.shape}"
        )
    if len(TE_GRID) != n_te:
        raise ValueError("TE_GRID length does not match L_grid.")
    if len(NE_GRID) != n_ne:
        raise ValueError("NE_GRID length does not match L_grid.")

    return L_grid, S_grid_src


def steady_state(L: np.ndarray, S_src: np.ndarray, n_ion: float = N_ION) -> np.ndarray:
    """Solve L n_ss = -S_src * n_ion."""
    n_ss = np.linalg.solve(L, -S_src * n_ion)
    if not np.all(np.isfinite(n_ss)):
        raise FloatingPointError("Non-finite steady-state population.")

    # Clip numerical roundoff. Warn only for meaningful negativity.
    scale = max(float(np.max(np.abs(n_ss))), 1.0)
    tol = 1e-10 * scale
    min_val = float(np.min(n_ss))
    if min_val < -tol:
        print(
            f"WARNING: significant negative steady-state population: "
            f"min={min_val:.3e}, tol={tol:.3e}. Clipping for diagnostic."
        )
    return np.where(n_ss < 0.0, 0.0, n_ss)


def derivative_bracket(ti: int, n_te: int) -> Tuple[int, int]:
    if ti == 0:
        return 0, 1
    if ti == n_te - 1:
        return n_te - 2, n_te - 1
    return ti - 1, ti + 1


def nearest_new_index(ti: int, delta_te_nominal: float) -> Tuple[int | None, float]:
    te_old = float(TE_GRID[ti])
    te_target = te_old + float(delta_te_nominal)
    if te_target < float(TE_GRID[0]) or te_target > float(TE_GRID[-1]):
        return None, np.nan
    ti_new = int(np.argmin(np.abs(TE_GRID - te_target)))
    actual_delta = float(TE_GRID[ti_new] - TE_GRID[ti])
    if ti_new == ti:
        return None, np.nan
    return ti_new, actual_delta


def log_space_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0) & (y_pred > 0)
    if np.count_nonzero(valid) < 3:
        return np.nan
    yt = np.log10(y_true[valid])
    yp = np.log10(y_pred[valid])
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot <= 0:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


def ratio_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0) & (y_pred > 0)
    if np.count_nonzero(valid) == 0:
        return np.nan, np.nan, np.nan
    ratio = y_true[valid] / y_pred[valid]
    return (
        float(np.nanmedian(ratio)),
        float(np.nanpercentile(ratio, 10)),
        float(np.nanpercentile(ratio, 90)),
    )


# -----------------------------------------------------------------------------
# Radiative A-value loading
# -----------------------------------------------------------------------------

def load_radiative_rates_csv() -> pd.DataFrame:
    if not _RAD_CSV.exists():
        raise FileNotFoundError(
            f"Radiative rates file not found: {_RAD_CSV}\n"
            "Expected relative path from repo root: "
            "data/processed/Radiative/radiative_rates.csv"
        )
    df = pd.read_csv(_RAD_CSV)
    required = {
        "n_upper", "l_upper", "label_upper", "n_lower", "l_lower", "label_lower",
        "A_s-1", "idx_upper", "idx_lower"
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"radiative_rates.csv missing columns: {missing}")
    return df


def photon_energy_weight(line_name: str) -> float:
    if not USE_PHOTON_ENERGY:
        return 1.0
    # h*c/lambda constant is unnecessary; only relative line factors matter.
    return 1.0 / LINE_WAVELENGTH_NM[line_name]


def lookup_channel(df: pd.DataFrame, line_name: str, spec: Tuple[int, int, int, int, str]) -> LineChannel:
    n_u, l_u, n_l, l_l, label = spec
    rows = df[
        (df["n_upper"] == n_u)
        & (df["l_upper"] == l_u)
        & (df["n_lower"] == n_l)
        & (df["l_lower"] == l_l)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one radiative row for {label} "
            f"({n_u},{l_u})->({n_l},{l_l}); found {len(rows)}."
        )
    row = rows.iloc[0]
    A_s = float(row["A_s-1"])
    energy_factor = photon_energy_weight(line_name)
    return LineChannel(
        line_name=line_name,
        label=label,
        n_upper=n_u,
        l_upper=l_u,
        n_lower=n_l,
        l_lower=l_l,
        idx_upper=int(row["idx_upper"]),
        idx_lower=int(row["idx_lower"]),
        A_s=A_s,
        weight=A_s * energy_factor,
    )


def load_line_channels() -> Dict[str, List[LineChannel]]:
    df = load_radiative_rates_csv()
    channels: Dict[str, List[LineChannel]] = {}
    for line_name, specs in LINE_CHANNELS.items():
        channels[line_name] = [lookup_channel(df, line_name, spec) for spec in specs]
    return channels


def print_line_channels(channels: Dict[str, List[LineChannel]]) -> None:
    print()
    print("=" * 88)
    print("Radiative A-values loaded from data/processed/Radiative/radiative_rates.csv")
    print("=" * 88)
    print(f"USE_PHOTON_ENERGY = {USE_PHOTON_ENERGY}")
    print("If False, reported line ratio is photon-emissivity style; sensitivities are unchanged.")
    print()
    for line_name, chs in channels.items():
        print(pretty_name(line_name))
        for ch in chs:
            print(
                f"  {ch.label:8s}: idx_upper={ch.idx_upper:2d}, idx_lower={ch.idx_lower:2d}, "
                f"A={ch.A_s:.6e} s^-1, weight={ch.weight:.6e}"
            )
    print()


# -----------------------------------------------------------------------------
# Observable construction
# -----------------------------------------------------------------------------

def line_emissivity_proxy(n_ss: np.ndarray, line_name: str, channels: Dict[str, List[LineChannel]]) -> float:
    total = 0.0
    for ch in channels[line_name]:
        total += ch.weight * n_ss[ch.idx_upper]
    return float(total)


def compute_observable_grids(
    L_grid: np.ndarray,
    S_grid_src: np.ndarray,
    channels: Dict[str, List[LineChannel]],
) -> Dict[str, np.ndarray]:
    n_te, n_ne, _, _ = L_grid.shape
    I_Ha = np.full((n_te, n_ne), np.nan, dtype=float)
    I_Hb = np.full((n_te, n_ne), np.nan, dtype=float)
    R_ab = np.full((n_te, n_ne), np.nan, dtype=float)

    for ti in range(n_te):
        for ni in range(n_ne):
            n_ss = steady_state(L_grid[ti, ni], S_grid_src[ti, ni])
            ha = line_emissivity_proxy(n_ss, "Halpha", channels)
            hb = line_emissivity_proxy(n_ss, "Hbeta", channels)
            I_Ha[ti, ni] = ha
            I_Hb[ti, ni] = hb
            if np.isfinite(ha) and np.isfinite(hb) and hb > MIN_POSITIVE:
                R_ab[ti, ni] = ha / hb

    return {
        "Halpha": I_Ha,
        "Hbeta": I_Hb,
        "Halpha_over_Hbeta": R_ab,
    }


def compute_observable_sensitivity(name: str, O_grid: np.ndarray) -> ObservableSensitivity:
    n_te, n_ne = O_grid.shape
    a_grid = np.full((n_te, n_ne), np.nan, dtype=float)

    for ti in range(n_te):
        ti_lo, ti_hi = derivative_bracket(ti, n_te)
        dte = float(TE_GRID[ti_hi] - TE_GRID[ti_lo])
        for ni in range(n_ne):
            O_lo = O_grid[ti_lo, ni]
            O_hi = O_grid[ti_hi, ni]
            if np.isfinite(O_lo) and np.isfinite(O_hi) and O_lo > MIN_POSITIVE and O_hi > MIN_POSITIVE:
                a_grid[ti, ni] = (np.log(O_hi) - np.log(O_lo)) / dte

    return ObservableSensitivity(name=name, O_grid=O_grid, a_grid=a_grid, S_grid=np.abs(a_grid))


def compute_observable_step_results(
    delta_te_list: List[float],
    sensitivity: ObservableSensitivity,
) -> Dict[float, ObservableStepResult]:
    n_te, n_ne = sensitivity.O_grid.shape
    results: Dict[float, ObservableStepResult] = {}

    for delta_nominal in delta_te_list:
        eps_actual = np.full((n_te, n_ne), np.nan, dtype=float)
        pred_linear = np.full((n_te, n_ne), np.nan, dtype=float)
        pred_exp = np.full((n_te, n_ne), np.nan, dtype=float)
        actual_delta_te = np.full((n_te, n_ne), np.nan, dtype=float)

        for ti in range(n_te):
            ti_new, dte_actual = nearest_new_index(ti, delta_nominal)
            if ti_new is None:
                continue

            for ni in range(n_ne):
                O_old = sensitivity.O_grid[ti, ni]
                O_new = sensitivity.O_grid[ti_new, ni]
                if np.isfinite(O_old) and np.isfinite(O_new) and O_new > MIN_POSITIVE:
                    eps_actual[ti, ni] = abs(O_old - O_new) / O_new

                a = sensitivity.a_grid[ti, ni]
                if np.isfinite(a):
                    actual_delta_te[ti, ni] = dte_actual
                    pred_linear[ti, ni] = abs(a * dte_actual)
                    exponent = float(np.clip(-a * dte_actual, -700.0, 700.0))
                    pred_exp[ti, ni] = abs(np.exp(exponent) - 1.0)

        results[delta_nominal] = ObservableStepResult(
            delta_nominal=float(delta_nominal),
            eps_actual=eps_actual,
            pred_linear=pred_linear,
            pred_exp=pred_exp,
            actual_delta_te=actual_delta_te,
        )
    return results


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


def plot_sensitivity_map(sensitivity: ObservableSensitivity) -> None:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    vmax = np.nanpercentile(sensitivity.S_grid, 95)
    im = ax.pcolormesh(
        NE_GRID,
        TE_GRID,
        sensitivity.S_grid,
        shading="auto",
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
    )
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(rf"$|\partial_{{T_e}} \ln({pretty_name(sensitivity.name)})|$ [eV$^{{-1}}$]")

    ax.set_xscale("log")
    ax.set_xlabel(r"$n_e$ [cm$^{-3}$]")
    ax.set_ylabel(r"$T_e$ [eV]")
    ax.set_title(rf"QSS {pretty_name(sensitivity.name)} sensitivity")

    ti_r = int(np.argmin(np.abs(TE_GRID - 3.0)))
    ni_r = int(np.argmin(np.abs(NE_GRID - 1.39e14)))
    ax.plot(NE_GRID[ni_r], TE_GRID[ti_r], "c*", ms=14, label="ITER ref")
    ax.legend(fontsize=9)

    plt.tight_layout()
    stem = f"Balmer_v2_{safe_name(sensitivity.name)}_sensitivity_map"
    for ext in ["png", "pdf"]:
        fig.savefig(str(_FIG / f"{stem}.{ext}"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_collapse(name: str, step_results: Dict[float, ObservableStepResult], predictor_name: str) -> None:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(7, 6))

    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
    markers = ["o", "s", "^", "D", "v", "P"]
    all_x, all_y = [], []

    for k, (delta_nominal, res) in enumerate(step_results.items()):
        x = getattr(res, predictor_name).ravel()
        y = res.eps_actual.ravel()
        valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        all_x.append(x[valid])
        all_y.append(y[valid])
        ax.scatter(
            x[valid],
            y[valid],
            c=colors[k % len(colors)],
            marker=markers[k % len(markers)],
            s=18,
            alpha=0.55,
            label=rf"$\Delta T_e={delta_nominal:+.1f}$ eV",
        )

    if all_x and any(len(x) for x in all_x):
        x_cat = np.concatenate([x for x in all_x if len(x)])
        y_cat = np.concatenate([y for y in all_y if len(y)])
        valid = np.isfinite(x_cat) & np.isfinite(y_cat) & (x_cat > 0) & (y_cat > 0)
        min_val = min(np.nanmin(x_cat[valid]), np.nanmin(y_cat[valid]))
        max_val = max(np.nanmax(x_cat[valid]), np.nanmax(y_cat[valid]))
        lo = 10 ** np.floor(np.log10(min_val))
        hi = 10 ** np.ceil(np.log10(max_val))
    else:
        lo, hi = 1e-4, 10.0

    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label=r"$y=x$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    p_name = pretty_name(name)
    if predictor_name == "pred_linear":
        ax.set_xlabel(rf"linear predictor $|a_{{{p_name}}}\Delta T_e|$")
        ax.set_title(rf"{p_name} QSS target mismatch vs linear predictor")
        pred_suffix = "linear"
    elif predictor_name == "pred_exp":
        ax.set_xlabel(rf"local finite-step predictor $|\exp(-a_{{{p_name}}}\Delta T_e)-1|$")
        ax.set_title(rf"{p_name} QSS target mismatch vs finite-step predictor")
        pred_suffix = "exp"
    else:
        ax.set_xlabel(predictor_name)
        pred_suffix = predictor_name

    ax.set_ylabel(rf"actual {p_name} QSS target mismatch")
    ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    stem = f"Balmer_v2_{safe_name(name)}_collapse_{pred_suffix}"
    for ext in ["png", "pdf"]:
        fig.savefig(str(_FIG / f"{stem}.{ext}"), dpi=180, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Reporting and saving
# -----------------------------------------------------------------------------

def print_observable_metrics(sensitivity: ObservableSensitivity, step_results: Dict[float, ObservableStepResult]) -> List[dict]:
    ti_r = int(np.argmin(np.abs(TE_GRID - 3.0)))
    ni_r = int(np.argmin(np.abs(NE_GRID - 1.39e14)))
    rows = []

    print()
    print("=" * 88)
    print(f"ITER-nearest point: {pretty_name(sensitivity.name)}")
    print("=" * 88)
    print(f"TE_GRID nearest to 3 eV      : {TE_GRID[ti_r]:.6g} eV")
    print(f"NE_GRID nearest to 1.39e14   : {NE_GRID[ni_r]:.6e} cm^-3")
    print(f"O_QSS                        : {sensitivity.O_grid[ti_r, ni_r]:.6e}")
    print(f"a = d ln(O_QSS)/dTe          : {sensitivity.a_grid[ti_r, ni_r]:.6e} eV^-1")
    print(f"|a|                          : {sensitivity.S_grid[ti_r, ni_r]:.6e} eV^-1")
    print()

    header = (
        f"{'DeltaTe_nom':>12}  {'DeltaTe_act':>12}  "
        f"{'eps_actual':>12}  {'pred_lin':>12}  {'act/lin':>9}  "
        f"{'pred_exp':>12}  {'act/exp':>9}"
    )
    print(header)
    print("-" * len(header))

    for delta_nominal, res in step_results.items():
        eps = res.eps_actual[ti_r, ni_r]
        lin = res.pred_linear[ti_r, ni_r]
        exp = res.pred_exp[ti_r, ni_r]
        dta = res.actual_delta_te[ti_r, ni_r]
        ratio_lin = eps / lin if np.isfinite(eps) and np.isfinite(lin) and lin > 0 else np.nan
        ratio_exp = eps / exp if np.isfinite(eps) and np.isfinite(exp) and exp > 0 else np.nan
        print(
            f"{delta_nominal:12.3f}  {dta:12.5f}  "
            f"{eps:12.5g}  {lin:12.5g}  {ratio_lin:9.3f}  "
            f"{exp:12.5g}  {ratio_exp:9.3f}"
        )

    print()
    print("=" * 88)
    print(f"Global collapse metrics: {pretty_name(sensitivity.name)}")
    print("=" * 88)

    for delta_nominal, res in step_results.items():
        y = res.eps_actual.ravel()
        for pred_name, pred in [("linear", res.pred_linear.ravel()), ("exp", res.pred_exp.ravel())]:
            r2 = log_space_r2(y, pred)
            med, p10, p90 = ratio_stats(y, pred)
            print(
                f"DeltaTe={delta_nominal:+.2f} eV | {pred_name:6s} predictor | "
                f"logR2={r2: .4f} | actual/pred median={med: .3f} "
                f"[p10={p10: .3f}, p90={p90: .3f}]"
            )
            rows.append({
                "observable": sensitivity.name,
                "DeltaTe_nominal_eV": delta_nominal,
                "predictor": pred_name,
                "logR2": r2,
                "actual_over_pred_median": med,
                "actual_over_pred_p10": p10,
                "actual_over_pred_p90": p90,
            })
    print()
    return rows


def save_outputs(
    channels: Dict[str, List[LineChannel]],
    sensitivities: Dict[str, ObservableSensitivity],
    all_step_results: Dict[str, Dict[float, ObservableStepResult]],
    summary_rows: List[dict],
) -> None:
    os.makedirs(str(_SENS), exist_ok=True)
    save_dict = {"TE_GRID": np.asarray(TE_GRID), "NE_GRID": np.asarray(NE_GRID)}

    for line_name, chs in channels.items():
        for ch in chs:
            save_dict[f"A_{line_name}_{ch.label}"] = np.asarray(ch.A_s)
            save_dict[f"weight_{line_name}_{ch.label}"] = np.asarray(ch.weight)
            save_dict[f"idx_upper_{line_name}_{ch.label}"] = np.asarray(ch.idx_upper)
            save_dict[f"idx_lower_{line_name}_{ch.label}"] = np.asarray(ch.idx_lower)

    for name, sens in sensitivities.items():
        save_dict[f"{name}_O_grid"] = sens.O_grid
        save_dict[f"{name}_a_grid"] = sens.a_grid
        save_dict[f"{name}_S_grid"] = sens.S_grid

    for name, step_results in all_step_results.items():
        for delta_nominal, res in step_results.items():
            key = str(delta_nominal).replace("-", "m").replace(".", "p")
            save_dict[f"{name}_eps_actual_DTe_{key}"] = res.eps_actual
            save_dict[f"{name}_pred_linear_DTe_{key}"] = res.pred_linear
            save_dict[f"{name}_pred_exp_DTe_{key}"] = res.pred_exp
            save_dict[f"{name}_actual_delta_te_DTe_{key}"] = res.actual_delta_te

    np.savez_compressed(str(_SENS / "Balmer_ratio_sensitivity_v2_results.npz"), **save_dict)
    pd.DataFrame(summary_rows).to_csv(str(_SENS / "Balmer_ratio_sensitivity_v2_summary.csv"), index=False)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    os.makedirs(str(_SENS), exist_ok=True)
    os.makedirs(str(_FIG), exist_ok=True)

    print("Loading CR matrix grids...")
    L_grid, S_grid_src = load_grids()
    print(f"L_grid shape: {L_grid.shape}")
    print(f"S_grid shape: {S_grid_src.shape}")
    print()

    print("Loading Hoang-Binh radiative A-values from relative path...")
    print(f"  {_RAD_CSV.relative_to(_REPO)}")
    channels = load_line_channels()
    print_line_channels(channels)

    print("Computing line emissivity and ratio grids...")
    observable_grids = compute_observable_grids(L_grid, S_grid_src, channels)

    print("Computing sensitivities...")
    sensitivities = {name: compute_observable_sensitivity(name, grid) for name, grid in observable_grids.items()}

    print(f"Computing step mismatches for DeltaTe = {DELTA_TE_LIST} eV...")
    all_step_results = {
        name: compute_observable_step_results(DELTA_TE_LIST, sens)
        for name, sens in sensitivities.items()
    }

    summary_rows: List[dict] = []
    for name in OBSERVABLE_ORDER:
        summary_rows.extend(print_observable_metrics(sensitivities[name], all_step_results[name]))

    print("Saving arrays and summary table...")
    save_outputs(channels, sensitivities, all_step_results, summary_rows)

    print("Generating figures...")
    for name in OBSERVABLE_ORDER:
        plot_sensitivity_map(sensitivities[name])
        plot_collapse(name, all_step_results[name], "pred_linear")
        plot_collapse(name, all_step_results[name], "pred_exp")

    print()
    print("Done.")
    print("Saved outputs:")
    print("  data/processed/sensitivity/Balmer_ratio_sensitivity_v2_results.npz")
    print("  data/processed/sensitivity/Balmer_ratio_sensitivity_v2_summary.csv")
    print("  figures/Balmer_v2_*.png and .pdf")
    print()
    print("Interpretation reminder:")
    print("  Use the linear predictor for Halpha/Hbeta if its logR2 and median ratio are better.")
    print("  Do not use the local exponential predictor where it overpredicts large steps.")


if __name__ == "__main__":
    main()
