"""
Halpha_sensitivity.py
=====================

H-alpha-specific QSS manifold sensitivity analysis.

This script tests whether the initial QSS H-alpha emissivity mismatch after a
temperature step can be predicted from the signed logarithmic sensitivity of
the QSS H-alpha emissivity.

Definitions
-----------
For a QSS steady state n^ss(Te, ne), define an H-alpha emissivity proxy:

    I_Ha(Te, ne) =
        A_3S_2P * n_3S
      + A_3P_2S * n_3P
      + A_3D_2P * n_3D

The common photon energy h nu is omitted because it cancels in fractional
errors.

Signed H-alpha sensitivity:

    a_Ha(Te, ne) = d/dTe ln I_Ha^QSS(Te, ne)

Actual finite-step QSS mismatch:

    eps_Ha_step =
        | I_Ha^QSS(Te) - I_Ha^QSS(Te + DeltaTe) |
        / I_Ha^QSS(Te + DeltaTe)

Linear predictor:

    eps_lin = |a_Ha * DeltaTe_actual|

Signed finite-step exponential predictor:

    eps_exp = | exp(-a_Ha * DeltaTe_actual) - 1 |

Outputs
-------
data/processed/sensitivity/Halpha_sensitivity_results.npz

figures/Halpha_sensitivity_map.png/.pdf
figures/Halpha_sensitivity_collapse_linear.png/.pdf
figures/Halpha_sensitivity_collapse_exp.png/.pdf

Run
---
cd ~/Desktop/non_markovian_cr/src/rates
python Halpha_sensitivity.py

Notes
-----
1. This script assumes the state ordering:
   index 0 = 1S
   index 1 = 2S
   index 2 = 2P
   index 3 = 3S
   index 4 = 3P
   index 5 = 3D

2. By default, H-alpha A-values are estimated from low-density CR-matrix
   off-diagonal entries:
      L[2P,3S], L[2S,3P], L[2P,3D]
   averaged over Te at the lowest ne grid point.

   This is a practical exploratory fallback. For final publication, replace
   estimate_halpha_weights_from_L(...) with your actual radiative A-values
   from the Hoang-Binh data used during matrix assembly.
"""

import os
import sys
import pathlib
from dataclasses import dataclass

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Paths and imports
# -----------------------------

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_SENS = _REPO / "data/processed/sensitivity"
_FIG = _REPO / "figures"

sys.path.insert(0, str(_HERE))

from assemble_cr_matrix import TE_GRID, NE_GRID


# -----------------------------
# Constants and state indices
# -----------------------------

N_ION = 1e14

MIN_POSITIVE = 1e-300

# Assumed 43-state ordering:
IDX_1S = 0
IDX_2S = 1
IDX_2P = 2
IDX_3S = 3
IDX_3P = 4
IDX_3D = 5

# H-alpha radiative channels:
# upper index, lower index, label
HALPHA_CHANNELS = [
    (IDX_3S, IDX_2P, "3S->2P"),
    (IDX_3P, IDX_2S, "3P->2S"),
    (IDX_3D, IDX_2P, "3D->2P"),
]

# If you want strict publication-grade values, replace these after checking
# your Hoang-Binh radiative table. Leave as None to estimate from L.
USER_HALPHA_A = {
    "3S->2P": None,
    "3P->2S": None,
    "3D->2P": None,
}


# -----------------------------
# Basic utilities
# -----------------------------

def load_grids():
    L_grid = np.load(str(_REPO / "data/processed/cr_matrix/L_grid.npy"))
    S_grid_src = np.load(str(_REPO / "data/processed/cr_matrix/S_grid.npy"))

    if L_grid.ndim != 4:
        raise ValueError(f"L_grid should be 4D, got {L_grid.shape}")
    if S_grid_src.ndim != 3:
        raise ValueError(f"S_grid should be 3D, got {S_grid_src.shape}")

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


def steady_state(L, S_src, n_ion=N_ION):
    """
    Solve:
        L n_ss = -S_src * n_ion
    """
    n_ss = np.linalg.solve(L, -S_src * n_ion)

    if not np.all(np.isfinite(n_ss)):
        raise FloatingPointError("Non-finite steady-state population.")

    # Clip tiny negative roundoff. Warn on larger negative values.
    scale = max(float(np.max(np.abs(n_ss))), 1.0)
    tol = 1e-10 * scale
    min_val = float(np.min(n_ss))

    if min_val < -tol:
        print(
            f"WARNING: significant negative steady-state population: "
            f"min={min_val:.3e}, tol={tol:.3e}. Clipping for diagnostic."
        )

    return np.where(n_ss < 0.0, 0.0, n_ss)


def derivative_bracket(ti, n_te):
    """
    Return finite-difference bracket indices around ti.
    """
    if ti == 0:
        return 0, 1
    if ti == n_te - 1:
        return n_te - 2, n_te - 1
    return ti - 1, ti + 1


def nearest_new_index(ti, delta_te_nominal):
    """
    Snap Te_old + DeltaTe_nominal to nearest grid point.
    Return (ti_new, DeltaTe_actual).
    """
    te_old = float(TE_GRID[ti])
    te_target = te_old + float(delta_te_nominal)

    if te_target < float(TE_GRID[0]) or te_target > float(TE_GRID[-1]):
        return None, np.nan

    ti_new = int(np.argmin(np.abs(TE_GRID - te_target)))
    actual_delta = float(TE_GRID[ti_new] - TE_GRID[ti])

    if ti_new == ti:
        return None, np.nan

    return ti_new, actual_delta


def log_space_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
        & (y_true > 0)
        & (y_pred > 0)
    )

    if np.count_nonzero(valid) < 3:
        return np.nan

    yt = np.log10(y_true[valid])
    yp = np.log10(y_pred[valid])

    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)

    if ss_tot <= 0:
        return np.nan

    return 1.0 - ss_res / ss_tot


def ratio_stats(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
        & (y_true > 0)
        & (y_pred > 0)
    )

    if np.count_nonzero(valid) == 0:
        return np.nan, np.nan, np.nan

    ratio = y_true[valid] / y_pred[valid]

    return (
        float(np.nanmedian(ratio)),
        float(np.nanpercentile(ratio, 10)),
        float(np.nanpercentile(ratio, 90)),
    )


# -----------------------------
# H-alpha emissivity weights
# -----------------------------

def estimate_halpha_weights_from_L(L_grid):
    """
    Estimate H-alpha radiative weights from low-density off-diagonal
    matrix entries.

    Matrix convention assumed:
        dn_i/dt = sum_j L[i,j] n_j + source_i
    So transition upper -> lower appears as positive L[lower, upper].

    This estimates:
        A_3S_2P ~ median_T L[2P,3S] at lowest ne
        A_3P_2S ~ median_T L[2S,3P] at lowest ne
        A_3D_2P ~ median_T L[2P,3D] at lowest ne

    This is NOT a substitute for actual radiative A-values in the final paper.
    """
    ni_low = 0
    weights = {}

    for upper, lower, label in HALPHA_CHANNELS:
        user_val = USER_HALPHA_A.get(label, None)

        if user_val is not None:
            weights[label] = float(user_val)
            continue

        vals = L_grid[:, ni_low, lower, upper]
        vals = vals[np.isfinite(vals) & (vals > 0)]

        if vals.size == 0:
            raise ValueError(
                f"Could not estimate positive weight for {label} "
                f"from L[lower={lower}, upper={upper}] at lowest ne."
            )

        weights[label] = float(np.nanmedian(vals))

    return weights


def print_halpha_weights(weights):
    print()
    print("=" * 78)
    print("H-alpha weights used")
    print("=" * 78)
    for _, _, label in HALPHA_CHANNELS:
        print(f"{label:8s}: {weights[label]:.6e} s^-1")
    print()
    print("WARNING:")
    print("  If these were estimated from L_grid, replace them with actual")
    print("  Hoang-Binh radiative A-values before publication.")
    print()


def halpha_intensity_proxy(n_ss, weights):
    """
    H-alpha emissivity proxy.

    Common factor h*nu is omitted because it cancels in fractional errors.
    """
    total = 0.0

    for upper, lower, label in HALPHA_CHANNELS:
        total += weights[label] * n_ss[upper]

    return float(total)


# -----------------------------
# Core calculations
# -----------------------------

@dataclass
class HalphaSensitivity:
    a_Ha: np.ndarray          # signed d ln I_Ha / dTe, shape (N_TE,N_NE)
    S_Ha: np.ndarray          # abs(a_Ha), shape (N_TE,N_NE)
    I_Ha_grid: np.ndarray     # QSS Halpha proxy, shape (N_TE,N_NE)


@dataclass
class HalphaStepResult:
    delta_nominal: float
    eps_actual: np.ndarray
    pred_linear: np.ndarray
    pred_exp: np.ndarray
    actual_delta_te: np.ndarray


def compute_I_Ha_grid(L_grid, S_grid_src, weights):
    n_te, n_ne, _, _ = L_grid.shape
    I_grid = np.full((n_te, n_ne), np.nan, dtype=float)

    for ti in range(n_te):
        for ni in range(n_ne):
            n_ss = steady_state(L_grid[ti, ni], S_grid_src[ti, ni])
            I_grid[ti, ni] = halpha_intensity_proxy(n_ss, weights)

    return I_grid


def compute_halpha_sensitivity(L_grid, S_grid_src, weights):
    n_te, n_ne, _, _ = L_grid.shape

    I_grid = compute_I_Ha_grid(L_grid, S_grid_src, weights)
    a_Ha = np.full((n_te, n_ne), np.nan, dtype=float)

    for ti in range(n_te):
        ti_lo, ti_hi = derivative_bracket(ti, n_te)
        dte = float(TE_GRID[ti_hi] - TE_GRID[ti_lo])

        for ni in range(n_ne):
            I_lo = I_grid[ti_lo, ni]
            I_hi = I_grid[ti_hi, ni]

            if (
                np.isfinite(I_lo)
                and np.isfinite(I_hi)
                and I_lo > MIN_POSITIVE
                and I_hi > MIN_POSITIVE
            ):
                a_Ha[ti, ni] = (np.log(I_hi) - np.log(I_lo)) / dte

    S_Ha = np.abs(a_Ha)

    return HalphaSensitivity(
        a_Ha=a_Ha,
        S_Ha=S_Ha,
        I_Ha_grid=I_grid,
    )


def compute_halpha_step_results(delta_te_list, sensitivity):
    n_te, n_ne = sensitivity.I_Ha_grid.shape
    results = {}

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
                I_old = sensitivity.I_Ha_grid[ti, ni]
                I_new = sensitivity.I_Ha_grid[ti_new, ni]

                if (
                    np.isfinite(I_old)
                    and np.isfinite(I_new)
                    and I_new > MIN_POSITIVE
                ):
                    eps_actual[ti, ni] = abs(I_old - I_new) / I_new

                a = sensitivity.a_Ha[ti, ni]

                if np.isfinite(a):
                    actual_delta_te[ti, ni] = dte_actual
                    pred_linear[ti, ni] = abs(a * dte_actual)

                    exponent = -a * dte_actual
                    exponent = float(np.clip(exponent, -700.0, 700.0))
                    pred_exp[ti, ni] = abs(np.exp(exponent) - 1.0)

        results[delta_nominal] = HalphaStepResult(
            delta_nominal=float(delta_nominal),
            eps_actual=eps_actual,
            pred_linear=pred_linear,
            pred_exp=pred_exp,
            actual_delta_te=actual_delta_te,
        )

    return results


# -----------------------------
# Plotting
# -----------------------------

def setup_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def plot_halpha_sensitivity_map(sensitivity):
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(6.8, 5.2))

    vmax = np.nanpercentile(sensitivity.S_Ha, 95)

    im = ax.pcolormesh(
        NE_GRID,
        TE_GRID,
        sensitivity.S_Ha,
        shading="auto",
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
    )

    cb = fig.colorbar(im, ax=ax)
    cb.set_label("|d ln(I_Ha)/dTe| [eV^-1]")

    ax.set_xscale("log")
    ax.set_xlabel("ne [cm^-3]")
    ax.set_ylabel("Te [eV]")
    ax.set_title("QSS H-alpha emissivity sensitivity")

    ti_r = int(np.argmin(np.abs(TE_GRID - 3.0)))
    ni_r = int(np.argmin(np.abs(NE_GRID - 1.39e14)))
    ax.plot(NE_GRID[ni_r], TE_GRID[ti_r], "c*", ms=14, label="ITER ref")
    ax.legend(fontsize=9)

    plt.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(str(_FIG / f"Halpha_sensitivity_map.{ext}"), dpi=180, bbox_inches="tight")

    plt.close(fig)


def plot_collapse(step_results, predictor_name, outfile_stem):
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(7, 6))

    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
    markers = ["o", "s", "^", "D", "v", "P"]

    all_x = []
    all_y = []

    for k, (delta_nominal, res) in enumerate(step_results.items()):
        x = getattr(res, predictor_name).ravel()
        y = res.eps_actual.ravel()

        valid = (
            np.isfinite(x)
            & np.isfinite(y)
            & (x > 0)
            & (y > 0)
        )

        all_x.append(x[valid])
        all_y.append(y[valid])

        ax.scatter(
            x[valid],
            y[valid],
            c=colors[k % len(colors)],
            marker=markers[k % len(markers)],
            s=18,
            alpha=0.55,
            label=f"DeltaTe={delta_nominal:+.1f} eV",
        )

    if all_x:
        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)
        valid = (
            np.isfinite(all_x)
            & np.isfinite(all_y)
            & (all_x > 0)
            & (all_y > 0)
        )

        if np.any(valid):
            min_val = min(np.nanmin(all_x[valid]), np.nanmin(all_y[valid]))
            max_val = max(np.nanmax(all_x[valid]), np.nanmax(all_y[valid]))
            lo = 10 ** np.floor(np.log10(min_val))
            hi = 10 ** np.ceil(np.log10(max_val))
        else:
            lo, hi = 1e-3, 10.0
    else:
        lo, hi = 1e-3, 10.0

    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="y=x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    if predictor_name == "pred_linear":
        ax.set_xlabel("|a_Ha DeltaTe|")
        ax.set_title("H-alpha step mismatch vs linear predictor")
    elif predictor_name == "pred_exp":
        ax.set_xlabel("|exp(-a_Ha DeltaTe)-1|")
        ax.set_title("H-alpha step mismatch vs finite-step predictor")
    else:
        ax.set_xlabel(predictor_name)

    ax.set_ylabel("actual H-alpha QSS step mismatch")
    ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(str(_FIG / f"{outfile_stem}.{ext}"), dpi=180, bbox_inches="tight")

    plt.close(fig)


# -----------------------------
# Reporting and saving
# -----------------------------

def print_metrics(sensitivity, step_results):
    ti_r = int(np.argmin(np.abs(TE_GRID - 3.0)))
    ni_r = int(np.argmin(np.abs(NE_GRID - 1.39e14)))

    print()
    print("=" * 78)
    print("ITER-nearest point: H-alpha sensitivity")
    print("=" * 78)
    print(f"TE_GRID nearest to 3 eV      : {TE_GRID[ti_r]:.6g} eV")
    print(f"NE_GRID nearest to 1.39e14   : {NE_GRID[ni_r]:.6e} cm^-3")
    print(f"I_Ha proxy                   : {sensitivity.I_Ha_grid[ti_r, ni_r]:.6e}")
    print(f"a_Ha = d ln(I_Ha)/dTe        : {sensitivity.a_Ha[ti_r, ni_r]:.6e} eV^-1")
    print(f"|a_Ha|                       : {sensitivity.S_Ha[ti_r, ni_r]:.6e} eV^-1")
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
    print("=" * 78)
    print("Global H-alpha collapse metrics")
    print("=" * 78)

    for delta_nominal, res in step_results.items():
        y = res.eps_actual.ravel()

        for name, pred in [
            ("linear", res.pred_linear.ravel()),
            ("exp", res.pred_exp.ravel()),
        ]:
            r2 = log_space_r2(y, pred)
            med, p10, p90 = ratio_stats(y, pred)

            print(
                f"DeltaTe={delta_nominal:+.2f} eV | {name:6s} predictor | "
                f"logR2={r2: .4f} | actual/pred median={med: .3f} "
                f"[p10={p10: .3f}, p90={p90: .3f}]"
            )


def save_outputs(weights, sensitivity, step_results):
    os.makedirs(str(_SENS), exist_ok=True)

    save_dict = {
        "TE_GRID": np.asarray(TE_GRID),
        "NE_GRID": np.asarray(NE_GRID),
        "a_Ha": sensitivity.a_Ha,
        "S_Ha": sensitivity.S_Ha,
        "I_Ha_grid": sensitivity.I_Ha_grid,
    }

    for label, val in weights.items():
        save_dict[f"A_{label.replace('->', '_to_')}"] = np.asarray(val)

    for delta_nominal, res in step_results.items():
        key = str(delta_nominal).replace("-", "m").replace(".", "p")
        save_dict[f"eps_actual_DTe_{key}"] = res.eps_actual
        save_dict[f"pred_linear_DTe_{key}"] = res.pred_linear
        save_dict[f"pred_exp_DTe_{key}"] = res.pred_exp
        save_dict[f"actual_delta_te_DTe_{key}"] = res.actual_delta_te

    np.savez_compressed(
        str(_SENS / "Halpha_sensitivity_results.npz"),
        **save_dict,
    )


# -----------------------------
# Main
# -----------------------------

def main():
    os.makedirs(str(_SENS), exist_ok=True)
    os.makedirs(str(_FIG), exist_ok=True)

    print("Loading CR matrix grids...")
    L_grid, S_grid_src = load_grids()
    print(f"L_grid shape: {L_grid.shape}")
    print(f"S_grid shape: {S_grid_src.shape}")
    print()

    print("Estimating/loading H-alpha radiative weights...")
    weights = estimate_halpha_weights_from_L(L_grid)
    print_halpha_weights(weights)

    print("Computing H-alpha QSS sensitivity...")
    sensitivity = compute_halpha_sensitivity(L_grid, S_grid_src, weights)

    delta_te_list = [0.3, 0.6, 1.0, 2.0, 3.0]
    print(f"Computing H-alpha step mismatch for DeltaTe = {delta_te_list} eV...")
    step_results = compute_halpha_step_results(delta_te_list, sensitivity)

    print_metrics(sensitivity, step_results)

    print()
    print("Saving arrays...")
    save_outputs(weights, sensitivity, step_results)

    print("Generating figures...")
    plot_halpha_sensitivity_map(sensitivity)
    plot_collapse(
        step_results,
        predictor_name="pred_linear",
        outfile_stem="Halpha_sensitivity_collapse_linear",
    )
    plot_collapse(
        step_results,
        predictor_name="pred_exp",
        outfile_stem="Halpha_sensitivity_collapse_exp",
    )

    print()
    print("Done.")
    print("Saved figures:")
    print("  figures/Halpha_sensitivity_map.png")
    print("  figures/Halpha_sensitivity_collapse_linear.png")
    print("  figures/Halpha_sensitivity_collapse_exp.png")
    print()
    print("Interpretation:")
    print("  Use the exponential predictor if actual/pred_exp clusters near 1.")
    print("  If H-alpha collapse is weaker than global collapse, that means the")
    print("  global max-norm manifold error and observable Balmer error are governed")
    print("  by different state sensitivities.")


if __name__ == "__main__":
    main()
