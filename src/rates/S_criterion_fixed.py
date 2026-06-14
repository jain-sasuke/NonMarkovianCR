"""
S_criterion_fixed.py
====================

Corrected sensitivity analysis for QSS manifold mismatch.

This script fixes the earlier S_criterion.py issues:

1. eps_step is computed using population ratios:
       r_p = n_p / n_1s
   not absolute populations n_p.

2. The signed derivative is computed as:
       a_p = d/dTe ln(r_p)
   using log finite differences:
       a_p ~= [ln r_p(Te_hi) - ln r_p(Te_lo)] / (Te_hi - Te_lo)

3. The code uses the actual grid step:
       DeltaTe_actual = TE_GRID[ti_new] - TE_GRID[ti_old]
   not only the nominal requested DeltaTe.

4. It compares two predictors:
   - linear predictor:
       eps_lin = max_p |a_p * DeltaTe_actual|
   - finite-step exponential predictor:
       eps_exp = max_p |exp(-a_p * DeltaTe_actual) - 1|

5. It saves:
   data/processed/sensitivity/S_grid_fixed.npy
   data/processed/sensitivity/a_grid_fixed.npy
   data/processed/sensitivity/p_max_grid_fixed.npy
   data/processed/sensitivity/S_criterion_fixed_results.npz

6. It creates:
   figures/S_criterion_fixed_map.png/.pdf
   figures/S_criterion_fixed_collapse_linear.png/.pdf
   figures/S_criterion_fixed_collapse_exp.png/.pdf

Run:
   cd ~/Desktop/non_markovian_cr/src/rates
   python S_criterion_fixed.py
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
# Constants
# -----------------------------

N_ION = 1e14          # cm^-3; keep same convention as original script
MIN_RATIO = 1e-300    # avoid log(0)
NEG_REL_TOL = 1e-10
NEG_ABS_TOL = 1e-300

L_LETTERS = ["S", "P", "D", "F", "G", "H", "I", "K"]


# -----------------------------
# Utilities
# -----------------------------

def build_state_labels(n_states: int):
    """
    Assumes ordering:
      n=1..8 l-resolved: 1S, 2S, 2P, 3S, 3P, 3D, ...
      n=9..15 bundled.
    If this does not match the repo ordering, update this function.
    """
    labels = []

    for n in range(1, 9):
        for ell in range(n):
            if ell < len(L_LETTERS):
                labels.append(f"{n}{L_LETTERS[ell]}")
            else:
                labels.append(f"n={n},l={ell}")

    for n in range(9, 16):
        labels.append(f"n={n}")

    if len(labels) != n_states:
        return [f"p={i}" for i in range(n_states)]

    return labels


def load_grids():
    L_grid = np.load(str(_REPO / "data/processed/cr_matrix/L_grid.npy"))
    S_grid_src = np.load(str(_REPO / "data/processed/cr_matrix/S_grid.npy"))

    if L_grid.ndim != 4:
        raise ValueError(f"L_grid should be 4D, got shape {L_grid.shape}")
    if S_grid_src.ndim != 3:
        raise ValueError(f"S_grid should be 3D, got shape {S_grid_src.shape}")

    n_te, n_ne, n_states, n_states_2 = L_grid.shape
    if n_states != n_states_2:
        raise ValueError(f"L matrices must be square, got {n_states} x {n_states_2}")

    if S_grid_src.shape != (n_te, n_ne, n_states):
        raise ValueError(
            f"S_grid shape mismatch: expected {(n_te, n_ne, n_states)}, got {S_grid_src.shape}"
        )

    if n_te != len(TE_GRID):
        raise ValueError(f"TE_GRID length mismatch: {len(TE_GRID)} vs L_grid {n_te}")
    if n_ne != len(NE_GRID):
        raise ValueError(f"NE_GRID length mismatch: {len(NE_GRID)} vs L_grid {n_ne}")

    return L_grid, S_grid_src


def steady_state(L, S_src, n_ion=N_ION):
    """
    Solve:
        L n_ss = -S_src * n_ion

    Small negative values can appear from numerical roundoff.
    Large negative values are warned and then clipped, because otherwise log ratios break.
    """
    n_ss = np.linalg.solve(L, -S_src * n_ion)

    if not np.all(np.isfinite(n_ss)):
        raise FloatingPointError("Non-finite values in steady-state solution.")

    scale = max(float(np.max(np.abs(n_ss))), 1.0)
    neg_tol = max(NEG_ABS_TOL, NEG_REL_TOL * scale)
    min_val = float(np.min(n_ss))

    if min_val < -neg_tol:
        print(
            f"WARNING: significant negative steady-state population: min={min_val:.3e}, "
            f"tol={neg_tol:.3e}. Clipping for diagnostic calculation."
        )

    n_ss = np.where(n_ss < 0.0, 0.0, n_ss)
    return n_ss


def population_ratios(n_ss):
    """
    Return:
        r_p = n_p / n_1s
    """
    if n_ss[0] <= 0 or not np.isfinite(n_ss[0]):
        return np.full_like(n_ss, np.nan, dtype=float)

    r = n_ss / n_ss[0]
    r[0] = 1.0
    return r


def derivative_bracket(ti, n_te):
    """
    Return indices (lo, hi) for finite-difference derivative around ti.
    Uses forward/backward at boundaries and central inside.
    """
    if ti == 0:
        return 0, 1
    if ti == n_te - 1:
        return n_te - 2, n_te - 1
    return ti - 1, ti + 1


def nearest_new_index(ti, delta_te_nominal):
    """
    Find nearest grid index to TE_GRID[ti] + nominal DeltaTe.
    Return:
        ti_new, actual_delta_te
    Return (None, nan) if outside grid.
    """
    te_old = float(TE_GRID[ti])
    te_target = te_old + float(delta_te_nominal)

    if te_target < float(TE_GRID[0]) or te_target > float(TE_GRID[-1]):
        return None, np.nan

    ti_new = int(np.argmin(np.abs(TE_GRID - te_target)))
    actual_delta = float(TE_GRID[ti_new] - TE_GRID[ti])

    if ti_new == ti:
        # Requested step is too small relative to grid spacing.
        # Mark invalid because eps would be zero and misleading.
        return None, np.nan

    return ti_new, actual_delta


def log_space_r2(y_true, y_pred):
    """
    R2 in log10 space against y_pred.
    Returns nan if insufficient valid points.
    """
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
    """
    Return median, p10, p90 of y_true / y_pred over finite positive points.
    """
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
# Core calculations
# -----------------------------

@dataclass
class SensitivityResult:
    a_grid: np.ndarray       # signed derivative a_p = d ln r_p / dTe, shape (N_TE,N_NE,N_STATES)
    S_grid: np.ndarray       # max_p |a_p| over excited states, shape (N_TE,N_NE)
    p_max_grid: np.ndarray   # dominant state index, shape (N_TE,N_NE)


@dataclass
class StepResult:
    delta_nominal: float
    eps_actual: np.ndarray
    pred_linear: np.ndarray
    pred_exp: np.ndarray
    actual_delta_te: np.ndarray
    p_actual_max: np.ndarray
    p_exp_max: np.ndarray


def compute_signed_sensitivity_grid(L_grid, S_grid_src):
    """
    Compute signed derivative:
        a_p(Te,ne) = d/dTe ln r_p^QSS

    Then:
        S = max over excited states of |a_p|
    """
    n_te, n_ne, n_states, _ = L_grid.shape

    a_grid = np.full((n_te, n_ne, n_states), np.nan, dtype=float)
    S_grid = np.full((n_te, n_ne), np.nan, dtype=float)
    p_max_grid = np.full((n_te, n_ne), -1, dtype=int)

    for ti in range(n_te):
        ti_lo, ti_hi = derivative_bracket(ti, n_te)
        dte = float(TE_GRID[ti_hi] - TE_GRID[ti_lo])

        for ni in range(n_ne):
            n_lo = steady_state(L_grid[ti_lo, ni], S_grid_src[ti_lo, ni])
            n_hi = steady_state(L_grid[ti_hi, ni], S_grid_src[ti_hi, ni])

            r_lo = population_ratios(n_lo)
            r_hi = population_ratios(n_hi)

            a = np.full(n_states, np.nan, dtype=float)

            valid = (
                np.isfinite(r_lo)
                & np.isfinite(r_hi)
                & (r_lo > MIN_RATIO)
                & (r_hi > MIN_RATIO)
            )

            a[valid] = (np.log(r_hi[valid]) - np.log(r_lo[valid])) / dte

            # Ground-state ratio is exactly 1 by definition, exclude it.
            a[0] = 0.0

            a_grid[ti, ni, :] = a

            excited_abs = np.abs(a[1:])
            if np.any(np.isfinite(excited_abs)):
                local_idx = int(np.nanargmax(excited_abs)) + 1
                S_grid[ti, ni] = float(np.abs(a[local_idx]))
                p_max_grid[ti, ni] = local_idx

    return SensitivityResult(
        a_grid=a_grid,
        S_grid=S_grid,
        p_max_grid=p_max_grid,
    )


def compute_step_results(delta_te_list, L_grid, S_grid_src, sensitivity):
    """
    Compute actual ratio-based eps_step and predictors.

    Actual:
        eps_actual = max_p |r_old - r_new| / r_new

    Linear predictor:
        pred_linear = max_p |a_p * DeltaTe_actual|

    Signed exponential predictor:
        pred_exp = max_p |exp(-a_p * DeltaTe_actual) - 1|

    Important:
        DeltaTe_actual is taken from grid snapping:
        TE_GRID[ti_new] - TE_GRID[ti]
    """
    n_te, n_ne, n_states, _ = L_grid.shape
    results = {}

    for delta_nominal in delta_te_list:
        eps_actual = np.full((n_te, n_ne), np.nan, dtype=float)
        pred_linear = np.full((n_te, n_ne), np.nan, dtype=float)
        pred_exp = np.full((n_te, n_ne), np.nan, dtype=float)
        actual_delta_te = np.full((n_te, n_ne), np.nan, dtype=float)
        p_actual_max = np.full((n_te, n_ne), -1, dtype=int)
        p_exp_max = np.full((n_te, n_ne), -1, dtype=int)

        for ti in range(n_te):
            ti_new, dte_actual = nearest_new_index(ti, delta_nominal)

            if ti_new is None:
                continue

            for ni in range(n_ne):
                actual_delta_te[ti, ni] = dte_actual

                # Actual ratio-based eps_step.
                n_old = steady_state(L_grid[ti, ni], S_grid_src[ti, ni])
                n_new = steady_state(L_grid[ti_new, ni], S_grid_src[ti_new, ni])

                r_old = population_ratios(n_old)
                r_new = population_ratios(n_new)

                valid_r = (
                    np.isfinite(r_old[1:])
                    & np.isfinite(r_new[1:])
                    & (r_new[1:] > MIN_RATIO)
                )

                if np.any(valid_r):
                    err = np.full(n_states - 1, np.nan, dtype=float)
                    err[valid_r] = np.abs(r_old[1:][valid_r] - r_new[1:][valid_r]) / r_new[1:][valid_r]

                    local_idx = int(np.nanargmax(err)) + 1
                    eps_actual[ti, ni] = float(err[local_idx - 1])
                    p_actual_max[ti, ni] = local_idx

                # Predictors from signed derivative at old Te.
                a = sensitivity.a_grid[ti, ni, :].copy()
                a[0] = np.nan

                valid_a = np.isfinite(a)

                if np.any(valid_a):
                    lin_err = np.full(n_states, np.nan, dtype=float)
                    lin_err[valid_a] = np.abs(a[valid_a] * dte_actual)
                    pred_linear[ti, ni] = float(np.nanmax(lin_err))

                    # Avoid overflow in exp.
                    exponent = np.full(n_states, np.nan, dtype=float)
                    exponent[valid_a] = -a[valid_a] * dte_actual
                    exponent = np.clip(exponent, -700.0, 700.0)

                    exp_err = np.full(n_states, np.nan, dtype=float)
                    exp_err[valid_a] = np.abs(np.exp(exponent[valid_a]) - 1.0)

                    local_exp_idx = int(np.nanargmax(exp_err))
                    pred_exp[ti, ni] = float(exp_err[local_exp_idx])
                    p_exp_max[ti, ni] = local_exp_idx

        results[delta_nominal] = StepResult(
            delta_nominal=float(delta_nominal),
            eps_actual=eps_actual,
            pred_linear=pred_linear,
            pred_exp=pred_exp,
            actual_delta_te=actual_delta_te,
            p_actual_max=p_actual_max,
            p_exp_max=p_exp_max,
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


def plot_sensitivity_map(sensitivity, state_labels):
    setup_plot_style()

    S_grid = sensitivity.S_grid
    p_max = sensitivity.p_max_grid

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))

    # Panel A: S heatmap
    ax = axes[0]
    vmax = np.nanpercentile(S_grid, 95)
    im = ax.pcolormesh(
        NE_GRID,
        TE_GRID,
        S_grid,
        cmap="hot_r",
        shading="auto",
        vmin=0.0,
        vmax=vmax,
    )
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("S = max |d ln(r_p)/dTe| [eV^-1]")

    ax.set_xscale("log")
    ax.set_xlabel("ne [cm^-3]")
    ax.set_ylabel("Te [eV]")
    ax.set_title("(a) QSS manifold sensitivity")

    ti_r = int(np.argmin(np.abs(TE_GRID - 3.0)))
    ni_r = int(np.argmin(np.abs(NE_GRID - 1.39e14)))
    ax.plot(NE_GRID[ni_r], TE_GRID[ti_r], "c*", ms=14, label="ITER ref")
    ax.legend(fontsize=9)

    # Panel B: dominant state index
    ax2 = axes[1]
    p_plot = np.ma.masked_where(p_max < 0, p_max)

    im2 = ax2.pcolormesh(
        NE_GRID,
        TE_GRID,
        p_plot,
        cmap="tab20",
        shading="auto",
        vmin=0,
        vmax=max(42, np.nanmax(p_max)),
    )
    cb2 = fig.colorbar(im2, ax=ax2)
    cb2.set_label("dominant state index p_max")

    ax2.set_xscale("log")
    ax2.set_xlabel("ne [cm^-3]")
    ax2.set_ylabel("Te [eV]")
    ax2.set_title("(b) State maximizing |d ln(r_p)/dTe|")
    ax2.plot(NE_GRID[ni_r], TE_GRID[ti_r], "c*", ms=14)

    plt.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(str(_FIG / f"S_criterion_fixed_map.{ext}"), dpi=180, bbox_inches="tight")

    plt.close(fig)


def plot_collapse(step_results, predictor_name, outfile_stem):
    """
    predictor_name:
      'pred_linear' or 'pred_exp'
    """
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
            s=16,
            alpha=0.55,
            label=f"DeltaTe={delta_nominal:+.1f} eV",
        )

    if all_x:
        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)
        finite = (
            np.isfinite(all_x)
            & np.isfinite(all_y)
            & (all_x > 0)
            & (all_y > 0)
        )
        if np.any(finite):
            min_val = min(np.nanmin(all_x[finite]), np.nanmin(all_y[finite]))
            max_val = max(np.nanmax(all_x[finite]), np.nanmax(all_y[finite]))
            lo = 10 ** np.floor(np.log10(min_val))
            hi = 10 ** np.ceil(np.log10(max_val))
        else:
            lo, hi = 1e-4, 10
    else:
        lo, hi = 1e-4, 10

    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="y=x")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    if predictor_name == "pred_linear":
        ax.set_xlabel("linear predictor max |a_p DeltaTe|")
        ax.set_title("Ratio-based eps_step vs linear manifold predictor")
    elif predictor_name == "pred_exp":
        ax.set_xlabel("finite-step predictor max |exp(-a_p DeltaTe)-1|")
        ax.set_title("Ratio-based eps_step vs signed finite-step predictor")
    else:
        ax.set_xlabel(predictor_name)

    ax.set_ylabel("actual ratio-based eps_step")
    ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(str(_FIG / f"{outfile_stem}.{ext}"), dpi=180, bbox_inches="tight")

    plt.close(fig)


# -----------------------------
# Reporting
# -----------------------------

def print_metrics(step_results, state_labels):
    ti_r = int(np.argmin(np.abs(TE_GRID - 3.0)))
    ni_r = int(np.argmin(np.abs(NE_GRID - 1.39e14)))

    print()
    print("=" * 78)
    print("ITER-nearest point")
    print("=" * 78)
    print(f"TE_GRID nearest to 3 eV      : {TE_GRID[ti_r]:.6g} eV")
    print(f"NE_GRID nearest to 1.39e14   : {NE_GRID[ni_r]:.6e} cm^-3")
    print()

    header = (
        f"{'DeltaTe_nom':>12}  {'DeltaTe_act':>12}  "
        f"{'eps_actual':>12}  {'pred_lin':>12}  {'act/lin':>9}  "
        f"{'pred_exp':>12}  {'act/exp':>9}  {'p_actual':>12}  {'p_exp':>12}"
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

        p_act = int(res.p_actual_max[ti_r, ni_r])
        p_exp = int(res.p_exp_max[ti_r, ni_r])

        p_act_label = state_labels[p_act] if 0 <= p_act < len(state_labels) else "NA"
        p_exp_label = state_labels[p_exp] if 0 <= p_exp < len(state_labels) else "NA"

        print(
            f"{delta_nominal:12.3f}  {dta:12.5f}  "
            f"{eps:12.5g}  {lin:12.5g}  {ratio_lin:9.3f}  "
            f"{exp:12.5g}  {ratio_exp:9.3f}  "
            f"{p_act_label:>12}  {p_exp_label:>12}"
        )

    print()
    print("=" * 78)
    print("Global collapse metrics")
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

    print()


def save_outputs(sensitivity, step_results):
    os.makedirs(str(_SENS), exist_ok=True)

    np.save(str(_SENS / "S_grid_fixed.npy"), sensitivity.S_grid)
    np.save(str(_SENS / "a_grid_fixed.npy"), sensitivity.a_grid)
    np.save(str(_SENS / "p_max_grid_fixed.npy"), sensitivity.p_max_grid)

    save_dict = {
        "S_grid": sensitivity.S_grid,
        "a_grid": sensitivity.a_grid,
        "p_max_grid": sensitivity.p_max_grid,
        "TE_GRID": np.asarray(TE_GRID),
        "NE_GRID": np.asarray(NE_GRID),
    }

    for delta_nominal, res in step_results.items():
        key = str(delta_nominal).replace("-", "m").replace(".", "p")
        save_dict[f"eps_actual_DTe_{key}"] = res.eps_actual
        save_dict[f"pred_linear_DTe_{key}"] = res.pred_linear
        save_dict[f"pred_exp_DTe_{key}"] = res.pred_exp
        save_dict[f"actual_delta_te_DTe_{key}"] = res.actual_delta_te
        save_dict[f"p_actual_max_DTe_{key}"] = res.p_actual_max
        save_dict[f"p_exp_max_DTe_{key}"] = res.p_exp_max

    np.savez_compressed(
        str(_SENS / "S_criterion_fixed_results.npz"),
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

    n_te, n_ne, n_states, _ = L_grid.shape
    state_labels = build_state_labels(n_states)

    print(f"L_grid shape: {L_grid.shape}")
    print(f"S_grid shape: {S_grid_src.shape}")
    print(f"Number of states: {n_states}")
    print()

    print("Computing signed QSS manifold sensitivity a_p = d ln(r_p)/dTe ...")
    sensitivity = compute_signed_sensitivity_grid(L_grid, S_grid_src)

    ti_r = int(np.argmin(np.abs(TE_GRID - 3.0)))
    ni_r = int(np.argmin(np.abs(NE_GRID - 1.39e14)))
    p_ref = int(sensitivity.p_max_grid[ti_r, ni_r])
    p_ref_label = state_labels[p_ref] if 0 <= p_ref < len(state_labels) else "NA"

    print(f"S at ITER-nearest point: {sensitivity.S_grid[ti_r, ni_r]:.6g} eV^-1")
    print(f"Dominant S-state index: {p_ref} ({p_ref_label})")
    print()

    # Positive heating steps. Add negative values later if you want cooling tests.
    delta_te_list = [0.3, 0.6, 1.0, 2.0, 3.0]

    print(f"Computing actual ratio-based eps_step and predictors for DeltaTe = {delta_te_list} eV ...")
    step_results = compute_step_results(
        delta_te_list,
        L_grid,
        S_grid_src,
        sensitivity,
    )

    print_metrics(step_results, state_labels)

    print("Saving arrays...")
    save_outputs(sensitivity, step_results)

    print("Generating figures...")
    plot_sensitivity_map(sensitivity, state_labels)
    plot_collapse(
        step_results,
        predictor_name="pred_linear",
        outfile_stem="S_criterion_fixed_collapse_linear",
    )
    plot_collapse(
        step_results,
        predictor_name="pred_exp",
        outfile_stem="S_criterion_fixed_collapse_exp",
    )

    print()
    print("Done.")
    print("Saved figures:")
    print("  figures/S_criterion_fixed_map.png")
    print("  figures/S_criterion_fixed_collapse_linear.png")
    print("  figures/S_criterion_fixed_collapse_exp.png")
    print()
    print("Interpretation rule:")
    print("  If actual/pred_exp clusters closer to 1 than actual/pred_linear,")
    print("  use the signed finite-step exponential predictor in the paper.")
    print("  Do not claim S*|DeltaTe| is exact unless the linear ratio is near 1.")


if __name__ == "__main__":
    main()
