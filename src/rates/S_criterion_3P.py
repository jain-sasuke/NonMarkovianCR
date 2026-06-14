"""
S_criterion_3P.py
=================
QSS manifold sensitivity criterion restricted to the Hα upper level (3P).

RATIONALE
---------
The max-norm S = max_p |∂_Te ln r_p| is dominated by the n=15 bundle
(index 42) — an unobservable, truncation-sensitive state. We instead
define S on the 3P state (index 4), the upper level of the Hα (3→2)
transition. This:
  • ties S directly to the paper's flagship Hα diagnostic result,
  • uses a well-converged resolved state (not the n_max=15 boundary),
  • removes the variable-maximizer artifact (single state, single slope).

DEFINITIONS (all consistent — same r_p on both sides)
-----------------------------------------------------
  r_3P(Te)  = n_3P^QSS(Te) / n_1s^QSS(Te)         population ratio to ground
  a_3P(Te)  = d/dTe ln r_3P                         signed manifold slope [eV^-1]
  S_3P      = |a_3P|                                sensitivity magnitude

  eps_step (ratio-based, normalized to OLD state — matches Taylor link):
            = |r_3P(Te+ΔTe) − r_3P(Te)| / r_3P(Te)

  Predictors:
    linear : |a_3P · ΔTe|                           (ΔTe→0 limit)
    exp    : |exp(a_3P · ΔTe) − 1|                  (exact if ln r linear in Te)

  NOTE on normalization: with eps normalized to the OLD state,
  the exact finite-step predictor is |exp(a·ΔTe) − 1| (heating, ΔTe>0).
  The earlier script normalized to the NEW state, which forces the
  predictor to |exp(−a·ΔTe) − 1| and breaks the S·ΔTe linear link.
  We normalize to OLD so that linear and exp predictors share the ΔTe→0 limit.

CONSISTENCY CHECK
-----------------
At ITER ref (Te≈2.95 eV, ne≈1.39e14), for a heating step ΔTe=+0.6 eV,
the Hα population error should match the corrected thesis result.
The thesis reports eps(3P) at t=0+ = +43.7% for the heating step.
Since eps_step here is the *static QSS-manifold mismatch* (not the
transient t=0+ value), they are related but NOT identical — eps_step
is the asymptotic target gap, the +43.7% is the instantaneous CR-vs-QSS
gap. We print both interpretations and flag the comparison explicitly.

OUTPUTS
-------
data/processed/sensitivity/S_3P_grid.npy        (50,8) |a_3P|
data/processed/sensitivity/a_3P_grid.npy        (50,8) signed a_3P
data/processed/sensitivity/eps_3P_results.npz   eps_step + predictors per ΔTe
figures/S_3P_map.{png,pdf}                       S_3P(Te,ne) heatmap
figures/S_3P_collapse.{png,pdf}                  eps vs predictors

USAGE
-----
    cd ~/Desktop/non_markovian_cr
    PYTHONPATH=. python src/rates/S_criterion_3P.py
"""

import os
import sys
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_SENS = _REPO / "data/processed/sensitivity"
_FIG = _REPO / "figures"

sys.path.insert(0, str(_HERE))
from assemble_cr_matrix import TE_GRID, NE_GRID

# ── Constants ───────────────────────────────────────────────────────────────
N_ION = 1e14            # cm^-3
IDX_1S = 0
IDX_3P = 4              # verified: n=3, l=1 in the 43-state ordering
MIN_POP = 1e-300

# ITER reference (grid-snapped values from context)
TE_REF_NOMINAL = 2.95
NE_REF_NOMINAL = 1.39e14

# Thesis corrected Hα heating-step result (for consistency comparison)
THESIS_HALPHA_HEATING_PERCENT = 43.7   # eps(3P) at t=0+, ΔTe=+0.6 eV


def load_grids():
    L_grid = np.load(str(_REPO / "data/processed/cr_matrix/L_grid.npy"))
    S_src = np.load(str(_REPO / "data/processed/cr_matrix/S_grid.npy"))
    assert L_grid.ndim == 4 and L_grid.shape[2] == L_grid.shape[3] == 43, \
        f"L_grid shape unexpected: {L_grid.shape}"
    assert S_src.shape == L_grid.shape[:3], \
        f"S_grid shape unexpected: {S_src.shape}"
    return L_grid, S_src


def steady_state(L, S_src, n_ion=N_ION):
    """Solve L n = -S_src * n_ion; clip tiny negatives from roundoff."""
    n = np.linalg.solve(L, -S_src * n_ion)
    if not np.all(np.isfinite(n)):
        raise FloatingPointError("non-finite steady state")
    return np.where(n < 0.0, 0.0, n)


def r_3P(n_ss):
    """Population ratio n_3P / n_1s."""
    if n_ss[IDX_1S] <= MIN_POP:
        return np.nan
    return n_ss[IDX_3P] / n_ss[IDX_1S]


def compute_a_3P_grid(L_grid, S_src):
    """
    Signed manifold slope a_3P = d/dTe ln r_3P at every (Te, ne).
    Central finite difference in Te using adjacent grid points, on ln r_3P.
    """
    n_te, n_ne = len(TE_GRID), len(NE_GRID)
    a_grid = np.full((n_te, n_ne), np.nan)

    for ti in range(n_te):
        if ti == 0:
            lo, hi = 0, 1
        elif ti == n_te - 1:
            lo, hi = n_te - 2, n_te - 1
        else:
            lo, hi = ti - 1, ti + 1
        dTe = TE_GRID[hi] - TE_GRID[lo]

        for ni in range(n_ne):
            n_lo = steady_state(L_grid[lo, ni], S_src[lo, ni])
            n_hi = steady_state(L_grid[hi, ni], S_src[hi, ni])
            r_lo = r_3P(n_lo)
            r_hi = r_3P(n_hi)
            if not (np.isfinite(r_lo) and np.isfinite(r_hi)
                    and r_lo > MIN_POP and r_hi > MIN_POP):
                continue
            # d ln r / dTe via log difference (exact for log-linear r)
            a_grid[ti, ni] = (np.log(r_hi) - np.log(r_lo)) / dTe

    return a_grid


def compute_eps_3P(L_grid, S_src, delta_te_list):
    """
    For each nominal ΔTe and each (Te_old, ne):
      eps_step  = |r_old - r_new| / r_old        (normalized to OLD state)
      pred_lin  = |a_3P(Te_old) * ΔTe_actual|
      pred_exp  = |exp(a_3P(Te_old) * ΔTe_actual) - 1|
    Uses grid-snapped Te_new and the ACTUAL ΔTe between grid points.
    Returns dict keyed by nominal ΔTe.
    """
    n_te, n_ne = len(TE_GRID), len(NE_GRID)
    a_grid = compute_a_3P_grid(L_grid, S_src)
    out = {}

    for dte_nom in delta_te_list:
        eps = np.full((n_te, n_ne), np.nan)
        plin = np.full((n_te, n_ne), np.nan)
        pexp = np.full((n_te, n_ne), np.nan)
        dte_act_arr = np.full((n_te, n_ne), np.nan)

        for ti in range(n_te):
            te_target = TE_GRID[ti] + dte_nom
            if te_target < TE_GRID[0] or te_target > TE_GRID[-1]:
                continue
            ti_new = int(np.argmin(np.abs(TE_GRID - te_target)))
            if ti_new == ti:
                continue
            dte_act = TE_GRID[ti_new] - TE_GRID[ti]

            for ni in range(n_ne):
                n_old = steady_state(L_grid[ti, ni], S_src[ti, ni])
                n_new = steady_state(L_grid[ti_new, ni], S_src[ti_new, ni])
                r_old = r_3P(n_old)
                r_new = r_3P(n_new)
                if not (np.isfinite(r_old) and np.isfinite(r_new)
                        and r_old > MIN_POP):
                    continue
                eps[ti, ni] = abs(r_old - r_new) / r_old
                dte_act_arr[ti, ni] = dte_act
                a = a_grid[ti, ni]
                if np.isfinite(a):
                    plin[ti, ni] = abs(a * dte_act)
                    pexp[ti, ni] = abs(np.exp(a * dte_act) - 1.0)

        out[dte_nom] = dict(eps=eps, pred_lin=plin, pred_exp=pexp,
                            dte_act=dte_act_arr)
    return a_grid, out


def ref_indices():
    ti = int(np.argmin(np.abs(TE_GRID - TE_REF_NOMINAL)))
    ni = int(np.argmin(np.abs(NE_GRID - NE_REF_NOMINAL)))
    return ti, ni


def print_ref_table(a_grid, eps_dict, delta_te_list):
    ti, ni = ref_indices()
    a = a_grid[ti, ni]
    print("=" * 74)
    print("3P-RESTRICTED S CRITERION — ITER reference")
    print("=" * 74)
    print(f"  Te_ref = {TE_GRID[ti]:.4f} eV   ne_ref = {NE_GRID[ni]:.3e} cm^-3")
    print(f"  a_3P (signed slope d/dTe ln r_3P) = {a:+.4f} eV^-1")
    print(f"  S_3P = |a_3P| = {abs(a):.4f} eV^-1")
    print(f"  Sign: {'r_3P increases with Te' if a > 0 else 'r_3P DECREASES with Te'}")
    print()
    print("  Per-step comparison (eps normalized to OLD state):")
    print(f"  {'ΔTe_nom':>8}  {'ΔTe_act':>8}  {'eps_step':>9}  "
          f"{'lin=SΔTe':>9}  {'exp pred':>9}  {'eps/lin':>8}  {'eps/exp':>8}")
    for dte in delta_te_list:
        d = eps_dict[dte]
        e = d['eps'][ti, ni]
        pl = d['pred_lin'][ti, ni]
        pe = d['pred_exp'][ti, ni]
        da = d['dte_act'][ti, ni]
        if not np.isfinite(e):
            print(f"  {dte:>+8.2f}  {'--':>8}  (out of grid range)")
            continue
        rl = e / pl if pl > 0 else np.nan
        re = e / pe if pe > 0 else np.nan
        print(f"  {dte:>+8.2f}  {da:>8.3f}  {e:>9.4f}  "
              f"{pl:>9.4f}  {pe:>9.4f}  {rl:>8.3f}  {re:>8.3f}")
    print()
    print("  INTERPRETATION:")
    print("    eps/exp ≈ 1.0 across all ΔTe  → exponential predictor is exact,")
    print("                                    a_3P fully governs the manifold mismatch.")
    print("    eps/lin → 1.0 only as ΔTe→0   → linear S·ΔTe is the small-step limit.")
    print()


def consistency_check(a_grid, eps_dict):
    ti, ni = ref_indices()
    a = a_grid[ti, ni]
    print("=" * 74)
    print("CONSISTENCY CHECK vs corrected thesis Hα result")
    print("=" * 74)
    # The thesis +43.7% is the instantaneous CR-vs-QSS gap at t=0+ (transient).
    # eps_step here is the static QSS(old)-vs-QSS(new) manifold gap.
    # They are different quantities; we report eps_step at ΔTe=+0.6 and the
    # implied 3P slope for cross-comparison.
    dte = 0.6
    if dte in eps_dict:
        e = eps_dict[dte]['eps'][ti, ni]
        da = eps_dict[dte]['dte_act'][ti, ni]
        print(f"  Static QSS manifold mismatch eps_step(3P, ΔTe≈{da:.3f}) "
              f"= {e*100:.1f}%")
    print(f"  Thesis transient eps(3P) at t=0+ (heating, ΔTe=+0.6)   "
          f"= {THESIS_HALPHA_HEATING_PERCENT:.1f}%")
    print()
    print("  These are DIFFERENT quantities and need not match:")
    print("    • eps_step  = gap between OLD and NEW QSS targets (static)")
    print("    • t=0+ value = gap between frozen CR state and NEW QSS (transient)")
    print("  But both are driven by a_3P. Sign of a_3P must be consistent with")
    print("  the thesis observation that n_3P DECREASES as Te increases at ne=1e14.")
    print(f"  → a_3P = {a:+.4f} eV^-1.  Expected NEGATIVE (n_3P falls with Te).")
    if a < 0:
        print("  ✓ Sign CONSISTENT with thesis (heating lowers n_3P → CR overestimates Hα).")
    else:
        print("  ✗ Sign INCONSISTENT — investigate before using in paper.")
    print()
    # Direct check: n_3P at a few Te (thesis table)
    L_grid, S_src = load_grids()
    print("  n_3P^QSS vs Te at ne=1e14 (compare to thesis verified values):")
    print("    thesis: Te=2.0→1.125e7, 3.0→5.137e6, 3.6→3.579e6, 4.0→3.006e6")
    ni14 = int(np.argmin(np.abs(NE_GRID - 1e14)))
    for te_q in [2.0, 3.0, 3.6, 4.0]:
        tq = int(np.argmin(np.abs(TE_GRID - te_q)))
        n = steady_state(L_grid[tq, ni14], S_src[tq, ni14])
        print(f"    Te={TE_GRID[tq]:.2f}: n_3P = {n[IDX_3P]:.3e}")
    print("  (Absolute scale depends on n_ion={:.0e}; check monotonic DECREASE.)"
          .format(N_ION))
    print()


def plot_map(a_grid):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11})
    S3 = np.abs(a_grid)
    im = ax.pcolormesh(NE_GRID, TE_GRID, S3, cmap='viridis', shading='auto',
                       vmin=0, vmax=np.nanpercentile(S3, 97))
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r'$\mathcal{S}_{3P} = |\partial_{T_e}\ln r_{3P}^{\rm QSS}|$  [eV$^{-1}$]')
    ax.set_xscale('log')
    ax.set_xlabel(r'$n_e$ [cm$^{-3}$]')
    ax.set_ylabel(r'$T_e$ [eV]')
    ax.set_title(r'Hα upper-level (3P) QSS manifold sensitivity')
    ti, ni = ref_indices()
    ax.plot(NE_GRID[ni], TE_GRID[ti], 'r*', ms=14, label='ITER ref')
    ax.legend(fontsize=9)
    plt.tight_layout()
    os.makedirs(str(_FIG), exist_ok=True)
    for ext in ['pdf', 'png']:
        fig.savefig(str(_FIG / f'S_3P_map.{ext}'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved: figures/S_3P_map.{pdf,png}")


def plot_collapse(eps_dict, delta_te_list):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11,
                         'axes.grid': True, 'grid.alpha': 0.3})
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    markers = ['o', 's', '^', 'D', 'v']
    lims = [1e-3, 5]

    for ax, key, title in [
        (axes[0], 'pred_lin', r'Linear predictor  $\mathcal{S}_{3P}\,|\Delta T_e|$'),
        (axes[1], 'pred_exp', r'Exp predictor  $|e^{a_{3P}\Delta T_e}-1|$'),
    ]:
        for k, dte in enumerate(delta_te_list):
            d = eps_dict[dte]
            x = d[key].ravel()
            y = d['eps'].ravel()
            v = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            ax.scatter(x[v], y[v], c=colors[k % 5], marker=markers[k % 5],
                       s=14, alpha=0.5, label=rf'$\Delta T_e={dte:+.1f}$')
        ax.plot(lims, lims, 'k--', lw=1.4, label='$y=x$')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('predictor'); ax.set_ylabel(r'$\varepsilon_{\rm step}(3P)$')
        ax.set_title(title)
        ax.legend(fontsize=8, loc='upper left')

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(str(_FIG / f'S_3P_collapse.{ext}'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved: figures/S_3P_collapse.{pdf,png}")


if __name__ == '__main__':
    os.makedirs(str(_SENS), exist_ok=True)
    L_grid, S_src = load_grids()

    print("Computing a_3P(Te, ne) and eps_step(3P) ...")
    delta_te_list = [0.3, 0.6, 1.0, 2.0, 3.0]
    a_grid, eps_dict = compute_eps_3P(L_grid, S_src, delta_te_list)

    np.save(str(_SENS / 'S_3P_grid.npy'), np.abs(a_grid))
    np.save(str(_SENS / 'a_3P_grid.npy'), a_grid)
    np.savez(str(_SENS / 'eps_3P_results.npz'),
             a_grid=a_grid,
             **{f'eps_{d}': eps_dict[d]['eps'] for d in delta_te_list},
             **{f'plin_{d}': eps_dict[d]['pred_lin'] for d in delta_te_list},
             **{f'pexp_{d}': eps_dict[d]['pred_exp'] for d in delta_te_list})
    print()

    print_ref_table(a_grid, eps_dict, delta_te_list)
    consistency_check(a_grid, eps_dict)

    print("Generating figures ...")
    plot_map(a_grid)
    plot_collapse(eps_dict, delta_te_list)
    print()
    print("Done. Key question: in the ref table, does eps/exp ≈ 1.0 at every ΔTe?")
    print("If yes → a_3P is the exact governing slope; linear S·ΔTe is its ΔTe→0 limit.")
