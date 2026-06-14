"""
S_criterion.py
==============
Compute the QSS manifold sensitivity criterion:

    S(Te, ne) = max_p | d/dTe ln r_p^QSS(Te, ne) |

where r_p^QSS = n_p^QSS / n_1s^QSS is the population ratio of state p
to the ground state under QSS conditions.

Then test whether:

    eps_step ≈ S(Te, ne) * |DeltaTe|

for multiple step sizes. If this collapses, S is the mechanistic predictor
that replaces the empirical fit.

OUTPUTS
-------
data/processed/sensitivity/S_grid.npy          (50, 8) — S values
data/processed/sensitivity/p_max_grid.npy      (50, 8) — dominant state index
figures/S_criterion_map.png                    — Fig 3a heatmap
figures/S_criterion_collapse.png               — Fig 3b scatter

USAGE
-----
    cd src/rates
    python S_criterion.py
"""

import numpy as np
import os, sys, pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_SENS = _REPO / 'data/processed/sensitivity'
_FIG  = _REPO / 'figures'

sys.path.insert(0, str(_HERE))
from assemble_cr_matrix import TE_GRID, NE_GRID, build_L

N_ION = 1e14   # cm^-3
N_TE, N_NE = len(TE_GRID), len(NE_GRID)
DTE_FD = 0.05  # eV — finite difference step for dS/dTe (small enough to be accurate)


def steady_state(L, S_src, n_ion=N_ION):
    """Solve L n^ss = -S_src * n_ion. L is full rank."""
    n_ss = np.linalg.solve(L, -S_src * n_ion)
    return np.where(n_ss < 0, 0.0, n_ss)


def population_ratios(n_ss):
    """r_p = n_p / n_1s (ratio to ground state)."""
    if n_ss[0] <= 0:
        return np.zeros_like(n_ss)
    return n_ss / n_ss[0]


def compute_S_grid():
    """
    Compute S(Te, ne) at all 400 grid points via central finite differences.
    S = max_p |d/dTe ln r_p^QSS| = max_p |(1/r_p) dr_p/dTe|
    """
    L_grid = np.load(str(_REPO / 'data/processed/cr_matrix/L_grid.npy'))
    S_grid_src = np.load(str(_REPO / 'data/processed/cr_matrix/S_grid.npy'))

    S_val  = np.zeros((N_TE, N_NE))
    p_max  = np.zeros((N_TE, N_NE), dtype=int)

    for ti in range(N_TE):
        for ni in range(N_NE):
            Te = TE_GRID[ti]

            # Need L at Te+dTe and Te-dTe
            # Build by interpolation from grid, or use nearest available
            # Central difference: use ti+1 and ti-1 if available
            if ti == 0:
                ti_hi, ti_lo = 1, 0
                dTe = TE_GRID[1] - TE_GRID[0]
            elif ti == N_TE - 1:
                ti_hi, ti_lo = N_TE-1, N_TE-2
                dTe = TE_GRID[-1] - TE_GRID[-2]
            else:
                ti_hi, ti_lo = ti+1, ti-1
                dTe = TE_GRID[ti+1] - TE_GRID[ti-1]

            L_hi = L_grid[ti_hi, ni]
            S_hi = S_grid_src[ti_hi, ni]
            L_lo = L_grid[ti_lo, ni]
            S_lo = S_grid_src[ti_lo, ni]

            n_hi = steady_state(L_hi, S_hi)
            n_lo = steady_state(L_lo, S_lo)

            r_hi = population_ratios(n_hi)
            r_lo = population_ratios(n_lo)

            # d ln r_p / dTe = (r_hi - r_lo) / (dTe * r_mid)
            # Use r_mid from current grid point
            L_mid = L_grid[ti, ni]
            S_mid = S_grid_src[ti, ni]
            n_mid = steady_state(L_mid, S_mid)
            r_mid = population_ratios(n_mid)

            # Avoid division by zero for tiny populations
            mask = r_mid > 1e-30
            d_ln_r = np.zeros(43)
            d_ln_r[mask] = (r_hi[mask] - r_lo[mask]) / (dTe * r_mid[mask])

            # Exclude ground state (p=0) — ratio = 1 always
            d_ln_r[0] = 0.0

            S_val[ti, ni]  = np.max(np.abs(d_ln_r))
            p_max[ti, ni]  = np.argmax(np.abs(d_ln_r))

    return S_val, p_max


def compute_eps_step_grid(DeltaTe_list):
    """
    For each (te_old, ne) and step DeltaTe, compute eps_step.
    eps_step = max_p |n_p^QSS(old) - n_p^QSS(new)| / n_p^QSS(new)

    Returns dict: DeltaTe -> (N_TE, N_NE) array of eps_step values
    """
    L_grid = np.load(str(_REPO / 'data/processed/cr_matrix/L_grid.npy'))
    S_grid_src = np.load(str(_REPO / 'data/processed/cr_matrix/S_grid.npy'))

    results = {}
    for DeltaTe in DeltaTe_list:
        eps = np.zeros((N_TE, N_NE))
        for ti in range(N_TE):
            Te_old = TE_GRID[ti]
            Te_new = Te_old + DeltaTe
            if Te_new > TE_GRID[-1] or Te_new < TE_GRID[0]:
                eps[ti, :] = np.nan
                continue
            ti_new = int(np.argmin(np.abs(TE_GRID - Te_new)))

            for ni in range(N_NE):
                L_old = L_grid[ti, ni]
                S_old = S_grid_src[ti, ni]
                L_new = L_grid[ti_new, ni]
                S_new = S_grid_src[ti_new, ni]

                n_old = steady_state(L_old, S_old)
                n_new = steady_state(L_new, S_new)

                # eps_step = max over excited states
                mask = n_new[1:] > 1e-30
                err = np.zeros(42)
                err[mask] = np.abs(n_old[1:][mask] - n_new[1:][mask]) / n_new[1:][mask]
                eps[ti, ni] = np.max(err)
        results[DeltaTe] = eps
    return results


def plot_S_map(S_val, p_max):
    """Fig 3a: S(Te, ne) heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11})

    # Panel (a): S heatmap
    ax = axes[0]
    im = ax.pcolormesh(NE_GRID, TE_GRID, S_val,
                       cmap='hot_r', shading='auto',
                       vmin=0, vmax=np.nanpercentile(S_val, 95))
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r'$\mathcal{S}(T_e, n_e)$ [eV$^{-1}$]')
    ax.set_xscale('log')
    ax.set_xlabel(r'$n_e$ [cm$^{-3}$]')
    ax.set_ylabel(r'$T_e$ [eV]')
    ax.set_title(r'(a) QSS manifold sensitivity $\mathcal{S} = \max_p |\partial_{T_e} \ln r_p^{\rm QSS}|$')

    # Star at ITER reference
    ti_r = int(np.argmin(np.abs(TE_GRID - 3.0)))
    ni_r = int(np.argmin(np.abs(NE_GRID - 1.39e14)))
    ax.plot(NE_GRID[ni_r], TE_GRID[ti_r], 'c*', ms=14, label='ITER ref')
    ax.legend(fontsize=9)

    # Panel (b): dominant state index
    ax2 = axes[1]
    im2 = ax2.pcolormesh(NE_GRID, TE_GRID, p_max,
                          cmap='tab20', shading='auto',
                          vmin=0, vmax=42)
    cb2 = fig.colorbar(im2, ax=ax2)
    cb2.set_label('Dominant state index $p_{\\rm max}$')
    ax2.set_xscale('log')
    ax2.set_xlabel(r'$n_e$ [cm$^{-3}$]')
    ax2.set_ylabel(r'$T_e$ [eV]')
    ax2.set_title(r'(b) State $p$ maximising $|\partial_{T_e} \ln r_p^{\rm QSS}|$')
    ax2.plot(NE_GRID[ni_r], TE_GRID[ti_r], 'c*', ms=14)

    plt.tight_layout()
    os.makedirs(str(_FIG), exist_ok=True)
    for ext in ['pdf', 'png']:
        fig.savefig(str(_FIG / f'S_criterion_map.{ext}'), dpi=150, bbox_inches='tight')
    print("Saved: figures/S_criterion_map.{pdf,png}")
    plt.close(fig)


def plot_collapse(S_val, eps_dict):
    """Fig 3b: scatter eps_step vs S*|DeltaTe| for multiple step sizes."""
    fig, ax = plt.subplots(figsize=(7, 6))
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11,
                         'axes.grid': True, 'grid.alpha': 0.3})

    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    markers = ['o', 's', '^', 'D', 'v']

    for k, (DeltaTe, eps) in enumerate(eps_dict.items()):
        x = (S_val * abs(DeltaTe)).ravel()
        y = eps.ravel()
        valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        ax.scatter(x[valid], y[valid],
                   c=colors[k % len(colors)],
                   marker=markers[k % len(markers)],
                   s=15, alpha=0.5,
                   label=rf'$\Delta T_e = {DeltaTe:+.1f}$ eV')

    # 1:1 line
    lims = [1e-4, 10]
    ax.plot(lims, lims, 'k--', lw=1.5, label='$y = x$ (perfect prediction)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\mathcal{S}(T_e, n_e)\,|\Delta T_e|$ [dimensionless]')
    ax.set_ylabel(r'$\varepsilon_{\rm step}$ [dimensionless]')
    ax.set_title(r'QSS transient error predicted by manifold sensitivity'
                 '\n'
                 r'$\varepsilon_{\rm step} \approx \mathcal{S} |\Delta T_e|$')
    ax.legend(fontsize=9, loc='upper left')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(str(_FIG / f'S_criterion_collapse.{ext}'), dpi=150, bbox_inches='tight')
    print("Saved: figures/S_criterion_collapse.{pdf,png}")
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(str(_SENS), exist_ok=True)

    print("Computing S(Te, ne) grid via finite differences...")
    S_val, p_max = compute_S_grid()
    np.save(str(_SENS / 'S_grid.npy'), S_val)
    np.save(str(_SENS / 'p_max_grid.npy'), p_max)

    ti_r = int(np.argmin(np.abs(TE_GRID - 3.0)))
    ni_r = int(np.argmin(np.abs(NE_GRID - 1.39e14)))
    print(f"S at ITER ref (Te=3eV, ne=1.39e14): {S_val[ti_r, ni_r]:.4f} eV^-1")
    print(f"Dominant state: index {p_max[ti_r, ni_r]}")
    print()

    # Step sizes to test: positive (heating) and negative (cooling)
    DeltaTe_list = [0.3, 0.6, 1.0, 2.0, 3.0]
    print(f"Computing eps_step for DeltaTe = {DeltaTe_list} eV...")
    eps_dict = compute_eps_step_grid(DeltaTe_list)

    # Report S*|DeltaTe| vs eps_step at ITER reference for each step
    print()
    print("S*|DeltaTe| vs eps_step at ITER reference (Te=3eV, ne=1.39e14):")
    print(f"  S = {S_val[ti_r, ni_r]:.4f} eV^-1")
    print(f"  {'DeltaTe':>10}  {'S*|DTe|':>12}  {'eps_step':>12}  {'ratio':>8}")
    for DeltaTe in DeltaTe_list:
        pred = S_val[ti_r, ni_r] * abs(DeltaTe)
        eps  = eps_dict[DeltaTe][ti_r, ni_r]
        ratio = eps / pred if pred > 0 else float('nan')
        print(f"  {DeltaTe:>+10.1f}  {pred:>12.4f}  {eps:>12.4f}  {ratio:>8.3f}")

    print()
    print("Generating figures...")
    plot_S_map(S_val, p_max)
    plot_collapse(S_val, eps_dict)
    print()
    print("Done. Check figures/S_criterion_*.png")
    print()
    print("KEY QUESTION: Does eps_step / (S*|DeltaTe|) cluster near 1.0?")
    print("If yes -> S is the mechanistic predictor. Paper is proven.")
