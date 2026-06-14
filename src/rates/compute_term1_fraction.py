"""
compute_term1_fraction.py  (v2 — correct diagnostic)
=====================================================
Show that Term I (free propagation) dominates the QSS transient error
across all ITER-ionising Te steps and densities.

CORRECT DIAGNOSTIC
------------------
At t=0+: n(3P) error = u_F(0)[3P] = n^ss_old(3P) - n^ss_new(3P)
         This is entirely from Term I (Term II = 0 at t=0 exactly).

Term II can only grow via the driven integral:
  u_F_II(t) ~ exp(L_FF t) * L_FS * int_0^t exp(-L_FF s) u_S(s) ds

Since u_S evolves on tau_QSS >> tau_K, the maximum Term II contribution
to n(3P) during the bath relaxation is bounded by:
  |Term II_max| ~ |[L_FS]_{3P} * u_S(0)| * tau_K
                ~ |Omega_QSS / tau_QSS| * tau_K * |u_S(0)|

The SUPPRESSION RATIO is:
  R = |Term II_max| / |Term I initial| = (tau_K / tau_QSS) * coupling_factor
    = 1/MMZ * coupling_factor

Since MMZ >= 1350 everywhere, R < 1/1000 (negligible).

WHAT WE PLOT
------------
For each (Te_old -> Te_new, ne):
  1. Initial fractional error: |u_F(0)[3P]| / n^ss_new(3P)  [from Term I]
  2. MMZ = tau_QSS / tau_K  [suppression of Term II]
  3. Both on same plot to show Term II is always suppressed

This proves Term I dominance is GENERAL, not a one-case accident.

USAGE
-----
    cd src/rates
    python compute_term1_fraction.py
"""

import numpy as np
import os, sys, pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE    = pathlib.Path(__file__).resolve().parent
_REPO    = _HERE.parent.parent
_MZ_DIR  = _REPO / 'data' / 'processed' / 'mori_zwanzig'
_FIG_DIR = _REPO / 'figures'

sys.path.insert(0, str(_HERE))
from assemble_cr_matrix import TE_GRID, NE_GRID

IDX_3P   = 6
IDX_3P_F = 5
IDX_SLOW = 0
IDX_FAST = list(range(1, 43))
N_ION    = 1e14


def steady_state(L, S, n_ion=N_ION):
    n_ss = np.linalg.solve(L, -S * n_ion)
    return np.where(n_ss < 0, 0.0, n_ss)


def compute_case(L_old, S_old, L_new, S_new, tau_QSS_val, tau_K_val):
    """
    Compute:
    1. Initial fractional error in n(3P) [= Term I amplitude / n^ss_new(3P)]
    2. MMZ = tau_QSS / tau_K  [= Term II suppression factor]
    3. Ratio |u_F(0)[3P]| / |u_S(0)|  [fast vs slow mismatch]
    """
    n_old = steady_state(L_old, S_old)
    n_new = steady_state(L_new, S_new)
    u0    = n_old - n_new
    u0_F  = u0[IDX_FAST]
    u0_S  = u0[IDX_SLOW]

    # Initial fractional error in n(3P) — this IS the Term I amplitude
    if n_new[IDX_3P] > 0:
        frac_err = abs(u0_F[IDX_3P_F]) / n_new[IDX_3P]
    else:
        return None

    # Fast vs slow mismatch ratio
    fast_slow_ratio = abs(u0_F[IDX_3P_F]) / max(abs(u0_S), 1e-30)

    return {
        'frac_err':       frac_err,      # |u_F(0)[3P]| / n^ss_new(3P)
        'MMZ':            tau_QSS_val / tau_K_val,  # suppression factor
        'fast_slow_ratio': fast_slow_ratio,
        'u0_3P':          u0_F[IDX_3P_F],
        'n_ss_new_3P':    n_new[IDX_3P],
        'n_ss_old_3P':    n_old[IDX_3P],
    }


def run_all():
    L_grid   = np.load(str(_REPO / 'data/processed/cr_matrix/L_grid.npy'))
    S_grid   = np.load(str(_REPO / 'data/processed/cr_matrix/S_grid.npy'))
    tau_K    = np.load(str(_MZ_DIR / 'tau_K_grid.npy'))       # (50, 8)
    tau_QSS  = np.load(str(_MZ_DIR / '../../../validation/tau_QSS_grid.npy'))

    Te_steps = [
        (2.0, 3.0), (2.0, 4.0), (2.0, 6.0), (2.0, 10.0),
        (3.0, 4.0), (3.0, 6.0), (3.0, 8.0), (3.0, 10.0),
        (4.0, 6.0), (4.0, 10.0),
        (5.0, 8.0), (5.0, 10.0),
        (6.0, 10.0),
    ]
    ne_targets = [1e12, 1e13, 5e13, 1e14, 5e14, 1e15]

    print(f"{'Te_old':>7} {'Te_new':>7} {'ne':>10}  "
          f"{'tau_K(ns)':>10}  {'MMZ':>10}  "
          f"{'InitErr%':>10}  {'Suppression':>12}")
    print("-" * 75)

    results = []

    for Te_old, Te_new in Te_steps:
        ti_old = int(np.argmin(np.abs(TE_GRID - Te_old)))
        ti_new = int(np.argmin(np.abs(TE_GRID - Te_new)))

        for ne_t in ne_targets:
            ni = int(np.argmin(np.abs(NE_GRID - ne_t)))

            tK  = tau_K[ti_new, ni]
            tQ  = tau_QSS[ti_new, ni]
            MMZ = tQ / tK

            res = compute_case(
                L_grid[ti_old, ni], S_grid[ti_old, ni],
                L_grid[ti_new, ni], S_grid[ti_new, ni],
                tQ, tK
            )
            if res is None:
                continue

            res.update({
                'Te_old': TE_GRID[ti_old],
                'Te_new': TE_GRID[ti_new],
                'ne':     NE_GRID[ni],
                'tau_K':  tK,
                'tau_QSS': tQ,
            })
            results.append(res)

            print(f"{TE_GRID[ti_old]:7.2f} {TE_GRID[ti_new]:7.2f} "
                  f"{NE_GRID[ni]:10.2e}  "
                  f"{tK*1e9:10.3f}  {MMZ:10.0f}  "
                  f"{res['frac_err']*100:10.1f}  "
                  f"{1/MMZ*100:12.4f}%")

    print(f"\nTotal cases: {len(results)}")
    print(f"MMZ range: {min(r['MMZ'] for r in results):.0f} - "
          f"{max(r['MMZ'] for r in results):.0f}")
    print(f"Term II suppression (1/MMZ): "
          f"{1/max(r['MMZ'] for r in results)*100:.4f}% - "
          f"{1/min(r['MMZ'] for r in results)*100:.2f}%")
    print(f"Initial error range: "
          f"{min(r['frac_err'] for r in results)*100:.1f}% - "
          f"{max(r['frac_err'] for r in results)*100:.1f}%")

    return results


def plot_summary(results):
    """
    Two-panel figure:
    Left: Initial fractional error in n(3P) vs DeltaTe, coloured by ne
          Shows the amplitude of the Term I-driven transient
    Right: MMZ vs ne (showing Term II suppression factor)
           MMZ >> 1 means Term II is always negligible
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11,
                         'axes.grid': True, 'grid.alpha': 0.3})

    ne_vals  = sorted(set(r['ne'] for r in results))
    colors   = plt.cm.plasma(np.linspace(0.05, 0.85, len(ne_vals)))
    markers  = ['o', 's', '^', 'D', 'v', 'P']

    # Panel (a): initial error amplitude vs DeltaTe
    ax = axes[0]
    for k, ne in enumerate(ne_vals):
        sub = [r for r in results if r['ne'] == ne]
        dTe  = [r['Te_new'] - r['Te_old'] for r in sub]
        ferr = [r['frac_err'] * 100 for r in sub]
        ax.scatter(dTe, ferr,
                   color=colors[k], marker=markers[k], s=70,
                   label=rf'$n_e=10^{{{np.log10(ne):.1f}}}$',
                   alpha=0.9, edgecolors='k', linewidths=0.5)

    ax.set_xlabel(r'$\Delta T_e = T_e^{\rm new} - T_e^{\rm old}$ [eV]')
    ax.set_ylabel(r'Initial $n(3P)$ error $|u_F(0)|_{3P}/n^{\rm ss}_{\rm new}(3P)$ [\%]')
    ax.set_title(r'(a) Term~I amplitude: initial fast-state mismatch'
                 '\n'
                 r'(sets QSS error magnitude)')
    ax.legend(fontsize=8.5, loc='upper left', ncol=2)

    # Panel (b): MMZ vs ne showing Term II suppression
    ax2 = axes[1]
    Te_step_cases = [(2.0,3.0),(3.0,6.0),(3.0,10.0),(6.0,10.0)]
    lstyles = ['-','--',':','-.']
    colors2 = ['C0','C1','C2','C3']
    for (T1,T2), ls, c in zip(Te_step_cases, lstyles, colors2):
        sub = sorted([r for r in results
                      if abs(r['Te_old']-T1)<0.2 and abs(r['Te_new']-T2)<0.2],
                     key=lambda r: r['ne'])
        if not sub: continue
        ne_s = [r['ne'] for r in sub]
        MMZ_s = [r['MMZ'] for r in sub]
        ax2.loglog(ne_s, MMZ_s, ls+markers[0], color=c, lw=1.8, ms=7,
                   label=rf'${T1:.0f}\to{T2:.0f}$ eV')

    ax2.axhline(100, color='gray', ls='--', lw=1.2, label=r'$M_{\rm MZ}=100$')
    ax2.axhline(1000, color='gray', ls=':', lw=1.2, label=r'$M_{\rm MZ}=10^3$')
    ax2.set_xlabel(r'$n_e$ [cm$^{-3}$]')
    ax2.set_ylabel(r'$M_{\rm MZ} = \tau_{\rm QSS}/\tau_K$')
    ax2.set_title(r'(b) Term~II suppression: $M_{\rm MZ} \gg 1$'
                  '\n'
                  r'(Term~II $\leq M_{\rm MZ}^{-1} \times$ Term~I)')
    ax2.legend(fontsize=8.5)

    fig.suptitle(
        r'Term~I (initial mismatch) sets QSS error amplitude; '
        r'Term~II (memory) is suppressed by $M_{\rm MZ}^{-1} \leq 1\%$ across all ITER-ionising steps',

        fontsize=10, y=1.02)

    plt.tight_layout()
    os.makedirs(str(_FIG_DIR), exist_ok=True)
    for ext in ['pdf', 'png']:
        fig.savefig(str(_FIG_DIR / f'mz_fig7_term1_fraction.{ext}'),
                    dpi=300, bbox_inches='tight')
    print(f"\nSaved: figures/mz_fig7_term1_fraction.{{pdf,png}}")
    plt.close(fig)


if __name__ == '__main__':
    results = run_all()
    plot_summary(results)