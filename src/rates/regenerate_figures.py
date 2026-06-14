"""
regenerate_figures.py
=====================
Regenerate mz_fig1 and mz_fig3 with correct values.

mz_fig1: Memory kernel + L_FF eigenspectrum at ITER reference
  - Remove tau_relax = 25 ns line (that mode doesn't exist)
  - Fix tau_QSS annotation to 22.7 us

mz_fig3: Single-panel log10(M_MZ) map
  - Remove the two-panel M_thesis vs M_MZ comparison (M_thesis used wrong matrix)
  - Show only M_MZ = tau_QSS / tau_K (single panel, correct)

USAGE: cd src/rates && python regenerate_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pathlib, sys

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_MZ   = _REPO / 'data/processed/mori_zwanzig'
_FIG  = _REPO / 'figures'

sys.path.insert(0, str(_HERE))
from assemble_cr_matrix import TE_GRID, NE_GRID

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.grid': True, 'grid.alpha': 0.3,
    'text.usetex': False,
})

def load_mz():
    tau_K  = np.load(str(_MZ / 'tau_K_grid.npy'))          # (50,8)
    tau_QSS = np.load(str(_MZ / 'tau_relax_MZ.npy'))        # (50,8) -- this is tau_QSS!
    # Actually tau_relax_MZ might be 1/|lambda_1 of L_FF| not tau_QSS
    # Let's compute tau_QSS from L_grid directly
    L_grid = np.load(str(_REPO / 'data/processed/cr_matrix/L_grid.npy'))
    tau_QSS_grid = np.zeros((50, 8))
    for ti in range(50):
        for ni in range(8):
            L = L_grid[ti, ni]
            evals = np.linalg.eigvals(L)
            # tau_QSS = 1/|smallest |Re(lambda)| nonzero|
            re_abs = np.abs(evals.real)
            re_abs_sorted = np.sort(re_abs)
            # Skip the near-zero eigenvalue (should be ~0, but K_ion makes it small)
            tau_QSS_grid[ti, ni] = 1.0 / re_abs_sorted[0]
    M_MZ = tau_QSS_grid / tau_K
    
    # Load eigenvalues of L_FF at ITER reference for Fig 1
    evals_FF = np.load(str(_MZ / 'eigenvalues_FF.npy'))     # (50,8,42)
    K_t = np.load(str(_MZ / 'K_t_ITER_ref.npy'))            # (500,)
    t_arr = np.load(str(_MZ / 't_grid_ITER_ref.npy'))        # (500,)
    
    return tau_K, tau_QSS_grid, M_MZ, evals_FF, K_t, t_arr

def fig1_kernel(tau_K, tau_QSS_grid, evals_FF, K_t, t_arr):
    """Regenerate Fig 1: correct kernel + eigenspectrum."""
    ti, ni = 23, 5   # ITER ref: Te~3eV, ne~1.39e14
    
    tK_ref  = tau_K[ti, ni]
    tQSS_ref = tau_QSS_grid[ti, ni]
    ev_ref  = evals_FF[ti, ni]
    tau_modes = np.sort(1.0 / np.abs(ev_ref.real))[::-1]  # slowest first

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel (a): K(t)/K(0)
    ax = axes[0]
    K_norm = K_t / K_t[0]
    ax.semilogx(t_arr * 1e9, K_norm, 'k-', lw=2.2, label=r'$K(t)/K(0)$')
    ax.axvline(tK_ref * 1e9, color='C0', ls='--', lw=1.8,
               label=rf'$\tau_K = {tK_ref*1e9:.2f}$ ns (bath)')
    ax.axhline(1/np.e, color='gray', ls=':', lw=0.8)
    # Annotate tau_QSS far right
    ax.annotate(rf'$\tau_{{QSS}} = {tQSS_ref*1e6:.1f}\ \mu$s $\rightarrow$',
                xy=(1e3, 0.03), fontsize=9, color='C3',
                ha='right')
    ax.set_xlabel(r'Time $t$ [ns]')
    ax.set_ylabel(r'$K(t)/K(0)$')
    ax.set_title(rf'(a) Memory kernel — ITER reference'
                 '\n'
                 rf'$T_e = 3$ eV, $n_e = 10^{{14}}$ cm$^{{-3}}$')
    ax.legend(fontsize=9)
    ax.set_xlim(t_arr[0]*1e9, 1e4)
    ax.set_ylim(-0.05, 1.05)

    # Panel (b): L_FF eigenspectrum (12 slowest modes)
    ax2 = axes[1]
    n_show = 12
    tau_show = tau_modes[:n_show] * 1e9   # ns
    ax2.bar(range(1, n_show+1), tau_show, color='C0', alpha=0.8)
    ax2.axhline(tK_ref * 1e9, color='C0', ls='--', lw=1.5,
                label=rf'$\tau_K = {tK_ref*1e9:.2f}$ ns')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'Bath eigenmode index $k$')
    ax2.set_ylabel(r'$1/|\lambda_k|$ [ns]')
    ax2.set_title(r'(b) $\mathbf{L}_{FF}$ eigenspectrum (12 slowest modes)')
    
    # Spectral gap
    gap = tau_modes[0] / tau_modes[-1]
    ax2.text(0.97, 0.95, f'Spectral gap\n$= {gap:.2e}$',
             transform=ax2.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', fc='white', alpha=0.8), fontsize=9)
    ax2.legend(fontsize=9)
    
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(str(_FIG / f'mz_fig1_kernel_ITER.{ext}'), dpi=150, bbox_inches='tight')
    print("Saved: mz_fig1_kernel_ITER.{pdf,png}")
    plt.close(fig)


def fig3_M_MZ_single(tau_K, tau_QSS_grid, M_MZ):
    """Regenerate Fig 3: single-panel log10(M_MZ) map."""
    fig, ax = plt.subplots(figsize=(6, 5))

    TE = TE_GRID
    NE = NE_GRID
    
    # Plot M_MZ
    im = ax.pcolormesh(NE, TE, np.log10(M_MZ),
                       cmap='plasma', vmin=1, vmax=8,
                       shading='auto')
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r'$\log_{10}(M_{\rm MZ})$')

    # M=100 contour
    CS = ax.contour(NE, TE, M_MZ, levels=[100], colors='white',
                    linestyles='--', linewidths=1.5)
    ax.clabel(CS, fmt=r'$M_{\rm MZ}=100$', fontsize=8)

    # ITER reference star
    ti_ref, ni_ref = 23, 5
    ax.plot(NE[ni_ref], TE[ti_ref], 'w*', ms=14,
            label=rf'ITER ref ($M_{{\\rm MZ}}={M_MZ[ti_ref,ni_ref]:.0f}$)')

    ax.set_xscale('log')
    ax.set_xlabel(r'$n_e$ [cm$^{-3}$]')
    ax.set_ylabel(r'$T_e$ [eV]')
    ax.set_title(r'$\log_{10}(M_{\rm MZ}) = \log_{10}(\tau_{\rm QSS}/\tau_K)$')
    ax.legend(fontsize=9, loc='upper right')

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(str(_FIG / f'mz_fig3_M_comparison.{ext}'), dpi=150, bbox_inches='tight')
    print("Saved: mz_fig3_M_comparison.{pdf,png}")
    plt.close(fig)


if __name__ == '__main__':
    print("Loading MZ data...")
    tau_K, tau_QSS_grid, M_MZ, evals_FF, K_t, t_arr = load_mz()
    
    ti_ref, ni_ref = 23, 5
    print(f"ITER ref: tau_K={tau_K[ti_ref,ni_ref]*1e9:.2f} ns, "
          f"tau_QSS={tau_QSS_grid[ti_ref,ni_ref]*1e6:.1f} us, "
          f"M_MZ={M_MZ[ti_ref,ni_ref]:.0f}")
    print()
    
    print("Generating Fig 1...")
    fig1_kernel(tau_K, tau_QSS_grid, evals_FF, K_t, t_arr)
    
    print("Generating Fig 3...")
    fig3_M_MZ_single(tau_K, tau_QSS_grid, M_MZ)
    
    print("\nDone. Check figures/ directory.")
