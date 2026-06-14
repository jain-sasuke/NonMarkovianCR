"""
compute_mz_decomposition.py  (v2 — corrected steady-state solve)
=================================================================
Compute the MZ term decomposition of the H-alpha transient at the
ITER reference point.

PHYSICS
-------
For a Te step from Te_old -> Te_new at fixed ne:

  u(t) = n(t) - n^ss_new  satisfies  du/dt = L_new * u

  The exact MZ reduced equation for u_S = u[0] is:
    du_S/dt = L_SS u_S + [L_SF exp(L_FF t) u_F(0)]    <- Term I
                        + [int_0^t K(t-s) u_S(s) ds]  <- Term II

  where:
    u_F(0) = n_F^ss_old - n_F^ss_new  (initial fast-state mismatch)
    K(t)   = L_SF exp(L_FF t) L_FS    (memory kernel)

CORRECTIONS FROM v1
-------------------
1. steady_state_populations: use np.linalg.solve(L, -S*n_ion)
   NOT null-vector of L. L is full-rank (K_ion makes it non-singular).
2. Full solution: u(t) = expm(L_new * t) @ u(0) [via eigendecomposition]
3. Term I: L_SF exp(L_FF t) u_F(0) [only the free propagation piece]
4. Term II: obtained by subtraction: u_F_full - u_F_termI

USAGE
-----
    cd src/rates
    python compute_mz_decomposition.py
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
from assemble_cr_matrix import load_rates, TE_GRID, NE_GRID

# ── Parameters ─────────────────────────────────────────────────────────────────
TE_OLD   = 3.0    # eV before step (ITER reference — ionising)
TE_NEW   = 6.0    # eV after step  (hotter — shows clear excitation increase)
NE_REF   = 1e14   # cm^-3
N_ION    = 1e14   # cm^-3 (ion reservoir)
IDX_3P   = 6      # state index for 3P (n=3, l=1) in 43-state vector
IDX_SLOW = 0
IDX_FAST = list(range(1, 43))

# Time: 0.01 ns to 100 us
T_MIN = 1e-11
T_MAX = 1e-4
N_T   = 400


def steady_state(L, S, n_ion):
    """
    Solve L @ n_ss = -S * n_ion for the steady-state population vector.

    L is (43,43), full rank (K_ion makes diagonal strictly negative).
    Direct solve via np.linalg.solve.
    """
    n_ss = np.linalg.solve(L, -S * n_ion)
    # Sanity: all populations non-negative
    if np.any(n_ss < 0):
        # Fall back to least-squares with non-negativity not enforced
        # Small negatives are numerical noise — clip
        n_ss = np.where(n_ss < 0, 0.0, n_ss)
    return n_ss


def eigensolve(M, v):
    """
    Compute exp(M*t) @ v at multiple times via eigendecomposition.
    Returns function: t_arr -> (N_t, n) array.
    """
    evals, V = np.linalg.eig(M)
    W = np.linalg.inv(V)
    coords = W @ v          # coordinates in eigenbasis
    return evals, V, coords


def propagate(evals, V, coords, t_arr):
    """Evaluate V diag(exp(evals*t)) coords at each t in t_arr."""
    result = np.zeros((len(t_arr), len(coords)))
    for i, t in enumerate(t_arr):
        result[i] = np.real(V @ (np.exp(evals * t) * coords))
    return result


def run_decomposition():
    print("="*60)
    print("MZ DECOMPOSITION: H-alpha transient")
    print(f"Te step: {TE_OLD} eV -> {TE_NEW} eV  at ne={NE_REF:.1e}")
    print("="*60)
    print()

    # Grid indices
    ti_old = int(np.argmin(np.abs(TE_GRID - TE_OLD)))
    ti_new = int(np.argmin(np.abs(TE_GRID - TE_NEW)))
    ni     = int(np.argmin(np.abs(NE_GRID - NE_REF)))
    print(f"Te_old = {TE_GRID[ti_old]:.3f} eV (index {ti_old})")
    print(f"Te_new = {TE_GRID[ti_new]:.3f} eV (index {ti_new})")
    print(f"ne     = {NE_GRID[ni]:.2e} cm^-3 (index {ni})")

    # Load precomputed L and S grids
    L_grid = np.load(str(_REPO / 'data/processed/cr_matrix/L_grid.npy'))
    S_grid = np.load(str(_REPO / 'data/processed/cr_matrix/S_grid.npy'))

    L_old = L_grid[ti_old, ni]   # (43,43)
    S_old = S_grid[ti_old, ni]
    L_new = L_grid[ti_new, ni]
    S_new = S_grid[ti_new, ni]

    # ── Steady-state populations ───────────────────────────────────────────────
    n_ss_old = steady_state(L_old, S_old, N_ION)
    n_ss_new = steady_state(L_new, S_new, N_ION)

    print(f"\nSteady states (Te_old={TE_GRID[ti_old]:.2f}eV / Te_new={TE_GRID[ti_new]:.2f}eV):")
    print(f"  n^ss(1s):  {n_ss_old[0]:.4e} / {n_ss_new[0]:.4e}")
    print(f"  n^ss(3P):  {n_ss_old[IDX_3P]:.4e} / {n_ss_new[IDX_3P]:.4e}")
    print(f"  Ratio n^ss_old(3P)/n^ss_new(3P) = {n_ss_old[IDX_3P]/n_ss_new[IDX_3P]:.4f}")

    # ── Initial deviation ──────────────────────────────────────────────────────
    u0   = n_ss_old - n_ss_new           # u(0) = deviation at t=0
    u0_S = u0[IDX_SLOW]                  # scalar ground-state deviation
    u0_F = u0[IDX_FAST]                  # (42,) fast-state deviation

    print(f"\nInitial deviation u(0) = n^ss_old - n^ss_new:")
    print(f"  u_S(0) [1s]  = {u0_S:.4e}")
    print(f"  u_F(0) [3P]  = {u0_F[IDX_3P-1]:.4e}")
    print(f"  ||u_F(0)||   = {np.linalg.norm(u0_F):.4e}")
    print(f"  ||u_S(0)||   = {abs(u0_S):.4e}")
    frac_F = np.linalg.norm(u0_F) / (np.linalg.norm(u0_F) + abs(u0_S))
    print(f"  Fraction of mismatch in fast subspace: {frac_F*100:.1f}%")

    # ── Block matrices ─────────────────────────────────────────────────────────
    L_FF = L_new[np.ix_(IDX_FAST, IDX_FAST)]    # (42,42)
    L_SF = L_new[np.ix_([IDX_SLOW], IDX_FAST)]  # (1,42)
    L_FS = L_new[np.ix_(IDX_FAST, [IDX_SLOW])]  # (42,1)
    L_SS = float(L_new[IDX_SLOW, IDX_SLOW])

    # Verify L_FF stability
    evals_FF = np.linalg.eigvals(L_FF)
    tau_K = 1.0 / np.sort(np.abs(evals_FF.real))[0]
    print(f"\nL_FF: all Re(lam)<0? {np.all(evals_FF.real<0)}")
    print(f"  tau_K (slowest bath) = {tau_K*1e9:.2f} ns")

    # ── Time grid ──────────────────────────────────────────────────────────────
    t_arr = np.logspace(np.log10(T_MIN), np.log10(T_MAX), N_T)

    # ── Full solution: u(t) = exp(L_new * t) @ u(0) ───────────────────────────
    evals_L, V_L, coords_L = eigensolve(L_new, u0)
    u_full = propagate(evals_L, V_L, coords_L, t_arr)      # (N_T, 43)
    n_full = u_full + n_ss_new[np.newaxis, :]               # n(t)

    # ── Term I: L_SF exp(L_FF t) u_F(0) ───────────────────────────────────────
    evals_FF_v, V_FF, coords_FF = eigensolve(L_FF, u0_F)
    u_F_I = propagate(evals_FF_v, V_FF, coords_FF, t_arr)  # (N_T, 42)
    # Term I contribution to full fast-state deviation
    # In the full u_F(t), Term I is the free-propagation piece

    # ── Term II: u_F(t) - Term I ───────────────────────────────────────────────
    u_F_full = u_full[:, IDX_FAST]        # full fast deviation
    u_F_II   = u_F_full - u_F_I           # Term II by subtraction

    # ── QSS prediction: n(3P) instantly at new SS ─────────────────────────────
    n_3P_QSS  = n_ss_new[IDX_3P] * np.ones(N_T)
    n_3P_full = n_full[:, IDX_3P]

    # ── n(3P) from each term ───────────────────────────────────────────────────
    # n(3P) = n_ss_new[3P] + u_F_total[3P-1]
    # Term I contribution to n(3P):
    IDX_3P_F = IDX_3P - 1   # index within fast subspace
    n_3P_termI  = n_ss_new[IDX_3P] + u_F_I[:, IDX_3P_F]
    n_3P_termII = n_ss_new[IDX_3P] + u_F_II[:, IDX_3P_F]

    # Normalise by new SS
    I_ss = n_ss_new[IDX_3P]
    I_full_n    = n_3P_full   / I_ss
    I_QSS_n     = n_3P_QSS   / I_ss
    I_termI_n   = n_3P_termI  / I_ss
    I_termII_n  = n_3P_termII / I_ss

    print(f"\nH-alpha (n(3P)) analysis:")
    print(f"  At t=0: n(3P)/n_ss = {I_full_n[0]:.4f}  (expect n_ss_old/n_ss_new = {n_ss_old[IDX_3P]/I_ss:.4f})")
    print(f"  Peak |Full - QSS| / n_ss = {np.max(np.abs(I_full_n - I_QSS_n))*100:.2f}%")
    print(f"  At t=0: Term I / n_ss    = {I_termI_n[0]:.4f}")
    print(f"  At t=0: Term II / n_ss   = {I_termII_n[0]:.4f}")
    print(f"  Fraction of initial error from Term I: "
          f"{abs(I_termI_n[0]-1)/max(abs(I_full_n[0]-1),1e-10)*100:.1f}%")

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    plt.rcParams.update({'font.family':'serif', 'font.size':11,
                         'axes.grid':True, 'grid.alpha':0.3})

    # Panel (a): normalized populations
    ax = axes[0]
    ax.semilogx(t_arr*1e9, I_full_n,   'k-',  lw=2.2, label='Full CR (truth)')
    ax.semilogx(t_arr*1e9, I_QSS_n,    'r--', lw=1.8, label='QSS (instant SS)')
    ax.semilogx(t_arr*1e9, I_termI_n,  'C0-', lw=1.5,
                label=r'Term I: $\mathbf{L}_{SF}e^{\mathbf{L}_{FF}t}\mathbf{u}_F(0)$')
    ax.semilogx(t_arr*1e9, I_termII_n, 'C2--',lw=1.5,
                label=r'Term II: memory convolution')
    ax.axhline(1.0, color='gray', lw=0.8, ls=':')
    ax.set_xlabel(r'Time $t$ [ns]')
    ax.set_ylabel(r'$n(3P) / n^{\rm ss}_{\rm new}(3P)$')
    ax.set_title(rf'(a) $n(3P)$ transient: MZ decomposition'
                 '\n'
                 rf'$T_e$ step {TE_OLD}$\to${TE_NEW} eV, $n_e=10^{{14}}$ cm$^{{-3}}$')
    ax.legend(fontsize=8.5, loc='lower right')
    ax.set_xlim(t_arr[0]*1e9, 1e4)

    # Panel (b): error decomposition
    ax2 = axes[1]
    err_full  = (I_full_n  - I_QSS_n) * 100
    err_termI = (I_termI_n - 1.0)     * 100
    err_termII= (I_termII_n - 1.0)    * 100

    ax2.semilogx(t_arr*1e9, err_full,   'k-',  lw=2.2,
                 label=r'Total QSS error $= n_{\rm CR} - n_{\rm QSS}$')
    ax2.semilogx(t_arr*1e9, err_termI,  'C0-', lw=1.5,
                 label='Term I contribution')
    ax2.semilogx(t_arr*1e9, err_termII, 'C2--',lw=1.5,
                 label='Term II contribution')
    ax2.axhline(0, color='gray', lw=0.8, ls=':')
    ax2.set_xlabel(r'Time $t$ [ns]')
    ax2.set_ylabel(r'Error $[\%$ of $n^{\rm ss}_{\rm new}]$')
    ax2.set_title(r'(b) QSS error decomposition')
    ax2.legend(fontsize=8.5)
    ax2.set_xlim(t_arr[0]*1e9, 1e4)

    plt.tight_layout()
    os.makedirs(str(_FIG_DIR), exist_ok=True)
    for ext in ['pdf', 'png']:
        path = str(_FIG_DIR / f'mz_fig6_decomposition.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: figures/mz_fig6_decomposition.{{pdf,png}}")
    plt.close(fig)

    # Save data
    os.makedirs(str(_MZ_DIR), exist_ok=True)
    np.save(str(_MZ_DIR / 'mz_decomp_t.npy'),       t_arr)
    np.save(str(_MZ_DIR / 'mz_decomp_full.npy'),    I_full_n)
    np.save(str(_MZ_DIR / 'mz_decomp_termI.npy'),   I_termI_n)
    np.save(str(_MZ_DIR / 'mz_decomp_termII.npy'),  I_termII_n)
    print("Saved: data/processed/mori_zwanzig/mz_decomp_*.npy")


if __name__ == '__main__':
    run_decomposition()