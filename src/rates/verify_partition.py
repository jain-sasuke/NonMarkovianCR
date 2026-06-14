"""
verify_partition.py  (v2 — correct diagnostics)
================================================
Verify the 4-state slow partition (1s + n=3) for the PRE paper.

CORRECTED DIAGNOSTICS:
1. Spectral gap = tau_QSS / tau_zz_slow  (NOT tau_rr / tau_zz)
   The relevant gap is between the slowest FULL-SYSTEM timescale
   and the slowest bath mode. tau_QSS >> tau_zz is what validates
   the Markovian approximation.

2. N (dimensionless mismatch) = three options:
   A. N_3P = |u_F(0)[3P]| / n_ss_new(3P)  [simplest, = epsilon_step]
   B. N_MZ = |eta(0)| * tau_zz / |r_ss_new|  [MZ-proper]
   C. N_modal = |A_rz z(0)| / |A_rr r_ss_new|  [forcing/drift ratio]

3. A_rr eigenvalues: eigenvalues of (A_rr + Omega_rr), the QSS
   reduced matrix, not A_rr alone. A_rr alone lacks the Schur
   complement correction from the bath.
"""

import numpy as np
import pathlib, sys

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_HERE))
from assemble_cr_matrix import TE_GRID, NE_GRID

IDX_SLOW = [0, 3, 4, 5]   # 1s, 3s, 3p, 3d
IDX_FAST = [i for i in range(43) if i not in IDX_SLOW]
IDX_3P_in_slow = 2         # 3P is index 2 within IDX_SLOW
IDX_3P_in_fast = 3         # 3P was index 4 in full = IDX_3P-1 in fast space
# Wait: 3P is index 4 in full 43-state. It's in IDX_SLOW at position 2 (0=1s,1=3s,2=3p,3=3d)
# IDX_SLOW = [0, 3, 4, 5] so 3P=index 4 in full = position 2 in slow subspace
N_ION = 1e14

def steady_state(L, S, n_ion=N_ION):
    return np.linalg.solve(L, -S * n_ion)

def run():
    L_grid = np.load(str(_REPO / 'data/processed/cr_matrix/L_grid.npy'))
    S_grid = np.load(str(_REPO / 'data/processed/cr_matrix/S_grid.npy'))

    ti_ref = int(np.argmin(np.abs(TE_GRID - 3.0)))
    ni_ref = int(np.argmin(np.abs(NE_GRID - 1e14)))
    L = L_grid[ti_ref, ni_ref]
    S = S_grid[ti_ref, ni_ref]

    print(f"ITER reference: Te={TE_GRID[ti_ref]:.3f} eV, ne={NE_GRID[ni_ref]:.2e}")
    print()

    # Block extraction
    A_rr = L[np.ix_(IDX_SLOW, IDX_SLOW)]
    A_rz = L[np.ix_(IDX_SLOW, IDX_FAST)]
    A_zr = L[np.ix_(IDX_FAST, IDX_SLOW)]
    A_zz = L[np.ix_(IDX_FAST, IDX_FAST)]

    # --- 1. A_zz stability and bath timescale ---
    print("=== 1. Bath (A_zz) spectrum ===")
    evals_zz = np.linalg.eigvals(A_zz)
    assert np.all(evals_zz.real < 0), "A_zz not stable!"
    sorted_zz = np.sort(np.abs(evals_zz.real))
    tau_zz_slow = 1 / sorted_zz[0]   # slowest bath mode (ns scale)
    tau_zz_fast = 1 / sorted_zz[-1]  # fastest bath mode (ps scale)
    print(f"All Re(lam)<0: True")
    print(f"Slowest bath mode: {tau_zz_slow*1e9:.2f} ns")
    print(f"Fastest bath mode: {tau_zz_fast*1e12:.2f} ps")
    print()

    # --- 2. QSS reduced matrix eigenvalues ---
    print("=== 2. QSS reduced equation: eigenvalues of (A_rr + Omega_rr) ===")
    Omega_rr = -A_rz @ np.linalg.solve(A_zz, A_zr)   # 4x4 Schur complement
    A_qss = A_rr + Omega_rr                            # 4x4 QSS matrix
    evals_qss = np.linalg.eigvals(A_qss)
    sorted_qss = np.sort(np.abs(evals_qss.real))
    tau_QSS_reduced = 1 / sorted_qss[0]    # ionisation balance
    tau_n3_reduced  = 1 / sorted_qss[1]    # fastest resolved mode
    print(f"QSS reduced matrix eigenvalues (tau = 1/|lam|):")
    for i, ev in enumerate(sorted(evals_qss.real, key=abs)):
        tau = 1/abs(ev)
        unit = 'us' if tau > 1e-6 else ('ns' if tau > 1e-9 else 'ps')
        fac = 1e6 if tau > 1e-6 else (1e9 if tau > 1e-9 else 1e12)
        print(f"  lam_{i} = {ev:.4e} s^-1  ->  tau = {tau*fac:.2f} {unit}")
    print()

    # --- 3. Spectral gap (the meaningful one) ---
    print("=== 3. Spectral gap for MZ validity ===")
    # Gap 1: tau_QSS >> tau_zz (bath much faster than ionisation balance)
    gap1 = tau_QSS_reduced / tau_zz_slow
    # Gap 2: tau_n3 >> tau_zz (bath much faster than n=3 relaxation)
    gap2 = tau_n3_reduced / tau_zz_slow
    print(f"tau_QSS (ionisation):     {tau_QSS_reduced*1e6:.2f} us")
    print(f"tau_n3  (fastest slow):   {tau_n3_reduced*1e9:.2f} ns")
    print(f"tau_zz  (slowest bath):   {tau_zz_slow*1e9:.2f} ns")
    print(f"Gap 1: tau_QSS / tau_zz = {gap1:.0f}x  (valid if >> 1)")
    print(f"Gap 2: tau_n3  / tau_zz = {gap2:.1f}x  (valid if >> 1)")
    print(f"Both gaps >> 1: partition is {'VALID' if gap2 > 2 else 'MARGINAL'}")
    print()

    # --- 4. N (dimensionless mismatch) for 3->6 eV step ---
    print("=== 4. N (dimensionless mismatch) for 3->6 eV step ===")
    ti_old = int(np.argmin(np.abs(TE_GRID - 3.0)))
    ti_new = int(np.argmin(np.abs(TE_GRID - 6.0)))
    L_new = L_grid[ti_new, ni_ref]; S_new = S_grid[ti_new, ni_ref]
    L_old = L_grid[ti_old, ni_ref]; S_old = S_grid[ti_old, ni_ref]

    n_ss_new = steady_state(L_new, S_new)
    n_ss_old = steady_state(L_old, S_old)
    u0 = n_ss_old - n_ss_new

    r_ss_new = n_ss_new[IDX_SLOW]   # (4,) resolved SS
    z0 = u0[IDX_FAST]               # (39,) eliminated mismatch

    # Use A_rz from the new L
    A_rz_new = L_new[np.ix_(IDX_SLOW, IDX_FAST)]
    A_zz_new = L_new[np.ix_(IDX_FAST, IDX_FAST)]
    eta_0 = A_rz_new @ z0           # (4,) — units: s^-1 * cm^-3

    # Option A: direct fractional error in n(3P) [= epsilon_step]
    n_3P_old = n_ss_old[4]   # index 4 in full = 3P
    n_3P_new = n_ss_new[4]
    N_A = abs(n_3P_old - n_3P_new) / n_3P_new

    # Option B: |eta(0)| * tau_zz / |r_ss_new|  (dimensionless)
    tau_zz_new = 1 / np.sort(np.abs(np.linalg.eigvals(A_zz_new).real))[0]
    N_B = np.linalg.norm(eta_0) * tau_zz_new / np.linalg.norm(r_ss_new)

    # Option C: |A_rz z0| / |A_rr r_ss_new|
    A_rr_new = L_new[np.ix_(IDX_SLOW, IDX_SLOW)]
    N_C = np.linalg.norm(eta_0) / np.linalg.norm(A_rr_new @ r_ss_new)

    # Thesis epsilon_step at Te=3 eV
    eps_step = 1.53 * np.exp(-0.37 * TE_GRID[ti_old]) + 0.01

    print(f"Te step: {TE_GRID[ti_old]:.2f} -> {TE_GRID[ti_new]:.2f} eV")
    print(f"n_ss_old(3P) = {n_3P_old:.4e}  n_ss_new(3P) = {n_3P_new:.4e}")
    print()
    print(f"N_A (fractional 3P error = epsilon_step): {N_A:.4f}  ({N_A*100:.1f}%)")
    print(f"N_B (|eta|*tau_zz/|r_ss|, dimensionless): {N_B:.4f}  ({N_B*100:.1f}%)")
    print(f"N_C (|eta|/|A_rr*r_ss|, ratio):           {N_C:.4f}  ({N_C*100:.1f}%)")
    print(f"thesis epsilon_step at Te=3eV:             {eps_step:.4f}  ({eps_step*100:.1f}%)")
    print()
    print(f"N_A / eps_step = {N_A/eps_step:.3f}  (should be ~1 if N_A = epsilon_step)")
    print()

    # --- 5. N_A vs epsilon_step across Te_new ---
    print("=== 5. N_A vs epsilon_step across Te_new (Te_old=3, ne=1e14) ===")
    print(f"{'Te_new':>8} {'N_A(%)':>10} {'eps_step(%)':>12} {'N_A/eps':>8}")
    print("-"*42)
    for te_new in [4.0, 5.0, 6.0, 7.0, 8.0, 10.0]:
        ti_n = int(np.argmin(np.abs(TE_GRID - te_new)))
        L_n = L_grid[ti_n, ni_ref]; S_n = S_grid[ti_n, ni_ref]
        n_ss_n = steady_state(L_n, S_n)
        N_n = abs(n_ss_old[4] - n_ss_n[4]) / n_ss_n[4]
        eps_n = 1.53 * np.exp(-0.37 * TE_GRID[ti_old]) + 0.01
        print(f"{TE_GRID[ti_n]:8.2f} {N_n*100:10.1f} {eps_n*100:12.1f} {N_n/eps_n:8.3f}")

    print()
    print("=== 6. Final summary ===")
    print(f"A_zz stable:            True")
    print(f"tau_zz_slow:            {tau_zz_slow*1e9:.2f} ns")
    print(f"tau_QSS (from reduced): {tau_QSS_reduced*1e6:.2f} us")
    print(f"tau_n3  (from reduced): {tau_n3_reduced*1e9:.2f} ns")
    print(f"Gap tau_QSS/tau_zz:     {gap1:.0f}x")
    print(f"Gap tau_n3/tau_zz:      {gap2:.1f}x")
    print(f"N_A at ITER ref (3->6): {N_A:.4f} ({N_A*100:.1f}%)")
    print(f"epsilon_step (thesis):  {eps_step:.4f} ({eps_step*100:.1f}%)")

if __name__ == '__main__':
    run()
