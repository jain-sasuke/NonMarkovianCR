"""
check_modal_weights.py
======================
Compute modal weights c_k for K(t) at ITER reference.
Verifies the claim: top 2 slow modes carry >X% of total |c_k|.

K(t) = L_SF exp(L_FF t) L_FS = sum_k c_k exp(lambda_k t)
c_k = (L_SF v_k)(w_k L_FS)   [scalar for 1-state slow]

where v_k, w_k are right/left eigenvectors of L_FF.

USAGE: cd src/rates && python check_modal_weights.py
"""
import numpy as np, pathlib, sys

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_HERE))
from assemble_cr_matrix import TE_GRID, NE_GRID

IDX_SLOW = [0]
IDX_FAST = list(range(1, 43))

L_grid = np.load(str(_REPO / 'data/processed/cr_matrix/L_grid.npy'))

ti = int(np.argmin(np.abs(TE_GRID - 3.0)))
ni = int(np.argmin(np.abs(NE_GRID - 1.39e14)))
L  = L_grid[ti, ni]

print(f"ITER reference: Te={TE_GRID[ti]:.3f} eV, ne={NE_GRID[ni]:.2e}")
print()

L_FF = L[np.ix_(IDX_FAST, IDX_FAST)]   # 42x42
L_SF = L[np.ix_(IDX_SLOW, IDX_FAST)]   # 1x42
L_FS = L[np.ix_(IDX_FAST, IDX_SLOW)]   # 42x1

evals, V = np.linalg.eig(L_FF)
W = np.linalg.inv(V)

# c_k = (L_SF v_k)(w_k L_FS) — scalars
c = np.array([(L_SF @ V[:, k])[0] * (W[k, :] @ L_FS)[0]
              for k in range(42)], dtype=complex)

tau = 1.0 / np.abs(evals.real)   # mode timescales

# Sort by |c_k| descending
order = np.argsort(np.abs(c.real))[::-1]
total = np.sum(np.abs(c.real))

print("Top 10 modes by |c_k|:")
print(f"{'rank':>4}  {'tau_k (ns)':>12}  {'|c_k|':>14}  {'|c_k|/total%':>13}  cumsum%")
print("-"*65)
cum = 0
for i, k in enumerate(order[:10]):
    cum += np.abs(c.real[k])
    print(f"  {i+1:2d}  {tau[k]*1e9:12.3f}  {np.abs(c.real[k]):14.4e}"
          f"  {np.abs(c.real[k])/total*100:12.2f}%  {cum/total*100:.1f}%")

print()

# Fraction carried by the 2 slowest modes
slow2_idx = order[:2]
frac_top2 = np.sum(np.abs(c.real[slow2_idx])) / total
print(f"Top 2 modes fraction: {frac_top2*100:.1f}%")

# Fraction in modes with tau > 1.5 ns
mask = tau > 1.5e-9
frac_slow = np.sum(np.abs(c.real[mask])) / total
print(f"Modes with tau > 1.5 ns: {np.sum(mask)} modes, fraction = {frac_slow*100:.1f}%")

# Timescales of top 2 modes
print(f"\nTop 2 mode timescales: {tau[order[0]]*1e9:.3f} ns, {tau[order[1]]*1e9:.3f} ns")
print(f"A(2p->1s) pure radiative: 1.594 ns")
print(f"K(0) = L_SF @ L_FS = {float((L_SF @ L_FS)[0,0]):.4e}")
print(f"tau_K (from mori_zwanzig) ~ 2.06 ns")
