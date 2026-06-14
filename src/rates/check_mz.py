import numpy as np
from assemble_cr_matrix import load_rates, build_L, TE_GRID, NE_GRID

rates = load_rates()
ti = np.argmin(np.abs(TE_GRID - 3.0))
ne = 1e14

L = build_L(ti, ne, rates)
eigs = np.sort(np.linalg.eigvals(L).real)[::-1]
neg = eigs[eigs < -1.0]

print("Top 5 negative eigenvalues of full L(43x43):")
for k in range(5):
    print(f"  neg[{k}] = {neg[k]:.4e} s^-1  ->  tau = {1/abs(neg[k])*1e9:.2f} ns")