import numpy as np
M = np.load('validation/M_grid.npy')
tau_QSS = np.load('validation/tau_QSS_grid.npy')
tau_relax = np.load('validation/tau_relax_grid.npy')

# ITER reference indices
ti = np.argmin(np.abs(np.logspace(0,1,50) - 3.0))
ni = np.argmin(np.abs(np.logspace(12,15,8) - 1e14))

print(f"M_grid[ITER ref] = {M[ti,ni]:.1f}")
print(f"tau_QSS = {tau_QSS[ti,ni]*1e6:.2f} us")
print(f"tau_relax = {tau_relax[ti,ni]*1e9:.2f} ns")
print(f"M computed = {tau_QSS[ti,ni]/tau_relax[ti,ni]:.1f}")
print(f"Thesis M = 611")