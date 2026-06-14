import numpy as np
c   = np.load('data/processed/mori_zwanzig/mode_amplitudes.npy')
ev  = np.load('data/processed/mori_zwanzig/eigenvalues_FF.npy')
ti, ni = 23, 5   # ITER reference
c_ref  = c[ti, ni]
tau_modes = 1 / np.abs(ev[ti, ni].real)
mask_slow = tau_modes > 1.5e-9   # modes slower than 1.5 ns
frac = np.sum(np.abs(c_ref[mask_slow])) / np.sum(np.abs(c_ref))
print(f"Fraction in slow modes (tau > 1.5 ns): {frac*100:.1f}%")