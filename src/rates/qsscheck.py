import numpy as np
# Check all eigenvalue files
for fname in ['tau_relax_grid.npy', 'tau_QSS_grid.npy', 'M_grid.npy']:
    try:
        arr = np.load(f'validation/{fname}')
        ti = np.argmin(np.abs(np.logspace(0,1,50)-3.0))
        ni = np.argmin(np.abs(np.logspace(12,15,8)-1e14))
        print(f"{fname}: shape={arr.shape}, ITER ref={arr[ti,ni]:.4e}")
    except: pass