"""
verify_bundling_psm20.py
========================
Compare ACTUAL PSM20 ℓ-mixing rates (K_lmix.npy) against worst-case
radiative depopulation of np states. Verifies that the bundling assumption
(statistical ℓ-population within shells n=9..15) holds across the (Te, ne)
grid used by the CR matrix.

CRITERION
---------
Bundling requires the within-shell ℓ-relaxation rate to exceed the fastest
substate depopulation. The worst case is the np substate, depleted by the
Lyman channel A(np→1s). The validity criterion is:

    K_lmix(n) / A(np_total) > 10   (well-mixed)
    K_lmix(n) / A(np_total) > 1    (marginal)
    K_lmix(n) / A(np_total) < 1    (bundling INVALID)

USAGE
-----
From repo root:
    cd ~/Desktop/non_markovian_cr
    PYTHONPATH=. python src/validation/verify_bundling_psm20.py

If K_lmix.npy is not at the default path, edit PATH_K_LMIX below.

OUTPUTS
-------
    outputs/bundling/K_lmix_per_shell.npy   (7, n_Te, n_ne) extracted per shell
    outputs/bundling/bundling_ratio.npy     (7, n_Te, n_ne) K_lmix / A(np_total)
    outputs/bundling/bundling_report.txt    text summary
"""

import numpy as np
import os
import sys

# ── User-editable paths ────────────────────────────────────────────────────────
PATH_K_LMIX = 'data/processed/lmix/K_lmix.npy'
FALLBACK_PATHS = [
    'data/processed/cr_matrix/K_lmix.npy',
    'data/processed/K_lmix.npy',
    'data/processed/collisions/K_exc_full/K_lmix.npy',
]
PATH_TE = 'data/processed/cr_matrix/Te_grid_L.npy'
PATH_NE = 'data/processed/cr_matrix/ne_grid_L.npy'
OUTDIR  = 'outputs/bundling'

# ── Model structure (from assemble_cr_matrix.py) ───────────────────────────────
# State layout: indices 0..35 = resolved nℓ for n=1..8; indices 36..42 = bundled n=9..15
N_BUND   = np.arange(9, 16)                # principal quantum numbers
BUND_IDX = np.arange(36, 43)               # state indices in the 43-state vector
N_TE_EXPECTED = 50
N_NE_EXPECTED = 8

# ── A(np→1s) Einstein coefficients [s^-1] from NIST H I tables ────────────────
# Total A from np state ≈ 1.3 × A(np→1s) (Lyman dominant, Balmer/Paschen add ~30%)
A_NP_1S = {
    9:  5.776e6, 10: 4.196e6, 11: 3.143e6, 12: 2.413e6,
    13: 1.892e6, 14: 1.510e6, 15: 1.225e6,
}
A_NP_TOTAL_FACTOR = 1.3

def A_np_total(n):
    return A_NP_TOTAL_FACTOR * A_NP_1S[n]


# ── Find and load K_lmix.npy ───────────────────────────────────────────────────
def find_klmix():
    for p in [PATH_K_LMIX] + FALLBACK_PATHS:
        if os.path.exists(p):
            return p
    print("ERROR: K_lmix.npy not found at any of:")
    for p in [PATH_K_LMIX] + FALLBACK_PATHS:
        print(f"  {p}")
    print("\nEdit PATH_K_LMIX at the top of this script and rerun.")
    sys.exit(1)


# ── Interpret K_lmix shape ─────────────────────────────────────────────────────
def diagnose_shape(K):
    """Identify what K_lmix.npy axes mean. Return ('mode', extraction_info)."""
    sh = K.shape
    print(f"K_lmix.npy shape: {sh}")
    print(f"dtype: {K.dtype}, finite values: {np.isfinite(K).all()}, "
          f"min/max: {K.min():.3e} / {K.max():.3e}")

    g = (N_TE_EXPECTED, N_NE_EXPECTED)   # (50, 8)

    # Case A: full rate matrix (50, 8, 43, 43)
    if K.ndim == 4 and sh[:2] == g and sh[2] == sh[3] == 43:
        print("  → mode: full 43×43 ℓ-mixing contribution to L matrix")
        return 'full_matrix'

    # Case B: per-bundled-state outgoing rate (50, 8, 7)
    if K.ndim == 3 and sh[:2] == g and sh[2] == 7:
        print("  → mode: per-shell outgoing ℓ-mixing rate (7 bundled shells)")
        return 'per_shell'

    # Case C: per-state outgoing rate for all 43 (50, 8, 43)
    if K.ndim == 3 and sh[:2] == g and sh[2] == 43:
        print("  → mode: per-state outgoing rate (extract indices 36..42)")
        return 'per_state'

    # Case D: collisional matrix Te-axis last (43, 43, 50, 8) or similar — flag
    if K.ndim == 4 and sh[2:] == g and sh[0] == sh[1] == 43:
        print("  → mode: full matrix with Te,ne as LAST axes")
        return 'full_matrix_te_last'

    print(f"\nERROR: Unrecognised shape {sh}.")
    print("Expected one of:")
    print("  (50, 8, 43, 43)  full L_mix matrix")
    print("  (50, 8, 7)       per-bundled-shell rate")
    print("  (50, 8, 43)      per-state outgoing rate")
    print("  (43, 43, 50, 8)  full L_mix matrix, Te/ne last")
    print("\nInspect K_lmix.npy manually and add an extraction branch.")
    sys.exit(1)


def extract_per_shell(K, mode):
    """Return (7, n_Te, n_ne) — outgoing ℓ-mixing rate for each bundled shell."""
    K_shell = np.zeros((7, N_TE_EXPECTED, N_NE_EXPECTED))

    if mode == 'full_matrix':
        # K[Te, ne, i, j] = rate from j to i. Outgoing from j = -K[Te,ne,j,j]
        # (assuming diagonal stores total loss). If diagonal is zero/positive,
        # use column sum minus diagonal as a fallback.
        for b in range(7):
            j = BUND_IDX[b]
            diag = K[:, :, j, j]
            if np.all(diag <= 0) and np.any(diag < 0):
                K_shell[b] = -diag
            else:
                col = K[:, :, :, j].sum(axis=-1) - K[:, :, j, j]
                K_shell[b] = col

    elif mode == 'full_matrix_te_last':
        for b in range(7):
            j = BUND_IDX[b]
            diag = K[j, j, :, :]
            if np.all(diag <= 0) and np.any(diag < 0):
                K_shell[b] = -diag
            else:
                col = K[:, j, :, :].sum(axis=0) - K[j, j, :, :]
                K_shell[b] = col

    elif mode == 'per_shell':
        for b in range(7):
            K_shell[b] = K[:, :, b]

    elif mode == 'per_state':
        for b in range(7):
            K_shell[b] = K[:, :, BUND_IDX[b]]

    return K_shell


# ── Load Te, ne grids ──────────────────────────────────────────────────────────
def load_grids():
    if os.path.exists(PATH_TE) and os.path.exists(PATH_NE):
        return np.load(PATH_TE), np.load(PATH_NE)
    print(f"WARNING: grid files not found, using defaults from assemble_cr_matrix.py")
    Te = np.logspace(np.log10(1.0), np.log10(10.0), N_TE_EXPECTED)
    ne = np.logspace(12, 15, N_NE_EXPECTED)
    return Te, ne


# ── PS64 analytical comparison (independent check) ─────────────────────────────
def ps64_estimate(n, Te_eV, ne):
    """Pengelly-Seaton 1964 ℓ-mixing rate, averaged over substates, ×2 for ±channels."""
    T_K = Te_eV / 8.617333e-5
    a0 = 5.292e-9
    R_D = 6.9 * np.sqrt(T_K / ne)
    ln_Lambda = max(np.log(R_D / (n**2 * a0)), 1.0)
    ells = np.arange(n)
    w = 2 * ells + 1
    ell_factor = np.average(((ells + 0.5) / n)**2, weights=w)
    q = 1.294e-5 * (n**2) * ell_factor * (T_K**-0.5) * ln_Lambda
    return 2 * q * ne


# ── Reporting ──────────────────────────────────────────────────────────────────
def report(K_shell, Te_grid, ne_grid, outfile=None):
    A_rad = np.array([A_np_total(n) for n in N_BUND])
    ratio = K_shell / A_rad[:, None, None]

    iTe_3 = np.argmin(np.abs(Te_grid - 3.0))

    lines = []
    lines.append("=" * 76)
    lines.append("PSM20 BUNDLING VALIDITY CHECK — actual K_lmix.npy vs A(np_total)")
    lines.append("=" * 76)
    lines.append(f"Grid: Te = {Te_grid[0]:.2f}..{Te_grid[-1]:.2f} eV ({len(Te_grid)} pts)")
    lines.append(f"      ne = {ne_grid[0]:.2e}..{ne_grid[-1]:.2e} cm^-3 ({len(ne_grid)} pts)")
    lines.append("")
    lines.append("Per-shell summary at Te = 3 eV (closest to ITER reference):")
    lines.append("-" * 76)
    lines.append(f"{'n':>3}  {'A(np_tot)':>11}  {'K_lmix(lo ne)':>15}  "
                 f"{'K_lmix(hi ne)':>15}  {'ratio lo-ne':>13}  status")
    lines.append("-" * 76)
    for b, n in enumerate(N_BUND):
        Kl_lo = K_shell[b, iTe_3, 0]
        Kl_hi = K_shell[b, iTe_3, -1]
        r = Kl_lo / A_rad[b]
        status = "OK" if r >= 10 else ("MARGINAL" if r >= 1 else "INVALID")
        lines.append(f"{n:>3}  {A_rad[b]:>11.2e}  {Kl_lo:>15.2e}  "
                     f"{Kl_hi:>15.2e}  {r:>13.2f}  {status}")
    lines.append("-" * 76)

    # Worst offender across full grid
    iworst = np.unravel_index(np.argmin(ratio), ratio.shape)
    n_w  = N_BUND[iworst[0]]
    Te_w = Te_grid[iworst[1]]
    ne_w = ne_grid[iworst[2]]
    r_w  = ratio[iworst]
    K_w  = K_shell[iworst]
    A_w  = A_rad[iworst[0]]
    lines.append("")
    lines.append("Worst offender across full (n, Te, ne) grid:")
    lines.append(f"  n = {n_w}, Te = {Te_w:.3f} eV, ne = {ne_w:.3e} cm^-3")
    lines.append(f"  K_lmix = {K_w:.3e} s^-1   A(np) = {A_w:.3e} s^-1   "
                 f"ratio = {r_w:.3f}")

    # Grid fractions
    f_invalid  = (ratio < 1).mean()
    f_marginal = ((ratio >= 1) & (ratio < 10)).mean()
    f_ok       = (ratio >= 10).mean()
    lines.append("")
    lines.append("Grid fraction summary (over 7 × 50 × 8 = 2800 points):")
    lines.append(f"  ratio < 1  (invalid):    {f_invalid*100:5.1f}%")
    lines.append(f"  ratio 1–10 (marginal):   {f_marginal*100:5.1f}%")
    lines.append(f"  ratio ≥ 10 (valid):      {f_ok*100:5.1f}%")

    # Cross-check vs PS64 analytical prediction
    lines.append("")
    lines.append("Independent PS64 sanity check at the worst offender:")
    Kw_ps64 = ps64_estimate(n_w, Te_w, ne_w)
    r_ps64 = Kw_ps64 / A_w
    lines.append(f"  PS64 K_lmix = {Kw_ps64:.3e} s^-1   ratio_PS64 = {r_ps64:.3f}")
    rel = (K_w - Kw_ps64) / Kw_ps64 if Kw_ps64 > 0 else float('nan')
    lines.append(f"  PSM20 vs PS64: {rel*100:+.1f}% (expected <10% per Badnell+2021 Fig 4)")

    # Verdict
    lines.append("")
    lines.append("=" * 76)
    if f_invalid > 0.01:
        lines.append("VERDICT: bundling INVALID at >1% of grid — investigate before Paper 1.")
        lines.append("Possible causes:")
        lines.append("  • K_lmix.npy shape interpretation wrong (rerun and verify mode)")
        lines.append("  • Lifetime cutoff active (small Te shrinks K below PS64 estimate)")
        lines.append("  • A_np values inconsistent with your radiative file")
    elif f_marginal > 0.10:
        lines.append("VERDICT: bundling MARGINAL across >10% of grid — add explicit caveat.")
        lines.append("Restrict quantitative claims at the affected (n, ne) corner.")
    else:
        lines.append("VERDICT: bundling assumption holds across the working grid.")
        lines.append("Cite Badnell+2021 PSM20 in Methods; state n=9 marginal-ne caveat.")
    lines.append("=" * 76)

    text = "\n".join(lines)
    print(text)
    if outfile:
        with open(outfile, 'w') as f:
            f.write(text + "\n")
    return ratio


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("Locating K_lmix.npy ...")
    path = find_klmix()
    print(f"  found: {path}\n")

    K = np.load(path)
    mode = diagnose_shape(K)
    print()

    K_shell = extract_per_shell(K, mode)
    print(f"K_shell extracted, shape {K_shell.shape}\n")

    Te_grid, ne_grid = load_grids()

    os.makedirs(OUTDIR, exist_ok=True)
    ratio = report(K_shell, Te_grid, ne_grid,
                   outfile=f'{OUTDIR}/bundling_report.txt')

    np.save(f'{OUTDIR}/K_lmix_per_shell.npy', K_shell)
    np.save(f'{OUTDIR}/bundling_ratio.npy', ratio)
    print(f"\nSaved:")
    print(f"  {OUTDIR}/K_lmix_per_shell.npy   {K_shell.shape}")
    print(f"  {OUTDIR}/bundling_ratio.npy     {ratio.shape}")
    print(f"  {OUTDIR}/bundling_report.txt")


if __name__ == '__main__':
    main()
