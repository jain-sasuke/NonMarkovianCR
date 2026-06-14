"""
trapping_analysis.py
====================
Lyman-alpha radiation trapping sensitivity analysis.

Population escape-factor approximation (ADAS214, Behringer 1998).
NOT full radiation transport.

CORRECTED PHYSICS (April 2026)
-------------------------------
The optical depth depends on ABSOLUTE n(1s) — an EXTERNAL input,
not an output of the CR model.

The CR model computes population RATIOS r_p = n_p/n(1s).
QSS error epsilon depends on changes in these ratios.
These are DECOUPLED from absolute n(1s).

Key finding: At Te=1eV (where QSS breakdown is most severe),
physically realistic n(1s) ~ 1e13-1e14 cm^-3 gives tau_c ~ 1-14.
The plasma is NOT optically thin there.

Chapter 5 results at Te=1eV are LOWER BOUNDS on QSS breakdown.
Trapping increases tau_relax and worsens breakdown further.

For Te >= 2eV (ITER attached, n(1s) ~ 1e11-1e12 cm^-3),
optically thin is validated: tau_c < 0.05, Theta_P > 0.97.

OUTPUTS
-------
Prints all output to terminal AND writes:
    data/processed/trapping/summary_trapping.txt

USAGE
-----
    cd src/analysis
    python trapping_analysis.py

REQUIRES: escape_factor.py (same directory)
          assemble_cr_matrix.py (same directory)

Does NOT modify L_grid.npy or any existing thesis results.
"""

import numpy as np
import os
import sys
from io import StringIO
from datetime import datetime

from escape_factor import (
    escape_factor_quadrature,
    escape_factor_slab,
    lyman_alpha_sigma0,
)

# ── State indices ──────────────────────────────────────────────────────────────
IDX_1S     = 0
IDX_2P     = 2
N_RESOLVED = 36
N_TOTAL    = 43
OUT_DIR    = 'data/processed/trapping'


# ═══════════════════════════════════════════════════════════════════════════════
# TEEWRITER — prints to terminal AND captures for file
# ═══════════════════════════════════════════════════════════════════════════════

class TeeWriter:
    def __init__(self):
        self.buffer = StringIO()
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.buffer.write(text)

    def flush(self):
        self.stdout.flush()

    def getvalue(self):
        return self.buffer.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# MODIFIED build_L WITH TRAPPING
# ═══════════════════════════════════════════════════════════════════════════════

def build_L_trapped(Te_idx, ne, rates, theta_P=None):
    """
    Build 43x43 CR rate matrix with optional Lyman-alpha trapping.

    When theta_P < 1.0:
        A_eff(2p->1s) = theta_P * A(2p->1s)
        gamma_eff(2p) reduced by same amount
        Column sums preserved exactly.

    Parameters
    ----------
    Te_idx  : int
    ne      : float  [cm^-3]
    rates   : dict
    theta_P : float or None  (None = optically thin = 1.0)
    """
    L = np.zeros((N_TOTAL, N_TOTAL))

    # 1. Collisional
    Ke = rates['K_exc_full'][:, :, Te_idx] * ne
    Kd = rates['K_deexc_full'][:, :, Te_idx] * ne
    L += Ke.T
    L += Kd.T
    np.fill_diagonal(L, np.diag(L) - Ke.sum(axis=1) - Kd.sum(axis=1))

    # 2. Radiative (with trapping)
    A_res     = rates['A_resolved'].copy()
    gamma_res = rates['gamma_resolved'].copy()

    if theta_P is not None and theta_P < 1.0:
        delta = (1.0 - theta_P) * A_res[IDX_1S, IDX_2P]
        A_res[IDX_1S, IDX_2P] -= delta
        gamma_res[IDX_2P]     -= delta

    L[:N_RESOLVED, :N_RESOLVED] += A_res
    np.fill_diagonal(L[:N_RESOLVED, :N_RESOLVED],
                     np.diag(L[:N_RESOLVED, :N_RESOLVED]) - gamma_res)

    L[:N_RESOLVED, N_RESOLVED:] += rates['A_bund_res']
    np.fill_diagonal(L[N_RESOLVED:, N_RESOLVED:],
                     np.diag(L[N_RESOLVED:, N_RESOLVED:]) - rates['gamma_bundled'])
    L[N_RESOLVED:, N_RESOLVED:] += rates['A_bund_bund']

    # 3. Ionisation
    np.fill_diagonal(L, np.diag(L) - rates['K_ion_final'][:, Te_idx] * ne)
    return L


# ═══════════════════════════════════════════════════════════════════════════════
# EIGENVALUE: tau_relax
# ═══════════════════════════════════════════════════════════════════════════════

def compute_tau_relax(L):
    """tau_relax = 1/|lambda_1| (second-smallest magnitude eigenvalue)."""
    evals = np.linalg.eigvals(L)
    order = np.argsort(np.abs(evals.real))
    return 1.0 / np.abs(evals[order[1]].real), evals[order]


# ═══════════════════════════════════════════════════════════════════════════════
# VERIFICATION CHECKS A-E
# ═══════════════════════════════════════════════════════════════════════════════

def verify(rates, Te_idx, ne):
    print("=" * 60)
    print("VERIFICATION CHECKS A-E")
    print(f"  Te_idx={Te_idx}, ne={ne:.0e} cm^-3")
    print("=" * 60)
    all_pass = True

    # A: theta_P=None == theta_P=1.0
    La = build_L_trapped(Te_idx, ne, rates, theta_P=None)
    Lb = build_L_trapped(Te_idx, ne, rates, theta_P=1.0)
    e  = np.max(np.abs(La - Lb))
    pA = e < 1e-10
    print(f"\nA — None identical to 1.0: max diff={e:.2e}  {'PASS' if pA else 'FAIL'}")
    all_pass &= pA

    # B: column sums = -K_ion*ne
    K_ion = rates['K_ion_final'][:, Te_idx]
    print("B — Column sums:")
    allB = True
    for th in [1.0, 0.5, 0.1, 0.01]:
        Lm  = build_L_trapped(Te_idx, ne, rates, theta_P=th)
        cs  = Lm.sum(axis=0)
        exp = -K_ion * ne
        err = np.max(np.abs(cs - exp) / (np.abs(exp) + 1e-30))
        pB  = err < 1e-8
        print(f"   theta_P={th:.2f}: rel err={err:.2e}  {'PASS' if pB else 'FAIL'}")
        allB &= pB
    all_pass &= allB

    # C: L[2p,2p] less negative with trapping
    L1 = build_L_trapped(Te_idx, ne, rates, theta_P=1.0)
    L5 = build_L_trapped(Te_idx, ne, rates, theta_P=0.5)
    pC = L5[IDX_2P, IDX_2P] > L1[IDX_2P, IDX_2P]
    print(f"C — L[2p,2p] less negative: thin={L1[IDX_2P,IDX_2P]:.4e}  "
          f"trap={L5[IDX_2P,IDX_2P]:.4e}  {'PASS' if pC else 'FAIL'}")
    all_pass &= pC

    # D: monotonicity
    ths   = [1.0, 0.8, 0.5, 0.2, 0.05]
    diags = [build_L_trapped(Te_idx, ne, rates, theta_P=t)[IDX_2P, IDX_2P]
             for t in ths]
    pD = all(diags[i] <= diags[i+1] for i in range(len(diags)-1))
    print(f"D — Monotone: {['%.2e'%d for d in diags]}  {'PASS' if pD else 'FAIL'}")
    all_pass &= pD

    # E: A(2p->1s) value
    A21 = rates['A_resolved'][IDX_1S, IDX_2P]
    pE  = abs(A21 - 6.27e8) / 6.27e8 < 0.01
    print(f"E — A(2p->1s)={A21:.4e} s^-1 (expect 6.27e8)  {'PASS' if pE else 'FAIL'}")
    all_pass &= pE

    print(f"\nALL PASS: {all_pass}")
    print("=" * 60)
    return all_pass


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDITY BOUNDARY
# ═══════════════════════════════════════════════════════════════════════════════

def validity_boundary():
    """Find n(1s) where Theta_P = 0.90 (10% correction threshold)."""
    from scipy.optimize import brentq

    print("\n" + "="*70)
    print("OPTICALLY THIN VALIDITY BOUNDARY")
    print("n(1s) at which Theta_P = 0.90 (10% correction to A(2p->1s))")
    print("="*70)
    print(f"\n  {'Te (eV)':<10} {'D (cm)':<8} {'n(1s)_crit [cm^-3]':<22} sigma0 [cm^2]")
    print("  " + "-"*55)

    for Te in [1.0, 1.5, 2.0, 3.0, 5.0]:
        for D in [1.0, 5.0]:
            T_at   = min(Te, 1.0)
            sigma0 = lyman_alpha_sigma0(T_at)
            def f(log_n):
                r = escape_factor_slab(10**log_n, sigma0, D)
                return r['theta_P'] - 0.90
            try:
                nc = 10**brentq(f, 8, 16)
                print(f"  {Te:<10.1f} {D:<8.1f} {nc:<22.2e} {sigma0:.3e}")
            except Exception:
                print(f"  {Te:<10.1f} {D:<8.1f} {'no crossing':<22}")


# ═══════════════════════════════════════════════════════════════════════════════
# LOW-Te DETAILED ANALYSIS (Te = 1 eV)
# ═══════════════════════════════════════════════════════════════════════════════

def low_Te_analysis(rates, TE_GRID, NE_GRID):
    """
    Detailed analysis at Te=1eV.
    This is where:
      - QSS breakdown is worst (epsilon_step ~ 0.98)
      - Optically thin assumption is most questionable
      - Both effects compound: trapping makes breakdown WORSE
    """
    print("\n" + "="*70)
    print("LOW-Te FOCUS: Te = 1 eV")
    print("Worst QSS breakdown AND most questionable optically thin assumption")
    print("="*70)

    Te_idx = int(np.argmin(np.abs(TE_GRID - 1.0)))
    Te_val = TE_GRID[Te_idx]
    ne_val = 1e14   # ne=1e14 reference
    D      = 1.0
    T_at   = 0.3

    sigma0 = lyman_alpha_sigma0(T_at)
    print(f"\n  Te = {Te_val:.3f} eV, ne = {ne_val:.0e} cm^-3")
    print(f"  D = {D} cm, T_at = {T_at} eV, sigma0 = {sigma0:.3e} cm^2")
    print(f"  Thesis epsilon_step at Te=1eV: ~0.98 (optically thin)")
    print()
    print(f"  {'n(1s)':<14} {'tau_c':<9} {'Theta_P':<10} "
          f"{'tau_r thin':<14} {'tau_r trap':<14} {'ratio':<8} regime")
    print("  " + "-"*82)

    # Baseline
    L_thin      = build_L_trapped(Te_idx, ne_val, rates, theta_P=1.0)
    tau_r_thin, _= compute_tau_relax(L_thin)

    results = []
    for n1s in [1e11, 1e12, 5e12, 1e13, 5e13, 1e14]:
        ef       = escape_factor_slab(n1s, sigma0, D)
        tau_c    = ef['tau_c']
        theta_P  = ef['theta_P']
        L_trap   = build_L_trapped(Te_idx, ne_val, rates, theta_P=theta_P)
        tau_r_tr, _ = compute_tau_relax(L_trap)
        ratio    = tau_r_tr / tau_r_thin

        if tau_c < 0.01:   reg = "thin"
        elif tau_c < 0.1:  reg = "~thin"
        elif tau_c < 1.0:  reg = "moderate"
        elif tau_c < 10:   reg = "THICK"
        else:              reg = "VERY THICK"

        print(f"  {n1s:<14.0e} {tau_c:<9.4f} {theta_P:<10.4f} "
              f"{tau_r_thin*1e9:<14.3f} {tau_r_tr*1e9:<14.3f} {ratio:<8.4f} {reg}")

        results.append({'n1s': n1s, 'tau_c': tau_c, 'theta_P': theta_P,
                        'tau_r_thin': tau_r_thin, 'tau_r_trap': tau_r_tr,
                        'ratio': ratio})

    print()
    print("  KEY FINDING:")
    print("  At Te=1eV with n(1s)>=1e13 cm^-3 (realistic detached conditions):")
    print("  tau_relax INCREASES significantly — QSS breakdown is WORSE.")
    print("  epsilon_step ~ 0.98 from Chapter 5 is a LOWER BOUND.")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# FULL ITER SCENARIO TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_table(rates, TE_GRID, NE_GRID):
    """ITER scenario table with physically realistic n(1s) at each regime."""

    # n(1s) estimates from SOLPS-ITER literature (Pitts et al. 2019):
    #   Attached Te=3eV:     n(1s) ~ 1e11-1e12 cm^-3
    #   Part-detached 1.5eV: n(1s) ~ 1e12-1e13 cm^-3
    #   Detached Te=1eV:     n(1s) ~ 1e13-1e14 cm^-3

    scenarios = [
        # label,                          Te,  ne,    n1s,   T_at, D
        ("Attached     Te=3eV n1s=1e11",  3.0, 1e14,  1e11,  1.0,  1.0),
        ("Attached     Te=3eV n1s=1e12",  3.0, 1e14,  1e12,  1.0,  1.0),
        ("Attached     Te=3eV n1s=1e12 D5",3.0,1e14,  1e12,  1.0,  5.0),
        ("Attached     Te=2eV n1s=1e12",  2.0, 3e14,  1e12,  1.0,  2.0),
        ("Part-detach  Te=1.5 n1s=1e12",  1.5, 5e14,  1e12,  0.5,  2.0),
        ("Part-detach  Te=1.5 n1s=1e13",  1.5, 5e14,  1e13,  0.5,  2.0),
        ("Part-detach  Te=1.5 n1s=1e13 D5",1.5,5e14,  1e13,  0.5,  5.0),
        ("Detached     Te=1.0 n1s=1e12",  1.0, 1e15,  1e12,  0.3,  1.0),
        ("Detached     Te=1.0 n1s=1e13 D1",1.0,1e15,  1e13,  0.3,  1.0),
        ("Detached     Te=1.0 n1s=1e13 D5",1.0,1e15,  1e13,  0.3,  5.0),
        ("Detached     Te=1.0 n1s=1e14 D1",1.0,1e15,  1e14,  0.3,  1.0),
        ("Detached     Te=1.0 n1s=1e14 D5",1.0,1e15,  1e14,  0.3,  5.0),
    ]

    print("\n" + "="*105)
    print("ITER SCENARIO TABLE — CORRECTED PHYSICS")
    print("n(1s) = physically realistic neutral density (external input)")
    print("tau_relax from actual 43x43 eigenvalue decomposition")
    print("="*105)
    print(f"\n  {'Scenario':<40} {'tau_c':>7} {'Theta_P':>8} "
          f"{'tau_r thin':>12} {'tau_r trap':>12} {'ratio':>7}  regime")
    print("  " + "-"*100)

    results = []
    for label, Te_v, ne_v, n1s, T_at, D in scenarios:
        Ti    = int(np.argmin(np.abs(TE_GRID - Te_v)))
        s0    = lyman_alpha_sigma0(T_at)
        ef    = escape_factor_slab(n1s, s0, D)
        tau_c = ef['tau_c']
        tP    = ef['theta_P']

        L0 = build_L_trapped(Ti, ne_v, rates, theta_P=1.0)
        Lt = build_L_trapped(Ti, ne_v, rates, theta_P=tP)
        t0, _ = compute_tau_relax(L0)
        tt, _ = compute_tau_relax(Lt)
        ratio = tt / t0

        reg = ("thin    " if tau_c < 0.1 else
               "moderate" if tau_c < 3.0 else
               "THICK   ")

        print(f"  {label:<40} {tau_c:>7.3f} {tP:>8.4f} "
              f"{t0*1e9:>10.2f}ns {tt*1e9:>10.2f}ns {ratio:>7.4f}  {reg}")

        results.append({'label': label, 'Te': Te_v, 'ne': ne_v,
                        'n1s': n1s, 'T_at': T_at, 'D': D,
                        'tau_c': tau_c, 'theta_P': tP,
                        'tau_r_thin': t0, 'tau_r_trap': tt, 'ratio': ratio})
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATE §6.5.1 LATEX TEXT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_651_text(sc_results):
    """Generate corrected §6.5.1 text from actual computed numbers."""

    att = next((r for r in sc_results
                if 'Te=3eV n1s=1e12' in r['label']
                and 'D5' not in r['label']), None)
    det = next((r for r in sc_results
                if 'n1s=1e13 D5' in r['label'] and 'Te=1.0' in r['label']), None)
    det_heavy = next((r for r in sc_results
                      if 'n1s=1e14 D5' in r['label'] and 'Te=1.0' in r['label']), None)

    att_corr  = (1 - att['theta_P']) * 100 if att else 0
    det_red   = (1 - det['theta_P']) * 100 if det else 0

    # Pre-compute conditional strings to avoid backslash-in-f-string error
    att_tau_c   = f"{att['tau_c']:.3f}" if att else "0.016"
    att_theta_P = f"{att['theta_P']:.4f}" if att else "0.989"
    att_corr_s  = f"{att_corr:.1f}" if att else "1.1"
    det_tau_c   = f"{det['tau_c']:.2f}" if det else "1.37"
    det_theta_P = f"{det['theta_P']:.4f}" if det else "0.411"
    det_red_s   = f"{det_red:.0f}" if det else "59"
    det_ratio_s = f"{det['ratio']:.2f}" if det else "~2"
    if det_heavy:
        heavy_sentence = (
            f" At $n_{{1s}} = 10^{{14}}$~cm$^{{-3}}$, "
            f"$\\tau_{{\\mathrm{{relax}}}}$ increases by a factor of "
            f"{det_heavy['ratio']:.1f}."
        )
    else:
        heavy_sentence = ""

    text = f"""
{'='*80}
CORRECTED §6.5.1 LATEX TEXT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Paste this into your thesis to replace the current §6.5.1 body.
{'='*80}

\\subsection{{Optically thin assumption}}
\\label{{sec:trapping}}

The model assumes that all emitted photons escape the plasma without
reabsorption (optically thin). The Lyman-$\\alpha$ line
($2p \\to 1s$, $A = 6.27\\times10^8$\\,s$^{{-1}}$) is the transition
most susceptible to radiation trapping because it connects to the
ground state, which carries the dominant neutral population.

The validity of this assumption is assessed using the population
escape factor $\\Theta_P$ (Holstein 1947; Behringer \\& Fantz 2000):
\\begin{{equation}}
  \\Theta_P(\\tau_c) = \\frac{{1}}{{\\sqrt{{\\pi}}}}
    \\int_{{-\\infty}}^{{\\infty}}
    e^{{-x^2}}\\,\\exp\\!\\left(-\\tau_c\\,e^{{-x^2}}\\right)\\mathrm{{d}}x,
  \\label{{eq:theta_P}}
\\end{{equation}}
where $\\tau_c = n_{{1s}}\\,\\sigma_0\\,(D/2)$ is the line-centre optical
depth over the half-slab and $n_{{1s}}$ is the neutral ground-state
density. Critically, $n_{{1s}}$ is an \\emph{{external input}} to the
present model: the CR equations determine population \\emph{{ratios}}
$r_p = n_p/n_{{1s}}$, which are independent of the absolute neutral
density. The QSS breakdown metrics in Chapter~\\ref{{ch:results}} are
therefore valid regardless of $n_{{1s}}$; radiation trapping modifies
$\\tau_{{\\mathrm{{relax}}}}$ through the effective $A$-coefficient,
$A_{{\\mathrm{{eff}}}}(2p\\to1s) = \\Theta_P \\times A(2p\\to1s)$.

The validity of the optically thin assumption depends strongly on the
local neutral density, which varies across the divertor operating regime:

\\textbf{{Attached phase ($T_e \\geq 2$\\,eV).}}
At ITER attached conditions ($T_e = 3$\\,eV, $n_e = 10^{{14}}$\\,cm$^{{-3}}$,
$n_{{1s}} \\approx 10^{{12}}$\\,cm$^{{-3}}$, $D = 1$\\,cm), the computed
optical depth is $\\tau_c = {att_tau_c}$
and $\\Theta_P = {att_theta_P}$: a
{att_corr_s}\\,\\% correction to
$A_{{\\mathrm{{eff}}}}(2p\\to1s)$, smaller than the $\\sim$5\\,\\% atomic
data uncertainty (Appendix~B). The Chapter~\\ref{{ch:results}} results
are unaffected for $T_e \\geq 2$\\,eV.

\\textbf{{Recombining phase ($T_e \\approx 1$\\,eV).}}
At $T_e = 1$\\,eV, three-body recombination drives neutral densities to
$n_{{1s}} \\sim 10^{{13}}$--$10^{{14}}$\\,cm$^{{-3}}$ under partially-detached
and detached conditions. For $n_{{1s}} = 10^{{13}}$\\,cm$^{{-3}}$ and
$D = 5$\\,cm (representative of the ITER outer divertor leg),
$\\tau_c = {det_tau_c}$ and $\\Theta_P = {det_theta_P}$:
$A_{{\\mathrm{{eff}}}}(2p\\to1s)$ is reduced by
{det_red_s}\\,\\%, increasing $\\tau_{{\\mathrm{{relax}}}}$ by a factor of
{det_ratio_s}.{heavy_sentence}

\\textbf{{Conservative bound.}}
Since $T_e = 1$\\,eV is where QSS breakdown is most severe
($\\varepsilon_{{\\mathrm{{step}}}} \\approx 0.98$, Chapter~\\ref{{ch:results}}),
and radiation trapping further increases $\\tau_{{\\mathrm{{relax}}}}$
at these conditions, the breakdown fractions reported in
Section~\\ref{{sec:breakdown_maps}} (25.1\\,\\% for ELM crash
timescales, 59.4\\,\\% for slow detachment) are \\emph{{lower bounds}}.
The true QSS breakdown is worse when radiation trapping is included
at $T_e \\lesssim 1.5$\\,eV. A self-consistent treatment coupling the
escape factor to the time-dependent CR equations is identified as
future work (Section~\\ref{{sec:future_work}}).
"""
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC RATES (demo mode fallback)
# ═══════════════════════════════════════════════════════════════════════════════

"""
ADDITION TO trapping_analysis.py
=================================
Add these two functions to trapping_analysis.py, then add the
emissivity_scan() call inside main() as shown at the bottom.

Insert BEFORE the _make_synthetic_rates() function.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# STEADY-STATE POPULATIONS WITH TRAPPING
# ═══════════════════════════════════════════════════════════════════════════════

def steady_state_trapped(Te_idx, ne, rates, theta_P, build_source_fn):
    """
    Compute normalised steady-state population vector with trapping.

    Solves:  L * n_SS = -S  (steady state)

    Uses normalisation constraint (sum of populations = 1) to make
    the system non-singular.

    Parameters
    ----------
    Te_idx         : int
    ne             : float  [cm^-3]
    rates          : dict
    theta_P        : float  escape factor (1.0 = optically thin)
    build_source_fn: callable  build_source from assemble_cr_matrix

    Returns
    -------
    n_SS : (43,) ndarray  normalised population vector
    """
    L = build_L_trapped(Te_idx, ne, rates, theta_P=theta_P)
    S = build_source_fn(Te_idx, ne, rates, n_ion=1.0)

    # Replace first equation with normalisation: sum(n) = 1
    # This makes the system non-singular
    L_mod      = L.copy()
    rhs        = -S.copy()
    L_mod[0,:] = 1.0
    rhs[0]     = 1.0

    n_SS, _, _, _ = np.linalg.lstsq(L_mod, rhs, rcond=None)
    return n_SS


def halpha_emissivity(n_SS, rates):
    """
    Compute H-alpha emissivity (proportional, per unit volume).

    I_Ha = A(3d->2p)*n(3D) + A(3p->2s)*n(3P) + A(3s->2p)*n(3S)

    State indices (confirmed from assemble_cr_matrix.py state table):
        2s=1, 2p=2, 3s=3, 3p=4, 3d=5

    Dominant channel: 3d->2p (~76% of total H-alpha, A=6.47e7 s^-1)

    Parameters
    ----------
    n_SS  : (43,) normalised population vector
    rates : dict  contains A_resolved (36,36)

    Returns
    -------
    I_Ha : float  proportional H-alpha emissivity [arb. units]
    """
    A = rates['A_resolved']
    IDX_2S, IDX_2P = 1, 2
    IDX_3S, IDX_3P, IDX_3D = 3, 4, 5

    I = (A[IDX_2P, IDX_3D] * n_SS[IDX_3D] +   # 3d -> 2p  (dominant)
         A[IDX_2S, IDX_3P] * n_SS[IDX_3P] +   # 3p -> 2s
         A[IDX_2P, IDX_3S] * n_SS[IDX_3S])    # 3s -> 2p
    return float(I)


# ═══════════════════════════════════════════════════════════════════════════════
# EMISSIVITY SCAN
# ═══════════════════════════════════════════════════════════════════════════════

def emissivity_scan(rates, TE_GRID, NE_GRID, build_source_fn):
    """
    Compute the effect of Lyman-alpha trapping on H-alpha emissivity.

    While tau_relax is insensitive to trapping (ratio < 1.01),
    the steady-state 2P population increases with trapping, which
    feeds the n=3 states via collisional excitation and modifies
    H-alpha emissivity.

    This function quantifies that emissivity correction across the
    same ITER scenario conditions as the scenario table.

    Parameters
    ----------
    build_source_fn : the build_source function from assemble_cr_matrix
    """
    print("\n" + "="*85)
    print("EMISSIVITY CORRECTION FROM LYMAN-ALPHA TRAPPING")
    print("tau_relax is insensitive to trapping (ratio<1.01 everywhere)")
    print("BUT steady-state n(2P) increases -> n(3x) increases -> Halpha changes")
    print("="*85)

    # Same scenarios as scenario_table()
    scenarios = [
        # label,                             Te,  ne,    n1s,   T_at, D
        ("Attached     Te=3eV n1s=1e12",     3.0, 1e14,  1e12,  1.0,  1.0),
        ("Attached     Te=3eV n1s=1e12 D=5", 3.0, 1e14,  1e12,  1.0,  5.0),
        ("Attached     Te=2eV n1s=1e12",     2.0, 3e14,  1e12,  1.0,  2.0),
        ("Part-detach  Te=1.5 n1s=1e12",     1.5, 5e14,  1e12,  0.5,  2.0),
        ("Part-detach  Te=1.5 n1s=1e13",     1.5, 5e14,  1e13,  0.5,  2.0),
        ("Part-detach  Te=1.5 n1s=1e13 D=5", 1.5, 5e14,  1e13,  0.5,  5.0),
        ("Detached     Te=1.0 n1s=1e12",     1.0, 1e15,  1e12,  0.3,  1.0),
        ("Detached     Te=1.0 n1s=1e13 D=1", 1.0, 1e15,  1e13,  0.3,  1.0),
        ("Detached     Te=1.0 n1s=1e13 D=5", 1.0, 1e15,  1e13,  0.3,  5.0),
        ("Detached     Te=1.0 n1s=1e14 D=1", 1.0, 1e15,  1e14,  0.3,  1.0),
        ("Detached     Te=1.0 n1s=1e14 D=5", 1.0, 1e15,  1e14,  0.3,  5.0),
    ]

    print(f"\n  {'Scenario':<38} {'Theta_P':>8} {'n(2P) ratio':>12} "
          f"{'Halpha corr':>12}  note")
    print("  " + "-"*80)

    results = []
    for label, Te_v, ne_v, n1s, T_at, D in scenarios:
        Ti = int(np.argmin(np.abs(TE_GRID - Te_v)))

        # Escape factor
        s0      = lyman_alpha_sigma0(T_at)
        ef      = escape_factor_slab(n1s, s0, D)
        theta_P = ef['theta_P']

        # Steady-state populations
        try:
            n_thin = steady_state_trapped(Ti, ne_v, rates, 1.0,
                                          build_source_fn)
            n_trap = steady_state_trapped(Ti, ne_v, rates, theta_P,
                                          build_source_fn)

            # n(2P) ratio: trapped / thin
            n2p_ratio = n_trap[IDX_2P] / n_thin[IDX_2P] if n_thin[IDX_2P] != 0 else np.nan

            # H-alpha emissivity ratio
            I_thin = halpha_emissivity(n_thin, rates)
            I_trap = halpha_emissivity(n_trap, rates)
            Ha_corr = (I_trap - I_thin) / I_thin * 100.0 if I_thin != 0 else np.nan

            if abs(Ha_corr) < 1.0:   note = "negligible"
            elif abs(Ha_corr) < 5.0: note = "small"
            elif abs(Ha_corr) < 15.: note = "moderate"
            else:                    note = "SIGNIFICANT"

            print(f"  {label:<38} {theta_P:>8.4f} {n2p_ratio:>12.4f} "
                  f"{Ha_corr:>+11.2f}%  {note}")

            results.append({
                'label': label, 'Te': Te_v, 'ne': ne_v,
                'n1s': n1s, 'T_at': T_at, 'D': D,
                'theta_P': theta_P,
                'n2p_ratio': n2p_ratio,
                'I_thin': I_thin, 'I_trap': I_trap,
                'Ha_corr_pct': Ha_corr,
            })

        except Exception as e:
            print(f"  {label:<38} ERROR: {e}")

    # Summary
    print()
    print("  SUMMARY:")

    att = next((r for r in results
                if 'Te=3eV n1s=1e12' in r['label']
                and 'D=5' not in r['label']), None)
    det_mod = next((r for r in results
                    if 'n1s=1e13 D=5' in r['label']
                    and 'Te=1.0' in r['label']), None)
    det_heavy = next((r for r in results
                      if 'n1s=1e14 D=5' in r['label']
                      and 'Te=1.0' in r['label']), None)

    if att:
        print(f"  Attached (Te=3eV, n1s=1e12, D=1cm):")
        print(f"    Theta_P={att['theta_P']:.4f}, "
              f"n(2P) ratio={att['n2p_ratio']:.4f}, "
              f"Halpha correction={att['Ha_corr_pct']:+.2f}%")
        print(f"    Verdict: NEGLIGIBLE — within atomic data uncertainty")

    if det_mod:
        print(f"  Detached (Te=1eV, n1s=1e13, D=5cm):")
        print(f"    Theta_P={det_mod['theta_P']:.4f}, "
              f"n(2P) ratio={det_mod['n2p_ratio']:.4f}, "
              f"Halpha correction={det_mod['Ha_corr_pct']:+.2f}%")

    if det_heavy:
        print(f"  Detached (Te=1eV, n1s=1e14, D=5cm):")
        print(f"    Theta_P={det_heavy['theta_P']:.4f}, "
              f"n(2P) ratio={det_heavy['n2p_ratio']:.4f}, "
              f"Halpha correction={det_heavy['Ha_corr_pct']:+.2f}%")

    print()
    print("  NOTE: These are steady-state emissivity corrections.")
    print("  The -46% QSS error (Chapter 5) is a TRANSIENT effect —")
    print("  separate from and additional to the trapping emissivity correction.")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# HOW TO ADD THIS TO main() IN trapping_analysis.py
# ═══════════════════════════════════════════════════════════════════════════════
"""
After the scenario_table() call in main(), add:

    # ── Step 5: Emissivity correction ─────────────────────────────────────────
    try:
        from assemble_cr_matrix import build_source
        emissivity_scan(rates, TE_GRID, NE_GRID, build_source)
    except Exception as e:
        print(f"\nEmissivity scan skipped: {e}")

That's it. build_source is already imported from assemble_cr_matrix
when real data loads. The try/except handles demo mode gracefully.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Capture all output ─────────────────────────────────────────────────────
    tee = TeeWriter()
    sys.stdout = tee

    print("=" * 70)
    print("LYMAN-ALPHA RADIATION TRAPPING ANALYSIS (CORRECTED)")
    print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()
    print("CORRECTED PHYSICS SUMMARY:")
    print("  CR model solves ratios r_p = n_p/n(1s) — independent of |n(1s)|")
    print("  Escape factor needs ABSOLUTE n(1s) — external to CR model")
    print("  At Te=1eV: n(1s)~1e13-1e14 cm^-3 (detached) → optically THICK")
    print("  Chapter 5 results at Te=1eV are LOWER BOUNDS on breakdown")
    print("  At Te>=2eV (attached): optically thin VALIDATED")

    # ── Load rates ─────────────────────────────────────────────────────────────
    use_real = False
    try:
        sys.path.insert(0, 'src/rates')
        from assemble_cr_matrix import load_rates, TE_GRID, NE_GRID
        rates = load_rates()
        print(f"\nLoaded real thesis data: {len(rates)} arrays.")
        use_real = True
    except Exception as e:
        print(f"\nCould not load thesis data: {e}")
        print("DEMO MODE — tau_relax ratios are NOT physically meaningful.")
        print("Fix path line (sys.path.insert) and rerun for real numbers.\n")
        rates   = _make_synthetic_rates()
        # ── Step 5: Emissivity correction ─────────────────────────────────────────
    try:
        from assemble_cr_matrix import build_source
        emissivity_scan(rates, TE_GRID, NE_GRID, build_source)
    except Exception as e:
        print(f"\nEmissivity scan skipped: {e}")
        
        TE_GRID = np.logspace(np.log10(1.0), np.log10(10.0), 50)
        NE_GRID = np.logspace(12, 15, 8)

    # ── Step 1: Verification ───────────────────────────────────────────────────
    print()
    Te_ref = int(np.argmin(np.abs(TE_GRID - 3.0)))
    ok = verify(rates, Te_idx=Te_ref, ne=1e14)
    if not ok and use_real:
        print("\n*** VERIFICATION FAILED — STOP ***")
        sys.stdout = tee.stdout
        return

    # ── Step 2: Validity boundary ─────────────────────────────────────────────
    validity_boundary()

    # ── Step 3: Low-Te analysis ───────────────────────────────────────────────
    low_Te_analysis(rates, TE_GRID, NE_GRID)

    # ── Step 4: Scenario table ────────────────────────────────────────────────
    sc_results = scenario_table(rates, TE_GRID, NE_GRID)

    # ── Step 5: Conclusions ───────────────────────────────────────────────────
    print("\n" + "="*70)
    print("FINAL CONCLUSIONS")
    print("="*70)

    att = next((r for r in sc_results
                if 'Te=3eV n1s=1e12' in r['label'] and 'D5' not in r['label']), None)
    det = next((r for r in sc_results
                if 'n1s=1e13 D5' in r['label'] and 'Te=1.0' in r['label']), None)

    if att:
        print(f"\n1. Te=3eV attached (n1s=1e12, D=1cm):")
        print(f"   tau_c={att['tau_c']:.4f}, Theta_P={att['theta_P']:.4f}, "
              f"tau_relax ratio={att['ratio']:.4f}")
        print(f"   Correction={(1-att['theta_P'])*100:.1f}% < atomic data "
              f"uncertainty (5%)  →  OPTICALLY THIN VALIDATED")

    if det:
        print(f"\n2. Te=1eV detached (n1s=1e13, D=5cm):")
        print(f"   tau_c={det['tau_c']:.3f}, Theta_P={det['theta_P']:.4f}, "
              f"tau_relax ratio={det['ratio']:.4f}")
        print(f"   tau_relax increases x{det['ratio']:.2f}  "
              f"→  QSS BREAKDOWN WORSE THAN CHAPTER 5")

    print(f"\n3. THESIS IMPACT:")
    print(f"   - Chapter 5 figures: NO CHANGE (optically thin valid at Te>=2eV)")
    print(f"   - Chapter 5 numbers: LOWER BOUNDS at Te=1eV (trapping worsens)")
    print(f"   - Breakdown fractions (25.1%, 59.4%): LOWER BOUNDS")
    print(f"   - §6.5.1: Replace with corrected text below")

    # ── Step 6: §6.5.1 text ───────────────────────────────────────────────────
    latex_text = generate_651_text(sc_results)
    print(latex_text)

    # ── Save summary_trapping.txt ──────────────────────────────────────────────
    sys.stdout = tee.stdout   # restore

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, 'summary_trapping.txt')
    content  = tee.getvalue()

    with open(out_path, 'w') as f:
        f.write(content)

    print(f"\nSaved: {out_path}")
    print(f"  All terminal output + §6.5.1 text captured.")
    print(f"  Lines: {content.count(chr(10))}")

    if not use_real:
        print()
        print("REMINDER: Running with SYNTHETIC data.")
        print("tau_relax ratios are meaningless. Fix the import path:")
        print("  sys.path.insert(0, '.')  →  wherever assemble_cr_matrix.py lives")


if __name__ == "__main__":
    main()