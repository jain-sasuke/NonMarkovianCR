"""
escape_factor.py
================
Population escape factor for Lyman-alpha radiation trapping.

Implements the ADAS214 population escape factor (Behringer 1998, IPP 10/11)
by direct numerical quadrature — no fitting formula.

Method
------
The population escape factor Θ_P modifies the spontaneous emission rate:
    A_eff = Θ_P × A

For a homogeneous slab of half-thickness b, with constant emission (e1),
isotropic radiation field (sphere, g1), and a Doppler line profile (p1),
the population escape factor evaluated at the plasma centre is
(ADAS214 eq. 3.14.14, constant-emission case):

    Θ_P(τ₀) = ∫ φ(x) exp(-τ₀ φ(x)) dx

where:
    φ(x) = (1/√π) exp(-x²)  is the normalised Doppler profile
    τ₀ = κ₀ b = n(1s) σ₀ b  is the line-centre optical depth
                              over half the slab thickness
    x = (ν - ν₀) / Δν_D     is the normalised frequency offset

Physical meaning: Θ_P is the fraction of emitted photons from the centre
that are NOT reabsorbed on their way out. When Θ_P = 1 (optically thin),
all photons escape. When Θ_P → 0 (optically thick), most are reabsorbed.

NOTE ON τ_c CONVENTION:
    τ_c = κ₀ × b is the line-centre optical depth over HALF the slab.
    If D is the full slab thickness, then τ_c = κ₀ × D/2.
    The Doppler profile SHAPE in the optical depth is exp(-x²),
    NOT the normalised profile φ(x) = (1/√π)exp(-x²).
    This √π factor matters and must not be confused.

References
----------
[1] Behringer K (1998) IPP Report 10/11 (escape factor code)
[2] ADAS214 manual, www.adas.ac.uk/man/chap2-14.pdf
[3] Fujimoto T (2004) Plasma Spectroscopy, Ch. 8
[4] Holstein T (1947) Phys. Rev. 72, 1212
[5] Irons F E (1979) JQSRT 22, 1
"""

import numpy as np
from scipy.integrate import quad


# ═══════════════════════════════════════════════════════════════════════════════
# Core escape factor computation
# ═══════════════════════════════════════════════════════════════════════════════

def _doppler_profile(x):
    """Normalised Doppler profile: φ(x) = (1/√π) exp(-x²)."""
    return np.exp(-x**2) / np.sqrt(np.pi)


def escape_factor_quadrature(tau_c, rtol=1e-10):
    """
    Population escape factor by direct numerical quadrature.

    Evaluates:
        Θ_P = (1/√π) ∫ exp(-x²) exp(-τ_c exp(-x²)) dx

    where τ_c is the line-centre optical depth over the half-slab.
    The Doppler profile shape in the optical depth is exp(-x²),
    NOT the normalised profile φ(x) = (1/√π)exp(-x²).

    Parameters
    ----------
    tau_c : float
        Line-centre optical depth over half-slab thickness b.
        τ_c = κ₀ × b = n(1s) × σ₀ × (D/2)
    rtol : float
        Relative tolerance for quadrature.

    Returns
    -------
    theta_P : float
        Population escape factor, 0 < Θ_P ≤ 1.
    """
    if tau_c < 1e-6:
        return 1.0

    def integrand(x):
        gauss = np.exp(-x**2)
        return _doppler_profile(x) * np.exp(-tau_c * gauss)

    result, error = quad(integrand, -15, 15, limit=200, epsrel=rtol)
    return np.clip(result, 0.0, 1.0)


def escape_factor_slab(n_1s, sigma0, D_cm):
    """
    Population escape factor for a homogeneous slab.

    Parameters
    ----------
    n_1s   : float  Ground-state density [cm⁻³]
    sigma0 : float  Line-centre absorption cross section [cm²]
    D_cm   : float  Full slab thickness [cm]

    Returns
    -------
    dict with keys:
        tau_c     : line-centre optical depth over half-slab
        tau_full  : line-centre optical depth over full slab
        theta_P   : population escape factor
    """
    kappa0 = n_1s * sigma0          # absorption coefficient [cm⁻¹]
    tau_c = kappa0 * D_cm / 2       # optical depth over half-slab
    tau_full = kappa0 * D_cm        # optical depth over full slab
    theta_P = escape_factor_quadrature(tau_c)
    return {
        'tau_c': tau_c,
        'tau_full': tau_full,
        'theta_P': theta_P,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Vectorised version for parameter scans
# ═══════════════════════════════════════════════════════════════════════════════

def escape_factor_array(tau_c_array):
    """Vectorised wrapper: evaluate Θ_P for an array of τ_c values."""
    tau_c_array = np.atleast_1d(tau_c_array)
    return np.array([escape_factor_quadrature(t) for t in tau_c_array.flat]
                    ).reshape(tau_c_array.shape)


# ═══════════════════════════════════════════════════════════════════════════════
# Lyman-alpha cross section
# ═══════════════════════════════════════════════════════════════════════════════

def lyman_alpha_sigma0(T_at_eV):
    """
    Line-centre absorption cross section for Lyman-alpha.

    Uses the Ladenburg relation (ADAS214 eq. 3.14.5):
        σ₀ = (πe²/m_e c) × f₁₂ / (√π × Δν_D)

    Parameters
    ----------
    T_at_eV : float
        Neutral atom kinetic temperature [eV].

    Returns
    -------
    sigma0 : float  [cm²]
    """
    # CGS constants
    e_esu     = 4.80326e-10    # statcoulomb
    m_e       = 9.10938e-28    # g
    c_cgs     = 2.99792e10     # cm/s
    m_H       = 1.67262e-24    # g
    eV_to_erg = 1.60218e-12

    # Lyman-alpha
    f_12    = 0.4162           # absorption oscillator strength 1s→2p
    lambda0 = 1.21567e-5       # cm (121.567 nm)
    nu0     = c_cgs / lambda0

    # Doppler half-width at 1/e
    v_th = np.sqrt(2 * T_at_eV * eV_to_erg / m_H)
    Delta_nu_D = (nu0 / c_cgs) * v_th

    # Ladenburg relation
    prefactor = np.pi * e_esu**2 / (m_e * c_cgs)   # 2.654e-2 cm² Hz
    sigma0 = prefactor * f_12 / (np.sqrt(np.pi) * Delta_nu_D)

    return sigma0


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate():
    """
    Validate the escape factor implementation.

    Test 1: Optically thin limit  → Θ_P = 1
    Test 2: Large τ_c asymptotic  → Θ_P ~ 1/[τ_c √(π ln(τ_c/√π))]
            (Holstein 1947 / Fujimoto 2004 eq. 8.34)
    Test 3: Intermediate values   → monotonically decreasing
    Test 4: σ₀ spot-check
    Test 5: Small-τ expansion     → Θ_P ≈ 1 - τ_c/√2
    """
    print("=" * 72)
    print("ESCAPE FACTOR VALIDATION (CORRECTED)")
    print("Method: Direct quadrature of ADAS214 population escape factor")
    print("        Θ_P = (1/√π) ∫ exp(-x²) exp(-τ_c exp(-x²)) dx")
    print("        τ_c = κ₀ × D/2 (line-centre optical depth, half-slab)")
    print("=" * 72)

    # ── Test 1: Thin limit ─────────────────────────────────────────────
    print("\n── Test 1: Optically thin limit ──")
    for tau in [0, 1e-8, 1e-4, 1e-2]:
        theta = escape_factor_quadrature(tau)
        print(f"  τ_c = {tau:.1e}  →  Θ_P = {theta:.8f}")
    thin_ok = abs(escape_factor_quadrature(0) - 1.0) < 1e-10
    print(f"  PASS: Θ_P(0) = 1? {thin_ok}")

    # ── Test 5: Small-τ expansion ──────────────────────────────────────
    print("\n── Test 5: Small-τ_c expansion ──")
    print("  For τ_c << 1: Θ_P ≈ 1 - τ_c/√2")
    for tau in [0.001, 0.01, 0.1]:
        theta = escape_factor_quadrature(tau)
        linear = 1 - tau / np.sqrt(2)
        print(f"  τ_c = {tau:.3f}: Θ_P = {theta:.6f}, "
              f"linear = {linear:.6f}, diff = {abs(theta-linear):.2e}")

    # ── Test 2: Large-τ_c asymptotic ────────────────────────────────────
    print("\n── Test 2: Large-τ_c asymptotic (Holstein/Fujimoto) ──")
    print("  Holstein slab: g₀ ~ 1 / [τ_c √(π ln(τ_c/√π))]")
    print()
    print(f"  {'τ_c':>10s}  {'Θ_P (quad)':>12s}  {'Holstein':>12s}  {'ratio':>8s}")
    print(f"  {'─'*10}  {'─'*12}  {'─'*12}  {'─'*8}")

    for tau in [10, 30, 100, 300, 1000, 3000, 10000]:
        theta = escape_factor_quadrature(tau)
        if tau > np.sqrt(np.pi):
            g0 = 1.0 / (tau * np.sqrt(np.pi * np.log(tau / np.sqrt(np.pi))))
        else:
            g0 = float('nan')
        ratio = theta / g0 if not np.isnan(g0) else float('nan')
        print(f"  {tau:10.1f}  {theta:12.6e}  {g0:12.6e}  {ratio:8.4f}")

    print()
    print("  (Ratio should approach 1.0 for large τ_c)")

    # ── Test 3: Monotonicity ───────────────────────────────────────────
    print("\n── Test 3: Monotonicity ──")
    taus = np.logspace(-3, 5, 200)
    thetas = escape_factor_array(taus)
    is_mono = np.all(np.diff(thetas) <= 0)
    print(f"  Θ_P monotonically decreasing? {is_mono}")

    # ── Test 4: σ₀ spot-check ──────────────────────────────────────────
    print("\n── Test 4: σ₀ spot-check ──")
    s3 = lyman_alpha_sigma0(3.0)
    s1 = lyman_alpha_sigma0(1.0)
    print(f"  σ₀(T_at=3 eV) = {s3:.4e} cm²  (expected: 3.16e-14)")
    print(f"  σ₀(T_at=1 eV) = {s1:.4e} cm²  (expected: 5.47e-14)")

    # ── Full table ─────────────────────────────────────────────────────
    print("\n── Publication-ready table: Θ_P vs τ_c ──")
    print(f"  {'τ_c':>10s}  {'τ_full':>10s}  {'Θ_P':>12s}")
    print(f"  {'─'*10}  {'─'*10}  {'─'*12}")
    for tc in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3,
               1, 3, 10, 30, 100, 300, 1000, 3000, 10000]:
        theta = escape_factor_quadrature(tc)
        print(f"  {tc:10.3f}  {2*tc:10.3f}  {theta:12.6e}")

    # ── ITER scenario table ────────────────────────────────────────────
    print("\n── ITER scenario estimates ──")
    print("  (Sensitivity bounds, NOT predictions)")
    print()
    scenarios = [
        ("Attached",           3.0, 1e14, 1e12, 1.0, 1.0),
        ("Attached (high n0)", 3.0, 1e14, 1e13, 1.0, 1.0),
        ("Partial detach",     1.5, 3e14, 1e13, 0.5, 1.0),
        ("Partial detach",     1.5, 3e14, 1e13, 0.5, 5.0),
        ("Detached (OOS)",     1.0, 1e15, 1e14, 0.3, 1.0),
        ("Detached (OOS)",     1.0, 1e15, 1e14, 0.3, 5.0),
    ]
    print(f"  {'Scenario':<22s} {'Te':>4s} {'ne':>8s} {'n(1s)':>8s} "
          f"{'T_at':>4s} {'D':>4s} {'τ_c':>8s} {'τ_full':>8s} {'Θ_P':>8s}")
    print(f"  {'─'*22} {'─'*4} {'─'*8} {'─'*8} {'─'*4} {'─'*4} {'─'*8} {'─'*8} {'─'*8}")
    for name, Te, ne, n1s, Tat, D in scenarios:
        sig = lyman_alpha_sigma0(Tat)
        res = escape_factor_slab(n1s, sig, D)
        print(f"  {name:<22s} {Te:4.1f} {ne:8.0e} {n1s:8.0e} "
              f"{Tat:4.1f} {D:4.1f} {res['tau_c']:8.3f} "
              f"{res['tau_full']:8.3f} {res['theta_P']:8.4f}")

    print("\n" + "=" * 72)
    print("VALIDATION COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    validate()