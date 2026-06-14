"""
qc_ccc.py
=========
Quality control for the CCC electron-impact cross-section database.
Run after parse_ccc.py.

Checks:
  1. Threshold compliance  — sigma=0 below excitation threshold
  2. Detailed balance      — g_i*sigma_exc*E_i = g_f*sigma_deexc*E_f  (at E > 2*thr)
  3. Coverage              — all (ni,li,nf,lf) pairs present for n=1-9, delta_n != 0
  4. Physics scaling       — 1S->nP sigma_max decreases with n
  5. Spot-check plots      — sigma(E) for key transitions

Usage:
  python qc_ccc.py <ccc_csv> <output_dir>
  python src/parsers/qc_ccc.py data/processed/collisions/ccc/ccc_crosssections.csv figures/week2

Output:
  <output_dir>/ccc_qc_report.png   — QC summary figure
  Printed pass/fail for each check
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from collections import defaultdict


# ── Constants ──────────────────────────────────────────────────────────────────
L = ['S', 'P', 'D', 'F', 'G', 'H', 'I', 'J', 'K']

def E_n(n):
    """Hydrogen energy level (eV). E_n = -13.6 / n^2"""
    return -13.6 / n**2

def threshold(ni, nf):
    """Excitation threshold (eV). Positive for excitation (nf > ni)."""
    return E_n(nf) - E_n(ni)


# ── Check 1: Threshold compliance ─────────────────────────────────────────────

def check_threshold(df, grp):
    """
    Verify sigma=0 below excitation threshold for all excitation transitions.
    Tolerance: 10 meV (accounts for grid spacing near threshold).
    """
    print("\n=== CHECK 1: THRESHOLD COMPLIANCE ===")
    violations = []

    for (ni, li, nf, lf), g in grp:
        thr = threshold(ni, nf)
        if thr <= 0:
            continue  # de-excitation direction stored; skip
        below = g[g.E_eV < thr - 0.01]
        if len(below) > 0:
            violations.append({
                'transition': f"{ni}{L[li]}->{nf}{L[lf]}",
                'threshold_eV': thr,
                'E_min_below': below.E_eV.min(),
                'sigma_max_below': below.sigma_a0sq.max(),
            })

    n_exc = sum(1 for (ni,_,nf,__) in grp.groups if threshold(ni,nf) > 0)
    print(f"  Excitation transitions checked : {n_exc}")
    print(f"  Violations found              : {len(violations)}")

    if violations:
        print("  FAIL — violations:")
        for v in violations[:10]:
            print(f"    {v['transition']:15s}  thr={v['threshold_eV']:.3f} eV  "
                  f"E_min={v['E_min_below']:.4f} eV  sigma={v['sigma_max_below']:.3e} a0^2")
        status = "FAIL"
    else:
        print("  PASS")
        status = "PASS"

    return status, violations


# ── Check 2: Detailed balance ──────────────────────────────────────────────────

def check_detailed_balance(df, grp):
    """
    For each excitation/de-excitation pair, verify:
        g_i * sigma_exc(E_i) * E_i  =  g_f * sigma_deexc(E_f) * E_f
    where E_f = E_i - threshold.

    Tested at 8 energy points with E > 2*threshold to avoid near-threshold
    interpolation noise (the Maxwellian weight at E ~ threshold is negligible
    for Te = 1-10 eV).

    Returns ratio statistics and per-pair worst violations.
    """
    print("\n=== CHECK 2: DETAILED BALANCE ===")
    keys = set(grp.groups.keys())
    all_ratios = []
    pair_stats = []

    for (ni, li, nf, lf) in keys:
        thr = threshold(ni, nf)
        if thr <= 0:
            continue  # only test excitation direction
        rk = (nf, lf, ni, li)
        if rk not in keys:
            continue  # reverse transition not in database

        exc   = grp.get_group((ni, li, nf, lf)).sort_values('E_eV')
        deexc = grp.get_group(rk).sort_values('E_eV')

        # Test at E well above threshold.
        # For small gaps (e.g. n=7->8, thr~0.065 eV), 2*thr is still in the
        # steep near-threshold region. Use max(2*thr, thr+1.0) to ensure
        # we are in the smooth high-energy regime.
        E_lo = max(exc.E_eV.min(), thr * 2.0, thr + 1.0)
        E_hi = min(exc.E_eV.max(), deexc.E_eV.max() + thr) * 0.85
        if E_lo >= E_hi:
            continue

        E_test = np.linspace(E_lo, E_hi, 8)
        sig_exc   = np.interp(E_test,       exc.E_eV.values,   exc.sigma_a0sq.values)
        sig_deexc = np.interp(E_test - thr, deexc.E_eV.values, deexc.sigma_a0sq.values)

        gi = 2 * li + 1
        gf = 2 * lf + 1
        # Detailed balance: g_i * sigma_exc * E_i = g_f * sigma_deexc * E_f
        ratio = (sig_exc * gi * E_test) / (sig_deexc * gf * (E_test - thr))

        all_ratios.extend(ratio.tolist())
        pair_stats.append({
            'transition': f"{ni}{L[li]}->{nf}{L[lf]}",
            'threshold_eV': thr,
            'ratio_mean': ratio.mean(),
            'ratio_min':  ratio.min(),
            'ratio_max':  ratio.max(),
        })

    ratios = np.array(all_ratios)
    within_1pct  = (np.abs(ratios - 1) < 0.01).mean() * 100
    within_5pct  = (np.abs(ratios - 1) < 0.05).mean() * 100
    within_10pct = (np.abs(ratios - 1) < 0.10).mean() * 100

    print(f"  Pairs tested      : {len(pair_stats)}")
    print(f"  Test points       : {len(ratios)}")
    print(f"  Ratio mean        : {ratios.mean():.4f}  (expect 1.000)")
    print(f"  Ratio std         : {ratios.std():.4f}")
    print(f"  Within 1%         : {within_1pct:.1f}%")
    print(f"  Within 5%         : {within_5pct:.1f}%")
    print(f"  Within 10%        : {within_10pct:.1f}%")

    status = "PASS" if within_1pct > 95.0 else "FAIL"
    print(f"  {status}")

    # Top 5 worst violators
    pair_stats.sort(key=lambda x: abs(x['ratio_mean'] - 1), reverse=True)
    print(f"\n  Top 5 worst violators (ratio_mean furthest from 1.0):")
    print(f"  {'Transition':15s}  {'thr(eV)':>8s}  {'mean':>8s}  {'min':>8s}  {'max':>8s}")
    for p in pair_stats[:5]:
        print(f"  {p['transition']:15s}  {p['threshold_eV']:8.3f}  "
              f"{p['ratio_mean']:8.4f}  {p['ratio_min']:8.4f}  {p['ratio_max']:8.4f}")

    return status, ratios, pair_stats


# ── Check 3: Coverage ──────────────────────────────────────────────────────────

def check_coverage(df, grp, n_max=9):
    """
    Verify all (ni,li,nf,lf) pairs with ni != nf are present for n=1..n_max.
    li in [0..ni-1], lf in [0..nf-1].
    """
    print(f"\n=== CHECK 3: COVERAGE (n=1..{n_max}, delta_n != 0) ===")

    expected = set()
    for ni in range(1, n_max + 1):
        for li in range(ni):
            for nf in range(1, n_max + 1):
                for lf in range(nf):
                    if ni != nf:
                        expected.add((ni, li, nf, lf))

    have    = set(zip(df.n_i, df.l_i, df.n_f, df.l_f))
    missing = expected - have
    extra   = have - expected  # transitions outside expected range

    print(f"  Expected : {len(expected)}")
    print(f"  Present  : {len(have)}")
    print(f"  Missing  : {len(missing)}")
    print(f"  Extra    : {len(extra)}  (transitions outside n=1-{n_max})")

    if missing:
        print("  FAIL — missing transitions:")
        miss_by_ni = defaultdict(list)
        for (ni, li, nf, lf) in sorted(missing):
            miss_by_ni[ni].append(f"{nf}{L[lf]}")
        for ni in sorted(miss_by_ni):
            print(f"    n_i={ni}: {sorted(set(miss_by_ni[ni]))}")
        status = "FAIL"
    else:
        print("  PASS")
        status = "PASS"

    return status, missing


# ── Check 4: Physics scaling ───────────────────────────────────────────────────

def check_physics_scaling(df, grp):
    """
    1S -> nP: sigma_max should decrease monotonically with n.
    Physical reason: matrix element |<1s|r|np>|^2 falls with n,
    approximately as n^-3 for high n.
    """
    print("\n=== CHECK 4: PHYSICS SCALING (1S->nP sigma_max) ===")
    keys = set(grp.groups.keys())

    ns, sigma_max_list, thr_list = [], [], []
    for nf in range(2, 10):
        key = (1, 0, nf, 1)   # 1S -> nP
        if key not in keys:
            continue
        d = grp.get_group(key)
        ns.append(nf)
        sigma_max_list.append(d.sigma_a0sq.max())
        thr_list.append(threshold(1, nf))

    print(f"  {'n_f':>4}  {'thr(eV)':>10}  {'sigma_max(a0^2)':>16}  {'ratio_to_n2':>12}")
    sig_n2 = sigma_max_list[0] if sigma_max_list else 1.0
    for nf, sm, t in zip(ns, sigma_max_list, thr_list):
        print(f"  {nf:>4}  {t:>10.3f}  {sm:>16.4f}  {sm/sig_n2:>12.4f}")

    # Check monotonic decrease
    monotonic = all(sigma_max_list[i] > sigma_max_list[i+1]
                    for i in range(len(sigma_max_list)-1))
    status = "PASS" if monotonic else "FAIL"
    print(f"  Monotonic decrease: {monotonic}  →  {status}")

    return status, ns, sigma_max_list, thr_list


# ── Plots ──────────────────────────────────────────────────────────────────────

def make_qc_figure(df, grp, ratios, ns, sigma_max_list, output_path):
    """
    6-panel QC figure:
      Row 1: sigma(E) spot checks for 1S->2P, 1S->3P, 2S->3P
      Row 2: detailed balance histogram, coverage heatmap, 1S->nP scaling
    """
    keys = set(grp.groups.keys())
    fig = plt.figure(figsize=(16, 10))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Row 1: sigma(E) spot checks ───────────────────────────────────────
    spot = [
        (1, 0, 2, 1, '1S\u21922P'),
        (1, 0, 3, 1, '1S\u21923P'),
        (2, 0, 3, 1, '2S\u21923P'),
    ]
    for idx, (ni, li, nf, lf, label) in enumerate(spot):
        ax = fig.add_subplot(gs[0, idx])
        thr = threshold(ni, nf)

        # Excitation direction
        if (ni, li, nf, lf) in keys:
            d_exc = grp.get_group((ni, li, nf, lf)).sort_values('E_eV')
            ax.plot(d_exc.E_eV, d_exc.sigma_a0sq, 'b-', lw=1.8, label='excitation')

        # De-excitation direction (stored as n_i=nf, n_f=ni in CSV)
        if (nf, lf, ni, li) in keys:
            d_deexc = grp.get_group((nf, lf, ni, li)).sort_values('E_eV')
            ax.plot(d_deexc.E_eV, d_deexc.sigma_a0sq, 'r--', lw=1.2,
                    alpha=0.7, label='de-excitation')

        ax.axvline(thr, color='k', ls=':', lw=1.2,
                   label=f'threshold {thr:.2f} eV')
        ax.set_xlabel('Incident electron energy (eV)', fontsize=9)
        ax.set_ylabel('\u03c3 (a\u2080\u00b2)', fontsize=9)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # ── Row 2, Col 1: Detailed balance histogram ──────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(ratios, bins=60, color='steelblue', edgecolor='white', lw=0.3,
             range=(0.85, 1.15))
    ax4.axvline(1.0, color='r', ls='--', lw=2, label='ideal = 1.000')
    ax4.set_xlabel('Ratio: (g\u1d62\u03c3\u1d49\u02e3\u02b2E\u1d62) / (g\u1da0\u03c3\u1d30\u1d49\u02e3E\u1da0)', fontsize=9)
    ax4.set_ylabel('Count', fontsize=10)
    ax4.set_title(
        f'Detailed Balance (E > 2\u00d7threshold)\n'
        f'mean={ratios.mean():.4f}, std={ratios.std():.4f}',
        fontsize=10
    )
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    # Annotate within-1% fraction
    within_1 = (np.abs(ratios - 1) < 0.01).mean() * 100
    ax4.text(0.05, 0.92, f'{within_1:.1f}% within 1%',
             transform=ax4.transAxes, fontsize=9,
             bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))

    # ── Row 2, Col 2: Coverage heatmap ────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    trans_count = (df.groupby(['n_i', 'n_f'])
                     .apply(lambda x: len(x.groupby(['l_i', 'l_f'])),
                            include_groups=False)
                     .unstack(fill_value=0))

    im = ax5.imshow(trans_count.values, aspect='auto', cmap='Blues',
                    origin='lower')
    ax5.set_xticks(range(len(trans_count.columns)))
    ax5.set_xticklabels(trans_count.columns, fontsize=8)
    ax5.set_yticks(range(len(trans_count.index)))
    ax5.set_yticklabels(trans_count.index, fontsize=8)
    ax5.set_xlabel('n\u1da0 (final)', fontsize=10)
    ax5.set_ylabel('n\u1d62 (initial)', fontsize=10)
    ax5.set_title('\u2113-resolved transitions\nper (n\u1d62, n\u1da0) pair', fontsize=10)
    plt.colorbar(im, ax=ax5, label='# transitions')
    vmax = trans_count.values.max()
    for i in range(len(trans_count.index)):
        for j in range(len(trans_count.columns)):
            v = trans_count.values[i, j]
            if v > 0:
                ax5.text(j, i, str(v), ha='center', va='center',
                         fontsize=7,
                         color='white' if v > vmax * 0.6 else 'black')

    # ── Row 2, Col 3: 1S->nP sigma_max scaling ────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(ns, sigma_max_list, 'bo-', lw=2, ms=8)

    # Overlay n^-3 scaling guide
    n_arr = np.array(ns, dtype=float)
    scale = sigma_max_list[0] * (ns[0] ** 3)
    ax6.plot(n_arr, scale / n_arr**3, 'k--', lw=1, alpha=0.5,
             label='n\u207b\u00b3 guide')

    ax6.set_xlabel('n (final state)', fontsize=10)
    ax6.set_ylabel('\u03c3\u2098\u2090\u02e3 (a\u2080\u00b2)', fontsize=10)
    ax6.set_title('Physics: 1S\u2192nP\n\u03c3\u2098\u2090\u02e3 vs n', fontsize=10)
    ax6.set_yscale('log')
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=9)
    for nx, sx in zip(ns, sigma_max_list):
        ax6.annotate(f'n={nx}', (nx, sx),
                     textcoords='offset points', xytext=(5, 3), fontsize=8)

    fig.suptitle('CCC Cross-Section Database — QC Report',
                 fontsize=14, fontweight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def run_qc(csv_path, output_dir='.'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {csv_path}")
    df  = pd.read_csv(csv_path)
    grp = df.groupby(['n_i', 'l_i', 'n_f', 'l_f'])
    print(f"Loaded {len(df):,} rows, {len(grp)} transitions")

    # Run all checks
    s1, violations      = check_threshold(df, grp)
    s2, ratios, pairs   = check_detailed_balance(df, grp)
    s3, missing         = check_coverage(df, grp)
    s4, ns, sigmas, thr = check_physics_scaling(df, grp)

    # Figure
    fig_path = output_dir / 'ccc_qc_report.png'
    make_qc_figure(df, grp, ratios, ns, sigmas, fig_path)

    # Summary
    print("\n" + "=" * 45)
    print("  QC SUMMARY")
    print("=" * 45)
    checks = [
        ("1. Threshold compliance", s1),
        ("2. Detailed balance    ", s2),
        ("3. Coverage            ", s3),
        ("4. Physics scaling     ", s4),
    ]
    all_pass = True
    for name, status in checks:
        mark = "\u2705" if status == "PASS" else "\u274c"
        print(f"  {mark}  {name} : {status}")
        if status != "PASS":
            all_pass = False
    print("=" * 45)
    print(f"  Overall: {'ALL PASS' if all_pass else 'FAILURES PRESENT'}")
    print("=" * 45)

    return all_pass


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python qc_ccc.py <ccc_csv> [output_dir]")
        print("Example: python qc_ccc.py data/processed/collisions/ccc/ccc_crosssections.csv figures/week2")
        sys.exit(1)

    csv_path   = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '.'
    run_qc(csv_path, output_dir)