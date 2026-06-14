"""
plot_regime_persistence_fixed.py
================================

Replot the Balmer regime persistence heatmap honestly:

1. Cells with last_t_ratio_abs_error_gt_10pct_us == 0 are shown in gray:
   threshold was never exceeded.
2. Cells with censored_gt_10pct == True are hatched:
   error was still above 10% at t_max, so the reported time is a lower bound.
3. Colorbar label is exactly:
   last time |ratio error| > 10% [us]
4. The figure note states that this is last exceedance time, not integrated duration.

Run from repo rates directory:
    cd /Users/phi/Desktop/non_markovian_cr/src/rates
    python plot_regime_persistence_fixed.py

Optional overwrite original output:
    python plot_regime_persistence_fixed.py --overwrite
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch, Rectangle


HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
DEFAULT_CSV = REPO / "data/processed/sensitivity/regime/Balmer_transient_regime_heatmap_DTe_p0p60.csv"
FIG_DIR = REPO / "figures/regime"
OUT_DIR = REPO / "data/processed/sensitivity/regime"


def cell_edges(x: np.ndarray, log: bool = False) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 1:
        return np.array([x[0] * 0.8, x[0] * 1.2])
    if log:
        lx = np.log10(x)
        mids = 0.5 * (lx[:-1] + lx[1:])
        first = lx[0] - (mids[0] - lx[0])
        last = lx[-1] + (lx[-1] - mids[-1])
        return 10.0 ** np.concatenate([[first], mids, [last]])
    mids = 0.5 * (x[:-1] + x[1:])
    first = x[0] - (mids[0] - x[0])
    last = x[-1] + (x[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def pivot(df: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    te_vals = np.array(sorted(df["Te_old_grid_eV"].unique()), dtype=float)
    ne_vals = np.array(sorted(df["ne_grid_cm-3"].unique()), dtype=float)
    Z = np.full((te_vals.size, ne_vals.size), np.nan, dtype=float)
    for i, te in enumerate(te_vals):
        for j, ne in enumerate(ne_vals):
            sub = df[
                np.isclose(df["Te_old_grid_eV"], te)
                & np.isclose(df["ne_grid_cm-3"], ne)
            ]
            if not sub.empty:
                Z[i, j] = float(sub.iloc[0][value_col])
    return te_vals, ne_vals, Z


def pivot_bool(df: pd.DataFrame, value_col: str, te_vals: np.ndarray, ne_vals: np.ndarray) -> np.ndarray:
    Z = np.zeros((te_vals.size, ne_vals.size), dtype=bool)
    for i, te in enumerate(te_vals):
        for j, ne in enumerate(ne_vals):
            sub = df[
                np.isclose(df["Te_old_grid_eV"], te)
                & np.isclose(df["ne_grid_cm-3"], ne)
            ]
            if not sub.empty:
                Z[i, j] = bool(sub.iloc[0][value_col])
    return Z


def add_hatches(ax, censored_mask: np.ndarray, ne_edges: np.ndarray, te_edges: np.ndarray) -> None:
    # Draw one hatched rectangle per censored cell.
    # Rectangles are in data coordinates; this works correctly with log x-axis.
    for i in range(censored_mask.shape[0]):
        for j in range(censored_mask.shape[1]):
            if not censored_mask[i, j]:
                continue
            x0 = ne_edges[j]
            x1 = ne_edges[j + 1]
            y0 = te_edges[i]
            y1 = te_edges[i + 1]
            rect = Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                facecolor="none",
                edgecolor="black",
                hatch="///",
                linewidth=0.0,
                zorder=5,
            )
            ax.add_patch(rect)


def make_display_csv(df: pd.DataFrame, out_csv: pathlib.Path) -> None:
    d = df.copy()

    def fmt(row):
        value = float(row["last_t_ratio_abs_error_gt_10pct_us"])
        cens = bool(row.get("censored_gt_10pct", False))
        if not np.isfinite(value) or value <= 0.0:
            return "<10% threshold not exceeded"
        if cens:
            return f">{value:.3g} us (censored at t_max)"
        return f"{value:.3g} us"

    d["last_t_ratio_abs_error_gt_10pct_display"] = d.apply(fmt, axis=1)
    d.to_csv(out_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=pathlib.Path, default=DEFAULT_CSV)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite original figure stem instead of writing *_fixed.")
    parser.add_argument("--delta-label", default=r"$\Delta T_e \approx 0.6$ eV")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    required = [
        "Te_old_grid_eV",
        "ne_grid_cm-3",
        "last_t_ratio_abs_error_gt_10pct_us",
        "censored_gt_10pct",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    te_vals, ne_vals, Z_raw = pivot(df, "last_t_ratio_abs_error_gt_10pct_us")
    censored = pivot_bool(df, "censored_gt_10pct", te_vals, ne_vals)

    # Honest treatment:
    # Zero means the threshold was not exceeded. Do not put zero on LogNorm.
    zero_or_missing = (~np.isfinite(Z_raw)) | (Z_raw <= 0.0)
    Z_plot = np.ma.masked_where(zero_or_missing, Z_raw)

    ne_edges = cell_edges(ne_vals, log=True)
    te_edges = cell_edges(te_vals, log=False)

    positive = Z_raw[np.isfinite(Z_raw) & (Z_raw > 0.0)]
    if positive.size == 0:
        raise SystemExit("No positive persistence values to plot.")
    vmin = max(float(np.nanmin(positive)), 1e-3)
    vmax = float(np.nanmax(positive))

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d9d9d9")  # gray = threshold not exceeded

    plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.grid": False})
    fig, ax = plt.subplots(figsize=(7.6, 5.8))

    im = ax.pcolormesh(
        ne_edges,
        te_edges,
        Z_plot,
        shading="auto",
        cmap=cmap,
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )

    add_hatches(ax, censored, ne_edges, te_edges)

    ax.set_xscale("log")
    ax.set_xlabel(r"$n_e$ grid value [cm$^{-3}$]")
    ax.set_ylabel(r"initial $T_e$ grid value [eV]")
    ax.set_title(rf"Last exceedance time of $H\alpha/H\beta$ bias above 10%, {args.delta_label}")

    # Benchmark marker: nearest grid point to nominal 3 eV, 1e14 cm^-3.
    te_b = te_vals[int(np.argmin(np.abs(te_vals - 3.0)))]
    ne_b = ne_vals[int(np.argmin(np.abs(ne_vals - 1e14)))]
    ax.plot(ne_b, te_b, marker="*", ms=15, color="cyan", mec="black", mew=0.5, label="benchmark grid point")

    legend_items = [
        Patch(facecolor="#d9d9d9", edgecolor="black", label="<10% threshold not exceeded"),
        Patch(facecolor="white", edgecolor="black", hatch="///", label=r"censored: still >10% at $t_{max}$"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + legend_items, fontsize=8, loc="best", framealpha=0.9)

    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r"last time $|$ratio error$| > 10\%$ [$\mu$s]")

    fig.text(
        0.5,
        0.01,
        "Gray cells: threshold was not exceeded. Hatched cells: value is a lower bound, because the error remained above 10% at tmax. "
        "This is last exceedance time, not integrated duration.",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    plt.tight_layout(rect=[0, 0.045, 1, 1])

    stem = "Balmer_regime_last_gt10_DTe_p0p60" if args.overwrite else "Balmer_regime_last_gt10_DTe_p0p60_fixed"
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    display_csv = OUT_DIR / "Balmer_transient_regime_heatmap_DTe_p0p60_persistence_display.csv"
    make_display_csv(df, display_csv)

    print("Saved:")
    print(FIG_DIR / f"{stem}.png")
    print(FIG_DIR / f"{stem}.pdf")
    print(display_csv)
    print()
    print("Caption language:")
    print(
        "Last exceedance time of the absolute Halpha/Hbeta ratio error above 10% after a DeltaTe≈0.6 eV step. "
        "Gray cells indicate that the 10% threshold was never exceeded. Hatched cells indicate censored cases in which "
        "the error remained above 10% at tmax=1 ms, so the plotted value is a lower bound, not an integrated duration."
    )


if __name__ == "__main__":
    main()
