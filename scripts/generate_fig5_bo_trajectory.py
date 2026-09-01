#!/usr/bin/env python3
"""Generate enhanced BO trajectory figure from optimization_results.csv.

Reads BO trial data and produces a publication-quality convergence plot
with acquisition-function switching annotations and uncertainty bands.
No model loading required — works from CSV data alone.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Ellipse
from pathlib import Path
import sys

# ── rcParams for publication quality ──────────────────────────────────
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 10,
    "font.family": "sans-serif",
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.8,
})

COLOURS = {
    "samples": "#CC3311",
    "best_curve": "#E69F00",
    "optimal_star": "#F0E442",
    "variance_fill": "#88CCEE",
    "variance_line": "#4477AA",
    "sampling_band": "#F3F3F3",
    "ei_band": "#DCEBFA",
    "ucb_band": "#FDE8D3",
    "logei_band": "#DDEFE6",
    "ei_text": "#4477AA",
    "ucb_text": "#D55E00",
    "logei_text": "#228833",
    "connector": "#777777",
}


def load_trial_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["iteration"] = np.arange(len(df))
    df["cumulative_best"] = np.maximum.accumulate(df["mean"].values)
    return df


def square_panel(ax: plt.Axes) -> None:
    ax.set_box_aspect(1)


def infer_search_split(n_rows: int, n_acquisition: int = 150) -> tuple[int, int, int, int]:
    """Return sampling end plus EI/UCB/LogEI absolute boundaries."""
    if n_rows <= n_acquisition:
        raise ValueError(f"Expected more than {n_acquisition} rows, got {n_rows}")
    n_sampling = n_rows - n_acquisition
    return n_sampling, n_sampling + 50, n_sampling + 100, n_sampling + 150


def plot_bo_convergence(df: pd.DataFrame, save_path: str,
                         n_sampling: int | None = None,
                         ei_end: int | None = None,
                         ucb_end: int | None = None,
                         logei_end: int | None = None):
    """Plot optimization progress using continuous sample-index coordinates."""
    if None in (n_sampling, ei_end, ucb_end, logei_end):
        n_sampling, ei_end, ucb_end, logei_end = infer_search_split(len(df))

    global_best_val = df["cumulative_best"].max()
    global_best_idx = df["cumulative_best"].idxmax()

    fig = plt.figure(figsize=(15.0, 7.2))
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        width_ratios=[4.2, 1.45],
        height_ratios=[3, 1],
        wspace=0.25, hspace=0.10,
    )
    ax_main = fig.add_subplot(gs[0, 0])
    ax_var = fig.add_subplot(gs[1, 0], sharex=ax_main)
    ax_zoom = fig.add_subplot(gs[:, 1])

    phase_defs = [
        (0, n_sampling, "Sampling", COLOURS["sampling_band"], "#666666", 0.07),
        (n_sampling, ei_end, "EI", COLOURS["ei_band"], COLOURS["ei_text"], 0.20),
        (ei_end, ucb_end, "UCB", COLOURS["ucb_band"], COLOURS["ucb_text"], 0.20),
        (ucb_end, logei_end, "LogEI", COLOURS["logei_band"], COLOURS["logei_text"], 0.20),
    ]

    y_top = max(df["mean"].max(), df["cumulative_best"].max()) * 1.04
    for ax in [ax_main, ax_var]:
        for start, end, _, colour, _, alpha in phase_defs:
            ax.axvspan(start, end, alpha=alpha, color=colour, zorder=0)
        for boundary in [n_sampling, ei_end, ucb_end]:
            ax.axvline(boundary, color="#666666", linewidth=0.8, linestyle=":", alpha=0.55)

    ax_main.scatter(
        df["iteration"], df["mean"],
        c=COLOURS["samples"], s=17, alpha=0.36,
        edgecolors="none",
        label="All samples", zorder=2,
    )
    ax_main.plot(
        df["iteration"], df["cumulative_best"].values,
        color=COLOURS["best_curve"], linewidth=2.0, linestyle="--",
        label="Best value curve", zorder=3,
    )
    ax_main.scatter(
        global_best_idx, df.loc[global_best_idx, "mean"],
        s=190, marker="*", c=COLOURS["optimal_star"],
        edgecolor="black", linewidth=0.7,
        label=f"Global optimum ({global_best_val:.2f})", zorder=5,
    )

    ax_main.text(n_sampling / 2, y_top, "Sampling", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold", color="#666666")

    bo_window = df.iloc[n_sampling:logei_end]
    ellipse_center = (n_sampling + 75, global_best_val - 0.12)
    magnifier = Ellipse(
        ellipse_center, width=170, height=0.78,
        facecolor="none", edgecolor="#333333", linewidth=1.2,
        linestyle="-", zorder=6,
    )
    ax_main.add_patch(magnifier)

    ax_main.set_ylabel("Target value")
    ax_main.set_title("Optimization progress with uncertainty tracking", pad=10)
    ax_main.legend(loc="upper left", frameon=False, handlelength=2.5)
    ax_main.grid(True, alpha=0.20, linewidth=0.6)
    ax_main.set_ylim(df["mean"].min() - 0.15, y_top + 0.3)
    ax_main.text(-0.06, 1.04, "a", transform=ax_main.transAxes,
                 ha="left", va="bottom", fontsize=13, fontweight="bold")

    ax_var.fill_between(
        df["iteration"], df["std"],
        color=COLOURS["variance_fill"], alpha=0.25,
    )
    ax_var.plot(
        df["iteration"], df["std"],
        color=COLOURS["variance_line"], linewidth=0.8, alpha=0.90,
    )
    ax_var.set_xlabel("Sample index")
    ax_var.set_ylabel("Variance")
    ax_var.grid(True, alpha=0.20, linewidth=0.6)
    ax_var.set_xlim(-0.5, len(df) - 0.5)

    for start, end, label, colour, text_colour, alpha in phase_defs[1:]:
        ax_zoom.axvspan(start, end, alpha=alpha + 0.04, color=colour, zorder=0)
        ax_zoom.text((start + end) / 2, 0.98, label, transform=ax_zoom.get_xaxis_transform(),
                     ha="center", va="top", fontsize=8.5, fontweight="bold",
                     color=text_colour)
    for boundary in [ei_end, ucb_end]:
        ax_zoom.axvline(boundary, color="#666666", linewidth=0.8,
                        linestyle=":", alpha=0.65)
    ax_zoom.scatter(
        bo_window["iteration"], bo_window["mean"],
        c=COLOURS["samples"], s=24, alpha=0.55,
        edgecolors="none", zorder=2,
    )
    ax_zoom.plot(
        bo_window["iteration"], bo_window["cumulative_best"],
        color=COLOURS["best_curve"], linewidth=2.0, linestyle="--", zorder=3,
    )
    ax_zoom.scatter(
        global_best_idx, df.loc[global_best_idx, "mean"],
        s=170, marker="*", c=COLOURS["optimal_star"],
        edgecolor="black", linewidth=0.7, zorder=5,
    )
    ax_zoom.set_title("Final 150 BO candidates", pad=10)
    ax_zoom.set_xlabel("Sample index")
    ax_zoom.set_ylabel("Target value")
    ax_zoom.set_xlim(n_sampling - 2, logei_end + 2)
    ax_zoom.set_ylim(bo_window["mean"].min() - 0.15, global_best_val + 0.25)
    ax_zoom.grid(True, alpha=0.20, linewidth=0.6)
    ax_zoom.tick_params(axis="x", labelrotation=45)
    ax_zoom.text(-0.18, 1.04, "b", transform=ax_zoom.transAxes,
                 ha="left", va="bottom", fontsize=13, fontweight="bold")

    connector_pairs = [
        ((ellipse_center[0] + 85, ellipse_center[1] + 0.30), (0.0, 1.0)),
        ((ellipse_center[0] + 85, ellipse_center[1] - 0.30), (0.0, 0.0)),
    ]
    for xy_main, xy_zoom in connector_pairs:
        fig.add_artist(ConnectionPatch(
            xyA=xy_main, coordsA=ax_main.transData,
            xyB=xy_zoom, coordsB=ax_zoom.transAxes,
            color=COLOURS["connector"], linewidth=0.7, alpha=0.42,
        ))

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_param_space_exploration(df: pd.DataFrame, save_path: str,
                                  n_sobol: int = 5300):
    """Plot parameter-space coverage coloured by predicted RRR."""
    df_bo = df.iloc[n_sobol:].copy()
    params = [
        ("oxygen_pressure", "Temperature", "temperature",
         "$P_{\\mathrm{O_2}}$ (mbar)", "$T$ (°C)", True),
        ("oxygen_pressure", "Laser fluence", "laser_energy_density",
         "$P_{\\mathrm{O_2}}$ (mbar)", "$F$ (J/cm²)", True),
        ("temperature", "Laser fluence", "laser_energy_density",
         "$T$ (°C)", "$F$ (J/cm²)", False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (col_x, label_x, col_y, xlabel, ylabel, log_x) in zip(axes, params):
        sc = ax.scatter(
            df_bo[col_x], df_bo[col_y],
            c=df_bo["mean"], cmap="viridis", s=25, alpha=0.6, edgecolors="none",
        )
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.25)
        square_panel(ax)

        # Mark best point
        best = df_bo.loc[df_bo["mean"].idxmax()]
        ax.scatter(best[col_x], best[col_y], s=180, marker="*",
                   c=COLOURS["optimal_star"], edgecolor="black", linewidth=0.6,
                   zorder=5)

    cbar = fig.colorbar(sc, ax=axes, orientation="vertical", fraction=0.02, pad=0.03)
    cbar.set_label("Predicted RRR", fontsize=11)

    fig.suptitle("Parameter-Space Coverage During BO — LLM-Constrained / Series",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    PROJECT = Path(__file__).resolve().parents[1]
    MODEL_DIR = PROJECT / "LLM/XGB_BNN_series_hybrid_model"
    OUT_DIR = PROJECT / "npj_ComputMater_Manuscript.assets/generated"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Primary trial for the manuscript figure.
    trial = 1683
    csv_path = MODEL_DIR / f"pareto_solution/model_optim_prediction_results_trial_{trial}_fix_freq_4.0/optimization_results.csv"

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        sys.exit(1)

    print(f"Loading {csv_path}")
    df = load_trial_data(str(csv_path))
    n_sampling, ei_end, ucb_end, logei_end = infer_search_split(len(df))
    print(f"Data: {len(df)} total rows")
    print(f"Sampling rows: 0-{n_sampling - 1}")
    print(f"Acquisition rows: EI {n_sampling}-{ei_end - 1}, UCB {ei_end}-{ucb_end - 1}, LogEI {ucb_end}-{logei_end - 1}")
    print(f"Best RRR: {df['cumulative_best'].max():.3f} at sample index {df['cumulative_best'].idxmax()}")

    plot_bo_convergence(
        df,
        str(OUT_DIR / "Fig5_BO_convergence_enhanced.png"),
        n_sampling=n_sampling, ei_end=ei_end, ucb_end=ucb_end, logei_end=logei_end,
    )

    plot_param_space_exploration(
        df,
        str(OUT_DIR / "Fig5_BO_paramspace_enhanced.png"),
        n_sobol=n_sampling,
    )

    print("Done.")
