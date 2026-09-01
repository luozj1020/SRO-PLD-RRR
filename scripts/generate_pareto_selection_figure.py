#!/usr/bin/env python3
"""Generate a manuscript figure for two-stage five-objective Pareto selection."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "npj_ComputMater_Manuscript.assets/generated/Fig4_pareto_selection.png"

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "font.family": "sans-serif",
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.8,
})

MODE_COLOURS = {
    "LLM": "#4477AA",
    "tradition": "#CC6677",
}

FUSION_LABELS = {
    "attention": "Attention",
    "series": "Series",
    "uncertainty_1": "Uncert.\nguided",
    "uncertainty_2": "Bayes.\nweighted",
}


def load_pareto_tables() -> pd.DataFrame:
    frames = []
    for mode in ["LLM", "tradition"]:
        for path in (ROOT / mode).glob("XGB_BNN_*_hybrid_model/pareto_solution/pareto_frontier.csv"):
            model_type = path.parts[-3].replace("XGB_BNN_", "").replace("_hybrid_model", "")
            df = pd.read_csv(path)
            df["mode"] = mode
            df["model_type"] = model_type
            frames.append(df)
    pareto = pd.concat(frames, ignore_index=True)

    score_frames = [
        pd.read_csv(ROOT / "model_comprehensive_scores_LLM.csv"),
        pd.read_csv(ROOT / "model_comprehensive_scores_tradition.csv"),
    ]
    scores = pd.concat(score_frames, ignore_index=True)[
        ["mode", "model_type", "trial_number", "final_score"]
    ]
    return pareto.merge(scores, on=["mode", "model_type", "trial_number"], how="left")


def minmax(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = values.astype(float)
    lo, hi = values.min(), values.max()
    if np.isclose(hi, lo):
        scaled = pd.Series(np.full(len(values), 0.5), index=values.index)
    else:
        scaled = (values - lo) / (hi - lo)
    return scaled if higher_is_better else 1 - scaled


def add_panel_label(ax, label: str, x: float = -0.16, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=12, fontweight="bold")


def main() -> None:
    pareto = load_pareto_tables()

    pareto["secondary_desirability"] = np.nan
    for mode, higher in [("LLM", True), ("tradition", True)]:
        mask = pareto["mode"].eq(mode)
        pareto.loc[mask, "secondary_desirability"] = minmax(
            pareto.loc[mask, "secondary_score"], higher_is_better=higher
        )

    objective_cols = [
        ("sto_r2_trial", "STO $R^2$\npretrain", True),
        ("secondary_desirability", "Secondary\nobjective", True),
        ("stability_score", "Training\nstability", True),
        ("experiment_data_r2", "Local exp.\n$R^2$", True),
        ("original_sto_r2", "STO retention\n$R^2$", True),
    ]
    for col, _, higher in objective_cols:
        norm_col = f"norm_{col}"
        pareto[norm_col] = minmax(pareto[col], higher_is_better=higher)

    fig = plt.figure(figsize=(7.2, 7.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.05, 1.0],
                          wspace=0.36, hspace=0.35)
    ax_parallel = fig.add_subplot(gs[0, :])
    ax_tradeoff = fig.add_subplot(gs[1, 0])
    ax_counts = fig.add_subplot(gs[1, 1])

    x = np.arange(len(objective_cols))
    for _, row in pareto.iterrows():
        is_selected = row["mode"] == "LLM" and row["model_type"] == "series" and row["trial_number"] == 1683
        y = [row[f"norm_{col}"] for col, _, _ in objective_cols]
        ax_parallel.plot(
            x, y,
            color="#D55E00" if is_selected else MODE_COLOURS[row["mode"]],
            alpha=0.95 if is_selected else 0.24,
            linewidth=2.2 if is_selected else 0.8,
            zorder=4 if is_selected else 1,
        )
    ax_parallel.set_xticks(x, [label for _, label, _ in objective_cols])
    ax_parallel.set_ylim(-0.03, 1.03)
    ax_parallel.set_ylabel("Normalized desirability")
    ax_parallel.set_title("Five-objective Pareto-retained model candidates", pad=22)
    ax_parallel.grid(axis="y", alpha=0.22, linewidth=0.6)
    ax_parallel.spines[["top", "right"]].set_visible(False)
    add_panel_label(ax_parallel, "a", x=-0.08, y=1.08)

    marker_map = {
        "attention": "o",
        "series": "s",
        "uncertainty_1": "^",
        "uncertainty_2": "D",
    }
    for mode in ["LLM", "tradition"]:
        subset = pareto[pareto["mode"].eq(mode)]
        for model_type, marker in marker_map.items():
            part = subset[subset["model_type"].eq(model_type)]
            if part.empty:
                continue
            ax_tradeoff.scatter(
                part["experiment_data_r2"], part["original_sto_r2"],
                s=28, marker=marker, color=MODE_COLOURS[mode],
                alpha=0.68, edgecolor="white", linewidth=0.35,
                label=f"{mode}, {FUSION_LABELS[model_type].replace(chr(10), ' ')}",
            )
    chosen = pareto.query("mode == 'LLM' and model_type == 'series' and trial_number == 1683")
    if not chosen.empty:
        row = chosen.iloc[0]
        ax_tradeoff.scatter(row["experiment_data_r2"], row["original_sto_r2"],
                            s=130, marker="*", color="#D55E00",
                            edgecolor="black", linewidth=0.7, zorder=5)
    ax_tradeoff.axhline(0, color="#999999", linewidth=0.7, linestyle=":")
    ax_tradeoff.set_xlabel("Fine-tuned local experimental $R^2$")
    ax_tradeoff.set_ylabel("Fine-tuned STO-retention $R^2$")
    ax_tradeoff.set_title("Adaptability-retention trade-off")
    ax_tradeoff.grid(alpha=0.20, linewidth=0.6)
    ax_tradeoff.spines[["top", "right"]].set_visible(False)
    add_panel_label(ax_tradeoff, "b", x=-0.22)

    mode_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=5,
               markerfacecolor=MODE_COLOURS["LLM"], markeredgecolor="none",
               label="LLM-constrained"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=5,
               markerfacecolor=MODE_COLOURS["tradition"], markeredgecolor="none",
               label="Unconstrained"),
        Line2D([0], [0], marker="*", linestyle="none", markersize=8,
               markerfacecolor="#D55E00", markeredgecolor="black",
               label="LLM/Series trial 1683"),
    ]
    fusion_handles = [
        Line2D([0], [0], marker=marker, linestyle="none", markersize=5,
               markerfacecolor="#777777", markeredgecolor="none",
               label=FUSION_LABELS[model].replace("\n", " "))
        for model, marker in marker_map.items()
    ]
    ax_tradeoff.legend(
        handles=fusion_handles + mode_handles, frameon=False, loc="upper left",
        bbox_to_anchor=(0.0, -0.26), ncol=4, columnspacing=0.7,
        handletextpad=0.30, borderaxespad=0.0,
    )

    count_table = (
        pareto.groupby(["model_type", "mode"]).size()
        .unstack(fill_value=0)
        .reindex(["attention", "series", "uncertainty_1", "uncertainty_2"])
    )
    bar_x = np.arange(len(count_table))
    width = 0.36
    ax_counts.bar(bar_x - width / 2, count_table["LLM"], width,
                  color=MODE_COLOURS["LLM"], label="LLM-constrained")
    ax_counts.bar(bar_x + width / 2, count_table["tradition"], width,
                  color=MODE_COLOURS["tradition"], label="Unconstrained")
    ax_counts.set_xticks(bar_x, [FUSION_LABELS[idx] for idx in count_table.index])
    ax_counts.tick_params(axis="x", labelrotation=12)
    ax_counts.set_ylabel("Pareto solutions")
    ax_counts.set_title("Retained solutions by fusion")
    ax_counts.grid(axis="y", alpha=0.20, linewidth=0.6)
    ax_counts.spines[["top", "right"]].set_visible(False)
    add_panel_label(ax_counts, "c", x=-0.25)

    handles = [
        plt.Line2D([0], [0], color=MODE_COLOURS["LLM"], lw=2, label="LLM-constrained"),
        plt.Line2D([0], [0], color=MODE_COLOURS["tradition"], lw=2, label="Unconstrained"),
        plt.Line2D([0], [0], color="#D55E00", lw=2.4, label="LLM/Series trial 1683"),
    ]
    ax_parallel.legend(handles=handles, frameon=False, ncol=3,
                       loc="lower right", bbox_to_anchor=(1.0, 1.01),
                       borderaxespad=0.0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
