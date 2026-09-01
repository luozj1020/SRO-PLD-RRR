#!/usr/bin/env python3
"""Generate manuscript-ready figures for LLM series trial 1683."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "LLM" / "XGB_BNN_series_hybrid_model"
PRETRAIN = BASE / "pretrain" / "hyperparameter_tuning_results"
TRIAL_PRE = PRETRAIN / "all_trials" / "trial_1683"
PARETO = BASE / "pareto_solution"
TRIAL_FT = PARETO / "trial_1683"
BO_DIR = PARETO / "model_optim_prediction_results_trial_1683_fix_freq_4.0"
GRID_PATH = PARETO / "grid_search_samples" / "trial_1683_grid_samples.csv"
SCORE_PATH = ROOT / "model_comprehensive_scores_LLM.csv"
OUT = ROOT / "paper_figures" / "trial_1683_series"


PARAM_LABELS = {
    "oxygen_pressure": r"$p_{\mathrm{O_2}}$ (Torr)",
    "laser_energy_density": r"Fluence (J cm$^{-2}$)",
    "temperature": r"Temperature ($^\circ$C)",
    "frequency": "Frequency (Hz)",
    "thickness": "Thickness (nm)",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )


def square_panel(ax: plt.Axes) -> None:
    ax.set_box_aspect(1)


def parity_axes(ax: plt.Axes) -> None:
    ax.set_aspect("equal", adjustable="box")


def save_all(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{suffix}", bbox_inches="tight")


def load_data() -> dict[str, object]:
    all_trials = pd.read_csv(PRETRAIN / "all_trials_results.csv")
    pre_pred = pd.read_csv(TRIAL_PRE / "validation_predictions.csv")
    pareto = pd.read_csv(PARETO / "pareto_frontier.csv")
    combined = pd.read_csv(PARETO / "combined_metrics.csv")
    bo = pd.read_csv(BO_DIR / "optimization_results.csv")
    grid = pd.read_csv(GRID_PATH)
    scores = pd.read_csv(SCORE_PATH)
    perf = joblib.load(TRIAL_PRE / "performance_data.joblib")
    ft = joblib.load(TRIAL_FT / "finetune_results.joblib")
    history = joblib.load(TRIAL_FT / "model" / "finetune_history.joblib")
    score_row = scores[
        (scores["mode"] == "LLM")
        & (scores["model_type"] == "series")
        & (scores["trial_number"] == 1683)
    ].iloc[0]
    return {
        "all_trials": all_trials,
        "pre_pred": pre_pred,
        "pareto": pareto,
        "combined": combined,
        "bo": bo,
        "grid": grid,
        "perf": perf,
        "ft": ft,
        "history": history,
        "score_row": score_row,
    }


def draw_workflow(ax: plt.Axes, data: dict[str, object]) -> None:
    ax.set_axis_off()
    perf = data["perf"]
    ft = data["ft"]["evaluation_results"]
    score = data["score_row"]
    bo_best = data["bo"].sort_values("mean", ascending=False).iloc[0]

    boxes = [
        (
            "Pretraining",
            "322 records\n"
            f"STO $R^2$ = {perf['objectives']['sto_r2']:.3f}\n"
            f"stability = {perf['objectives']['stability_score']:.3f}",
        ),
        (
            "LLM-series selection",
            "XGB + BNN residual\n"
            f"physics = {score['score_B']:.3f}\n"
            f"final score = {score['final_score']:.3f}",
        ),
        (
            "Transfer tuning",
            "local PLD data\n"
            f"student $R^2$ = {ft['experiment_data_r2']:.3f}\n"
            f"RMSE = {ft['experiment_data_rmse']:.3f}",
        ),
        (
            "Bayesian opt.",
            "$f=4$ Hz search\n"
            f"best RRR = {bo_best['mean']:.2f}\n"
            f"$\\sigma$ = {bo_best['std']:.2f}",
        ),
    ]

    xs = [0.02, 0.27, 0.52, 0.77]
    for i, (x, (title, body)) in enumerate(zip(xs, boxes)):
        patch = FancyBboxPatch(
            (x, 0.28),
            0.205,
            0.46,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=0.9,
            edgecolor="#263238",
            facecolor=["#eef4f7", "#f4f0e8", "#edf4ee", "#f3eef5"][i],
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(
            x + 0.1025,
            0.66,
            title,
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=8,
        )
        ax.text(
            x + 0.1025,
            0.47,
            body,
            ha="center",
            va="center",
            linespacing=1.35,
            fontsize=7.2,
        )
        if i < len(xs) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + 0.215, 0.51),
                    (xs[i + 1] - 0.01, 0.51),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    linewidth=1.0,
                    color="#546e7a",
                    transform=ax.transAxes,
                )
            )


def plot_pretrain_selection(ax: plt.Axes, data: dict[str, object]) -> None:
    trials = data["all_trials"]
    pareto = data["pareto"]
    trial = trials[trials["trial_number"] == 1683].iloc[0]
    sc = ax.scatter(
        trials["sto_r2_trial"],
        trials["secondary_score"],
        c=trials["stability_score"],
        cmap="viridis",
        s=11,
        alpha=0.45,
        linewidths=0,
        rasterized=True,
    )
    ax.scatter(
        pareto["sto_r2_trial"],
        pareto["secondary_score"],
        s=28,
        facecolors="none",
        edgecolors="#37474f",
        linewidths=0.8,
        label="Pareto-retained",
    )
    ax.scatter(
        [trial["sto_r2_trial"]],
        [trial["secondary_score"]],
        marker="*",
        s=130,
        color="#d84315",
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
        label="Trial 1683",
    )
    ax.set_xlabel("Pretraining STO $R^2$")
    ax.set_ylabel("Secondary objective")
    ax.set_title("Pretraining model selection")
    square_panel(ax)
    ax.legend(loc="upper left", frameon=False)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Stability")


def plot_pretrain_parity(ax: plt.Axes, data: dict[str, object]) -> None:
    pred = data["pre_pred"].dropna(subset=["true_rrr", "predicted_rrr"])
    perf = data["perf"]["validation_performance"]
    sc = ax.scatter(
        pred["true_rrr"],
        pred["predicted_rrr"],
        c=pred["uncertainty"],
        cmap="magma_r",
        s=18,
        alpha=0.85,
        linewidths=0.2,
        edgecolors="white",
    )
    lo = min(pred["true_rrr"].min(), pred["predicted_rrr"].min())
    hi = max(pred["true_rrr"].max(), pred["predicted_rrr"].max())
    ax.plot([lo, hi], [lo, hi], color="#455a64", linewidth=1.0, linestyle="--")
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(lo - 0.5, hi + 0.5)
    parity_axes(ax)
    ax.set_xlabel("Measured RRR")
    ax.set_ylabel("Predicted RRR")
    ax.set_title("Pretraining validation")
    ax.text(
        0.04,
        0.96,
        f"overall $R^2$ = {perf['r2_score']:.3f}\nSTO $R^2$ = {perf['sto_r2']:.3f}\nRMSE = {perf['rmse']:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cfd8dc", alpha=0.92),
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Predictive uncertainty")


def plot_finetune_history(ax: plt.Axes, data: dict[str, object]) -> None:
    hist = data["history"]
    epochs = np.arange(1, len(hist["loss"]) + 1)
    ax.plot(epochs, hist["loss"], color="#1565c0", linewidth=1.0, label="total loss")
    ax.plot(epochs, hist["hard_loss"], color="#ef6c00", linewidth=0.9, alpha=0.85, label="hard loss")
    ax.set_xlabel("Fine-tuning epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Transfer fine-tuning")
    ax2 = ax.twinx()
    ax2.plot(epochs, hist["r2"], color="#2e7d32", linewidth=1.0, label="$R^2$")
    ax2.set_ylabel("$R^2$")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="center right", frameon=False)


def plot_score_breakdown(ax: plt.Axes, data: dict[str, object]) -> None:
    score = data["score_row"]
    labels = [
        "Pretrain",
        "Transfer",
        "LLM raw",
        "LLM corr.",
        "Final",
    ]
    values = [
        score["score_A"],
        score["score_C"],
        score["score_B"],
        score["score_B_corrected_rationality"],
        score["final_score"],
    ]
    colors = ["#607d8b", "#2e7d32", "#8d6e63", "#6d4c41", "#d84315"]
    ax.bar(labels, values, color=colors, width=0.68)
    ax.axhline(score["bayesian_edge_penalty_D"], color="#78909c", linestyle=":", linewidth=1.0)
    ax.text(
        0.02,
        score["bayesian_edge_penalty_D"] + 0.015,
        "BO edge penalty",
        color="#546e7a",
        transform=ax.get_yaxis_transform(),
        va="bottom",
    )
    ax.set_ylim(0, max(values) * 1.25)
    ax.set_ylabel("Normalized score")
    ax.set_title("Comprehensive scoring")
    ax.tick_params(axis="x", labelrotation=25)
    for i, v in enumerate(values):
        ax.text(i, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=7)


def plot_bo_progress(ax: plt.Axes, data: dict[str, object]) -> None:
    bo = data["bo"].copy()
    bo["best_so_far"] = bo["mean"].cummax()
    idx = np.arange(1, len(bo) + 1)
    ax.plot(idx, bo["mean"], color="#90a4ae", linewidth=0.45, alpha=0.45, label="candidate prediction")
    ax.plot(idx, bo["best_so_far"], color="#c62828", linewidth=1.4, label="best-so-far")
    best_idx = int(bo["mean"].idxmax()) + 1
    best = bo.loc[bo["mean"].idxmax()]
    ax.scatter([best_idx], [best["mean"]], marker="*", s=100, color="#d84315", edgecolor="white", zorder=5)
    ax.set_xlabel("BO candidate index")
    ax.set_ylabel("Predicted RRR")
    ax.set_title("Bayesian optimization trajectory")
    ax.legend(loc="lower right", frameon=False)
    ax.text(
        0.04,
        0.95,
        f"best = {best['mean']:.2f} ± {best['std']:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cfd8dc", alpha=0.92),
    )


def plot_bo_landscape(ax: plt.Axes, data: dict[str, object]) -> None:
    bo = data["bo"]
    best = bo.loc[bo["mean"].idxmax()]
    sc = ax.scatter(
        bo["temperature"],
        bo["oxygen_pressure"],
        c=bo["mean"],
        s=18 + 40 * (bo["thickness"].rank(pct=True).to_numpy()),
        cmap="plasma",
        alpha=0.75,
        linewidths=0,
        rasterized=True,
    )
    ax.set_yscale("log")
    ax.scatter(
        [best["temperature"]],
        [best["oxygen_pressure"]],
        marker="*",
        s=140,
        color="#00acc1",
        edgecolor="black",
        linewidth=0.5,
        zorder=5,
    )
    ax.set_xlabel(r"Temperature ($^\circ$C)")
    ax.set_ylabel(r"$p_{\mathrm{O_2}}$ (Torr)")
    ax.set_title("BO search landscape")
    square_panel(ax)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Predicted RRR")


def plot_parameter_recommendation(ax: plt.Axes, data: dict[str, object]) -> None:
    bo = data["bo"]
    best = bo.loc[bo["mean"].idxmax()]
    params = ["oxygen_pressure", "laser_energy_density", "temperature", "frequency", "thickness"]
    mins = bo[params].min()
    maxs = bo[params].max()
    vals = (best[params] - mins) / (maxs - mins).replace(0, np.nan)
    vals = vals.fillna(0.5)
    y = np.arange(len(params))
    ax.barh(y, vals, color="#455a64")
    ax.set_yticks(y)
    ax.set_yticklabels([PARAM_LABELS[p] for p in params])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Position within searched range")
    ax.set_title("Best BO recommendation")
    ax.invert_yaxis()
    for yi, p in enumerate(params):
        ax.text(
            min(vals[p] + 0.03, 0.98),
            yi,
            f"{best[p]:.4g}",
            va="center",
            ha="left" if vals[p] < 0.9 else "right",
            color="#263238",
            fontsize=7,
        )


def make_overview(data: dict[str, object]) -> None:
    fig = plt.figure(figsize=(8.2, 8.8), constrained_layout=True)
    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[0.8, 1, 1, 1])
    ax_flow = fig.add_subplot(gs[0, :])
    draw_workflow(ax_flow, data)
    panel_label(ax_flow, "a")
    axes = [
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
        fig.add_subplot(gs[3, 0]),
        fig.add_subplot(gs[3, 1]),
    ]
    plot_pretrain_selection(axes[0], data)
    plot_pretrain_parity(axes[1], data)
    plot_finetune_history(axes[2], data)
    plot_score_breakdown(axes[3], data)
    plot_bo_progress(axes[4], data)
    plot_parameter_recommendation(axes[5], data)
    for label, ax in zip("bcdefg", axes):
        panel_label(ax, label)
    save_all(fig, "fig_trial1683_series_full_chain")
    plt.close(fig)


def make_bo_figure(data: dict[str, object]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), constrained_layout=True)
    plot_bo_progress(axes[0], data)
    plot_bo_landscape(axes[1], data)
    plot_parameter_recommendation(axes[2], data)
    for label, ax in zip("abc", axes):
        panel_label(ax, label)
    save_all(fig, "fig_trial1683_series_bo_detail")
    plt.close(fig)


def make_pretrain_transfer_figure(data: dict[str, object]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), constrained_layout=True)
    plot_pretrain_selection(axes[0], data)
    plot_pretrain_parity(axes[1], data)
    plot_finetune_history(axes[2], data)
    for label, ax in zip("abc", axes):
        panel_label(ax, label)
    save_all(fig, "fig_trial1683_series_pretrain_transfer")
    plt.close(fig)


def write_summary(data: dict[str, object]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    perf = data["perf"]
    ft = data["ft"]["evaluation_results"]
    score = data["score_row"]
    best = data["bo"].sort_values("mean", ascending=False).iloc[0]
    grid_best = data["grid"].sort_values("mean", ascending=False).iloc[0]
    lines = [
        "# Trial 1683 LLM-series figure summary",
        "",
        "## Pretraining",
        f"- Validation overall R2: {perf['validation_performance']['r2_score']:.6f}",
        f"- Validation STO R2: {perf['validation_performance']['sto_r2']:.6f}",
        f"- Validation RMSE: {perf['validation_performance']['rmse']:.6f}",
        f"- Stability score: {perf['objectives']['stability_score']:.6f}",
        "",
        "## Transfer fine-tuning",
        f"- Local experiment student R2: {ft['experiment_data_r2']:.6f}",
        f"- Local experiment RMSE: {ft['experiment_data_rmse']:.6f}",
        f"- Teacher baseline R2 on local data: {ft['teacher_experiment_data_r2']:.6f}",
        "",
        "## LLM/comprehensive scoring",
        f"- Parameter rationality: {score['parameter_rationality']:.6f}",
        f"- Parameter synergy: {score['parameter_synergy']:.6f}",
        f"- Final score: {score['final_score']:.6f}",
        "",
        "## Bayesian optimization best candidate",
        f"- pO2: {best['oxygen_pressure']:.8g} Torr",
        f"- Fluence: {best['laser_energy_density']:.8g} J cm^-2",
        f"- Temperature: {best['temperature']:.8g} C",
        f"- Frequency: {best['frequency']:.8g} Hz",
        f"- Thickness: {best['thickness']:.8g} nm",
        f"- Predicted RRR: {best['mean']:.6f} +/- {best['std']:.6f}",
        "",
        "## Grid-search best candidate",
        f"- pO2: {grid_best['oxygen_pressure']:.8g} Torr",
        f"- Fluence: {grid_best['laser_energy_density']:.8g} J cm^-2",
        f"- Temperature: {grid_best['temperature']:.8g} C",
        f"- Frequency: {grid_best['frequency']:.8g} Hz",
        f"- Thickness: {grid_best['thickness']:.8g} nm",
        f"- Predicted RRR: {grid_best['mean']:.6f} +/- {grid_best['std']:.6f}",
        "",
    ]
    (OUT / "trial1683_series_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup_style()
    data = load_data()
    make_overview(data)
    make_pretrain_transfer_figure(data)
    make_bo_figure(data)
    write_summary(data)
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
