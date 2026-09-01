#!/usr/bin/env python3
"""Redraw final manuscript-style figures for trial 1683."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import QuadMesh
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
COLLECTED = ROOT / "paper_figures" / "trial_1683_series_collected_assets"
OUT = COLLECTED / "final_paper_figures"

STUDENT_MODEL_DIR = COLLECTED / "pareto_trial_1683" / "model"
TEACHER_MODEL_DIR = ROOT / "LLM" / "XGB_BNN_series_hybrid_model" / "pretrain" / "hyperparameter_tuning_results" / "all_trials" / "trial_1683" / "model"
BO_RESULTS = COLLECTED / "bo_outputs_trial_1683_fix_freq_4.0" / "optimization_results.csv"
FIXED_RESPONSE_GRID = COLLECTED / "fixed_20nm_4hz_response" / "trial1683_fixed_20nm_4hz_response_grid.csv"
FT_RESULTS = COLLECTED / "fine_tune_trial_1683" / "finetune_results.joblib"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "utils"))
sys.path.insert(0, str(ROOT / "LLM" / "XGB_BNN_series_hybrid_model"))
from model import HybridModel  # noqa: E402


BASE_FEATURES = [
    "Oxygen pressure",
    "Laser energy density",
    "Temperature",
    "Frequency",
    "Thickness",
]


def setup_style() -> None:
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
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


def save_all(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{suffix}", bbox_inches="tight")


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.15,
        1.12,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="top",
    )


def square_panel(ax: plt.Axes) -> None:
    ax.set_box_aspect(1)


def parity_axes(ax: plt.Axes) -> None:
    ax.set_aspect("equal", adjustable="box")


def load_experiment_data() -> tuple[pd.DataFrame, np.ndarray]:
    raw = pd.read_csv(ROOT / "data" / "sobol_samples_results.csv")
    x = pd.DataFrame(
        {
            "Oxygen pressure": raw["oxygen_pressure"],
            "Laser energy density": raw["laser_energy_density"],
            "Temperature": raw["temperature"],
            "Frequency": raw["frequency"],
            "Thickness": raw["thickness"],
        }
    )
    y = raw["RRR"].to_numpy(dtype=float)
    return x, y


def plot_param_space_final() -> None:
    df = pd.read_csv(BO_RESULTS).copy()
    df["log10_pO2"] = np.log10(df["oxygen_pressure"])
    best = df.loc[df["mean"].idxmax()]

    pairs = [
        ("log10_pO2", "laser_energy_density", r"$\log_{10}(p_{\mathrm{O_2}}/\mathrm{mbar})$", r"Laser energy density (J/cm$^2$)"),
        ("log10_pO2", "temperature", r"$\log_{10}(p_{\mathrm{O_2}}/\mathrm{mbar})$", r"Temperature ($^\circ$C)"),
        ("laser_energy_density", "temperature", r"Laser energy density (J/cm$^2$)", r"Temperature ($^\circ$C)"),
    ]

    norm = Normalize(vmin=df["mean"].quantile(0.02), vmax=df["mean"].quantile(0.995))
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 2.25),
        constrained_layout=True,
        subplot_kw={"box_aspect": 1},
    )
    fig.set_constrained_layout_pads(w_pad=0.12, h_pad=0.04, wspace=0.16)

    for ax, (xcol, ycol, xlabel, ylabel) in zip(axes, pairs):
        sc = ax.scatter(
            df[xcol],
            df[ycol],
            c=df["mean"],
            cmap="viridis",
            norm=norm,
            s=12,
            alpha=0.7,
            linewidths=0,
            rasterized=True,
        )
        best_x = np.log10(best["oxygen_pressure"]) if xcol == "log10_pO2" else best[xcol]
        best_y = np.log10(best["oxygen_pressure"]) if ycol == "log10_pO2" else best[ycol]
        ax.scatter(
            best_x,
            best_y,
            marker="*",
            s=105,
            c="#d7191c",
            edgecolor="black",
            linewidth=0.45,
            zorder=5,
        )
        ax.text(best_x, best_y, "Best  ", va="center", ha="right", fontsize=7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.22, linewidth=0.5)
        square_panel(ax)

    cax = axes[2].inset_axes([1.12, 0.14, 0.035, 0.72])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("Predicted RRR")
    for label, ax in zip("abc", axes):
        panel_label(ax, label)
    save_all(fig, "param_space_final")
    plt.close(fig)


def plot_prediction_comparison_final() -> None:
    x_exp, y_exp = load_experiment_data()
    student = HybridModel.load_model(str(STUDENT_MODEL_DIR))
    teacher = HybridModel.load_model(str(TEACHER_MODEL_DIR))
    eval_results = joblib.load(FT_RESULTS)["evaluation_results"]

    pred_student, std_student = student.predict(x_exp, n_samples=80)
    pred_teacher, _ = teacher.predict(x_exp, n_samples=80)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.04, hspace=0.03, wspace=0.08)
    ax = axes[0]

    min_v = min(y_exp.min(), pred_student.min(), pred_teacher.min()) - 0.25
    max_v = max(y_exp.max(), pred_student.max(), pred_teacher.max()) + 0.35
    ax.plot([min_v, max_v], [min_v, max_v], color="#455a64", linestyle="--", linewidth=1.0, zorder=1)
    ax.errorbar(
        y_exp,
        pred_student,
        yerr=std_student,
        fmt="o",
        color="#d73027",
        ecolor="#ef9a9a",
        elinewidth=0.7,
        capsize=1.5,
        markersize=4.2,
        markeredgecolor="black",
        markeredgewidth=0.35,
        label="Fine-tuned student",
        zorder=3,
    )
    ax.scatter(
        y_exp,
        pred_teacher,
        marker="^",
        s=27,
        color="#90a4ae",
        alpha=0.75,
        edgecolor="white",
        linewidth=0.25,
        label="Pretrained teacher",
        zorder=2,
    )
    ax.set_xlim(min_v, max_v)
    ax.set_ylim(min_v, max_v)
    parity_axes(ax)
    ax.set_xlabel("Measured RRR")
    ax.set_ylabel("Predicted RRR")
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        handlelength=1.2,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    ax.text(
        0.04,
        0.05,
        f"student $R^2$ = {eval_results['experiment_data_r2']:.3f}\n"
        f"teacher $R^2$ = {eval_results['teacher_experiment_data_r2']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cfd8dc", alpha=0.95),
    )
    ax.grid(True, alpha=0.22, linewidth=0.5)

    ax = axes[1]
    metrics = ["$R^2$", "MAE", "RMSE"]
    student_vals = [
        eval_results["experiment_data_r2"],
        eval_results["experiment_data_mae"],
        eval_results["experiment_data_rmse"],
    ]
    teacher_vals = [
        eval_results["teacher_experiment_data_r2"],
        eval_results["teacher_experiment_data_mae"],
        eval_results["teacher_experiment_data_rmse"],
    ]

    x = np.arange(len(metrics))
    width = 0.34
    ax.axhline(0, color="#455a64", linewidth=0.8)
    bars_t = ax.bar(x - width / 2, teacher_vals, width, label="Teacher", color="#b0bec5")
    bars_s = ax.bar(x + width / 2, student_vals, width, label="Student", color="#d73027")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Metric value")
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        handlelength=1.2,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    ax.grid(axis="y", alpha=0.22, linewidth=0.5)
    ymin = min(teacher_vals + student_vals) - 0.25
    ymax = max(teacher_vals + student_vals) + 0.35
    ax.set_ylim(ymin, ymax)
    for bars in (bars_t, bars_s):
        for bar in bars:
            val = bar.get_height()
            va = "bottom" if val >= 0 else "top"
            offset = 0.04 if val >= 0 else -0.04
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + offset,
                f"{val:.2f}",
                ha="center",
                va=va,
                fontsize=6.5,
            )

    for label, ax in zip("ab", axes):
        panel_label(ax, label)
    save_all(fig, "prediction_comparison_combined_final")
    plt.close(fig)


def _fixed_slice(grid: pd.DataFrame, x_feature: str, y_feature: str) -> pd.DataFrame:
    return grid[(grid["slice_x"] == x_feature) & (grid["slice_y"] == y_feature)].copy()


def _draw_heatmap(
    ax: plt.Axes,
    df: pd.DataFrame,
    value_col: str,
    x_feature: str,
    y_feature: str,
    title: str,
    cmap: str,
    norm: Normalize,
) -> QuadMesh:
    x = np.sort(df[x_feature].unique())
    y = np.sort(df[y_feature].unique())
    z = df[value_col].to_numpy().reshape(len(y), len(x))
    mesh = ax.pcolormesh(x, y, z, shading="auto", cmap=cmap, norm=norm)
    ax.contour(x, y, z, colors="white", linewidths=0.35, alpha=0.65)
    if x_feature == "Oxygen pressure":
        ax.set_xscale("log")
    if y_feature == "Oxygen pressure":
        ax.set_yscale("log")
    labels = {
        "Oxygen pressure": r"$p_{\mathrm{O_2}}$ (mbar)",
        "Laser energy density": r"Laser energy density (J/cm$^2$)",
        "Temperature": r"Temperature ($^\circ$C)",
    }
    ax.set_xlabel(labels[x_feature])
    ax.set_ylabel(labels[y_feature])
    if title:
        ax.set_title(title, pad=7)
    square_panel(ax)
    return mesh


def plot_fixed_response_final() -> None:
    grid = pd.read_csv(FIXED_RESPONSE_GRID)
    pairs = [
        ("Oxygen pressure", "Temperature", r"Laser energy density = 2.90 J/cm$^2$"),
        ("Laser energy density", "Temperature", r"$p_{\mathrm{O_2}}$ = 0.50 mbar"),
        ("Oxygen pressure", "Laser energy density", r"Temperature = 600 $^\circ$C"),
    ]

    pred_norm = Normalize(vmin=grid["predicted_rrr"].min(), vmax=grid["predicted_rrr"].max())
    unc_norm = Normalize(vmin=grid["uncertainty"].min(), vmax=grid["uncertainty"].max())

    fig = plt.figure(figsize=(7.45, 4.55), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.055], wspace=0.04, hspace=0.08)
    axes = np.array([[fig.add_subplot(gs[row, col]) for col in range(3)] for row in range(2)])
    cax_top = fig.add_subplot(gs[0, 3])
    cax_bottom = fig.add_subplot(gs[1, 3])
    top_mesh = None
    bottom_mesh = None
    for col, (x_feature, y_feature, condition) in enumerate(pairs):
        df = _fixed_slice(grid, x_feature, y_feature)
        top_mesh = _draw_heatmap(
            axes[0, col],
            df,
            "predicted_rrr",
            x_feature,
            y_feature,
            condition,
            "viridis",
            pred_norm,
        )
        bottom_mesh = _draw_heatmap(
            axes[1, col],
            df,
            "uncertainty",
            x_feature,
            y_feature,
            "",
            "magma_r",
            unc_norm,
        )

    for label, ax in zip("abcdef", axes.ravel()):
        ax.text(
            0.03,
            0.97,
            label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.2),
            zorder=10,
        )
    cbar_top = fig.colorbar(top_mesh, cax=cax_top)
    cbar_top.set_label("Predicted RRR")
    cbar_bottom = fig.colorbar(bottom_mesh, cax=cax_bottom)
    cbar_bottom.set_label("Uncertainty")
    save_all(fig, "trial1683_fixed_20nm_4hz_response_final")
    plt.close(fig)


def main() -> None:
    setup_style()
    plot_param_space_final()
    plot_prediction_comparison_final()
    plot_fixed_response_final()
    print(f"Wrote final figures to {OUT}")


if __name__ == "__main__":
    main()
