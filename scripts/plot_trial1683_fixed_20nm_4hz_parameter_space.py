#!/usr/bin/env python3
"""Plot trial 1683 fine-tuned model response at fixed thickness and frequency."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "paper_figures" / "trial_1683_series_collected_assets" / "pareto_trial_1683" / "model"
OUT_DIR = ROOT / "paper_figures" / "trial_1683_series_collected_assets" / "fixed_20nm_4hz_response"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "utils"))
sys.path.insert(0, str(ROOT / "LLM" / "XGB_BNN_series_hybrid_model"))
from model import HybridModel  # noqa: E402


FEATURES = [
    "Oxygen pressure",
    "Laser energy density",
    "Temperature",
    "Frequency",
    "Thickness",
]

FIXED = {
    "Frequency": 4.0,
    "Thickness": 20.0,
}

# The plotting window follows the fixed-frequency BO search domain used in trial 1683.
RANGES = {
    "Oxygen pressure": (1e-4, 0.5),
    "Laser energy density": (1.5, 3.0),
    "Temperature": (500.0, 800.0),
}

# Slice centers define the three 2D projections, with thickness/frequency overwritten above.
SLICE_CENTER = {
    "Oxygen pressure": 0.5,
    "Laser energy density": 2.9,
    "Temperature": 600.0,
}

LABELS = {
    "Oxygen pressure": r"$p_{\mathrm{O_2}}$ (mbar)",
    "Laser energy density": r"Fluence (J cm$^{-2}$)",
    "Temperature": r"Temperature ($^\circ$C)",
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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def make_grid(x_feature: str, y_feature: str, resolution: int = 90) -> pd.DataFrame:
    if x_feature == "Oxygen pressure":
        x = np.logspace(np.log10(RANGES[x_feature][0]), np.log10(RANGES[x_feature][1]), resolution)
    else:
        x = np.linspace(*RANGES[x_feature], resolution)

    if y_feature == "Oxygen pressure":
        y = np.logspace(np.log10(RANGES[y_feature][0]), np.log10(RANGES[y_feature][1]), resolution)
    else:
        y = np.linspace(*RANGES[y_feature], resolution)

    xx, yy = np.meshgrid(x, y)
    data = {
        "Oxygen pressure": np.full(xx.size, SLICE_CENTER["Oxygen pressure"]),
        "Laser energy density": np.full(xx.size, SLICE_CENTER["Laser energy density"]),
        "Temperature": np.full(xx.size, SLICE_CENTER["Temperature"]),
        "Frequency": np.full(xx.size, FIXED["Frequency"]),
        "Thickness": np.full(xx.size, FIXED["Thickness"]),
    }
    data[x_feature] = xx.ravel()
    data[y_feature] = yy.ravel()
    df = pd.DataFrame(data, columns=FEATURES)
    return df


def predict_in_chunks(model: HybridModel, df: pd.DataFrame, chunk_size: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    means = []
    stds = []
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start : start + chunk_size]
        mean, std = model.predict(chunk, n_samples=50)
        means.append(mean)
        stds.append(std)
    return np.concatenate(means), np.concatenate(stds)


def save_outputs(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"{stem}.{suffix}", bbox_inches="tight")


def square_panel(ax: plt.Axes) -> None:
    ax.set_box_aspect(1)


def draw_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    value: np.ndarray,
    x_feature: str,
    y_feature: str,
    title: str,
    cmap: str,
    cbar_label: str,
) -> None:
    x_unique = np.sort(df[x_feature].unique())
    y_unique = np.sort(df[y_feature].unique())
    zz = value.reshape(len(y_unique), len(x_unique))
    mesh = ax.pcolormesh(x_unique, y_unique, zz, shading="auto", cmap=cmap)
    ax.contour(x_unique, y_unique, zz, colors="white", linewidths=0.45, alpha=0.7)
    if x_feature == "Oxygen pressure":
        ax.set_xscale("log")
    if y_feature == "Oxygen pressure":
        ax.set_yscale("log")
    ax.set_xlabel(LABELS[x_feature])
    ax.set_ylabel(LABELS[y_feature])
    ax.set_title(title)
    square_panel(ax)
    cbar = plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(cbar_label)


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = HybridModel.load_model(str(MODEL_DIR))

    pairs = [
        ("Oxygen pressure", "Temperature", "Fluence fixed at 2.90 J cm$^{-2}$"),
        ("Laser energy density", "Temperature", r"$p_{\mathrm{O_2}}$ fixed at 0.50 mbar"),
        ("Oxygen pressure", "Laser energy density", r"Temperature fixed at 600 $^\circ$C"),
    ]

    records = []
    predictions: list[tuple[pd.DataFrame, np.ndarray, np.ndarray]] = []
    for x_feature, y_feature, _ in pairs:
        grid = make_grid(x_feature, y_feature)
        mean, std = predict_in_chunks(model, grid)
        grid_out = grid.copy()
        grid_out["predicted_rrr"] = mean
        grid_out["uncertainty"] = std
        grid_out["slice_x"] = x_feature
        grid_out["slice_y"] = y_feature
        records.append(grid_out)
        predictions.append((grid, mean, std))

    pd.concat(records, ignore_index=True).to_csv(OUT_DIR / "trial1683_fixed_20nm_4hz_response_grid.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.8), constrained_layout=True)
    for col, ((x_feature, y_feature, slice_title), (grid, mean, std)) in enumerate(zip(pairs, predictions)):
        draw_panel(
            axes[0, col],
            grid,
            mean,
            x_feature,
            y_feature,
            slice_title,
            "viridis",
            "Predicted RRR",
        )
        draw_panel(
            axes[1, col],
            grid,
            std,
            x_feature,
            y_feature,
            "",
            "magma_r",
            "Uncertainty",
        )
        axes[0, col].text(
            -0.16,
            1.08,
            chr(ord("a") + col),
            transform=axes[0, col].transAxes,
            fontweight="bold",
            fontsize=11,
            va="top",
        )
        axes[1, col].text(
            -0.16,
            1.08,
            chr(ord("d") + col),
            transform=axes[1, col].transAxes,
            fontweight="bold",
            fontsize=11,
            va="top",
        )

    fig.suptitle(
        "Trial 1683 fine-tuned response at fixed thickness = 20 nm and frequency = 4 Hz",
        fontsize=10,
        fontweight="bold",
    )
    save_outputs(fig, "trial1683_fixed_20nm_4hz_response")
    plt.close(fig)

    # Also save a compact two-panel view for the most interpretable pO2-temperature slice.
    grid, mean, std = predictions[0]
    fig2, axes2 = plt.subplots(1, 2, figsize=(5.6, 2.5), constrained_layout=True)
    draw_panel(
        axes2[0],
        grid,
        mean,
        "Oxygen pressure",
        "Temperature",
        "Target value",
        "viridis",
        "Predicted RRR",
    )
    draw_panel(
        axes2[1],
        grid,
        std,
        "Oxygen pressure",
        "Temperature",
        "Predictive uncertainty",
        "magma_r",
        "Uncertainty",
    )
    for label, ax in zip("ab", axes2):
        ax.text(-0.16, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=11, va="top")
    fig2.suptitle(r"Trial 1683 response, $d=20$ nm, $f=4$ Hz, fluence = 2.90 J cm$^{-2}$", fontsize=9)
    save_outputs(fig2, "trial1683_fixed_20nm_4hz_po2_temperature")
    plt.close(fig2)

    print(f"Wrote fixed-condition response figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
