#!/usr/bin/env python3
"""Generate parameter response surfaces (Fig. 7) from saved LLM/series model.

Loads the fine-tuned BNN, performs MC-Dropout inference over 2D parameter
grids, and produces contour plots of predicted RRR and predictive uncertainty.
"""

import sys
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, List

# ── Add project to path ──────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "LLM/XGB_BNN_series_hybrid_model"))
sys.path.insert(0, str(PROJECT / "utils"))

plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.titlesize": 13, "font.family": "sans-serif",
    "mathtext.fontset": "stix",
})

BASE_FEATURES = ["Oxygen pressure", "Laser energy density", "Temperature",
                 "Frequency", "Thickness"]


# ── Build BNN from saved artefacts ───────────────────────────────────
def load_bnn(trial_id: int) -> Tuple[nn.Module, object]:
    """Load BNN model and feature processor for the given trial."""
    MODEL_DIR = PROJECT / "LLM/XGB_BNN_series_hybrid_model"
    trial_dir = MODEL_DIR / "fine-tune/batch_finetuned_models" / f"trial_{trial_id}"

    fp = joblib.load(str(trial_dir / "feature_processor.joblib"))
    sd = torch.load(str(trial_dir / "model/bnn_model.pth"), map_location="cpu")

    # Get architecture params from pretraining results
    pretrain_csv = MODEL_DIR / "pretrain/hyperparameter_tuning_results/all_trials_results.csv"
    df_pt = pd.read_csv(pretrain_csv)
    row = df_pt[df_pt["trial_number"] == trial_id]
    if len(row) == 0:
        raise ValueError(f"Trial {trial_id} not found in pretraining results")

    r = row.iloc[0]
    base_dim = 2 ** int(r["bnn_first_hidden_dims_pow"])
    hidden_dims = [base_dim, base_dim // 2, base_dim // 4]
    dropout_rates = [
        float(r["bnn_dropout_rate_0"]),
        float(r["bnn_dropout_rate_1"]),
        float(r["bnn_dropout_rate_2"]),
    ]

    # Check: does the state dict have LayerNorm?
    has_layernorm = any("network." in k and ".weight" in k
                        and v.dim() == 1 and ".weight" in k
                        and int(k.split(".")[1]) not in [0, 4, 8, 12]
                        for k, v in sd.items())

    config = {
        "hidden_dims": hidden_dims,
        "dropout_rates": dropout_rates,
        "use_layernorm": True,
        "use_batchnorm": False,
        "activation": "silu",
    }

    print(f"  Architecture: input_dim=5, hidden={hidden_dims}, "
          f"dropout={[f'{d:.3f}' for d in dropout_rates]}, "
          f"layernorm=True, use_mask=True")

    from model import BNN
    bnn = BNN(input_dim=5, config=config, use_mask=True)
    bnn.load_state_dict(sd)
    bnn.eval()
    return bnn, fp


# ── MC-Dropout prediction ────────────────────────────────────────────
class MCDropoutPredictor:
    """Wraps BNN for MC-Dropout inference with mask handling."""

    def __init__(self, bnn: nn.Module, feature_processor):
        self.bnn = bnn
        self.fp = feature_processor

    @torch.no_grad()
    def predict(self, X_raw: pd.DataFrame, n_samples: int = 50
                ) -> Tuple[np.ndarray, np.ndarray]:
        X_proc = self.fp.transform(X_raw[BASE_FEATURES])
        mask = np.ones_like(X_proc)  # all features present (no missing)

        x_t = torch.tensor(X_proc, dtype=torch.float32)
        m_t = torch.tensor(mask, dtype=torch.float32)

        self.bnn.train()  # enable dropout
        preds = []
        for _ in range(n_samples):
            preds.append(self.bnn(x_t, m_t).cpu().numpy().ravel())
        self.bnn.eval()
        preds = np.array(preds)
        return preds.mean(axis=0), preds.std(axis=0)


# ── 2D response surface ──────────────────────────────────────────────
def generate_surface(
    pred: MCDropoutPredictor,
    x_param: str, y_param: str,
    fixed_params: Dict[str, float],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    log_x: bool = False,
    log_y: bool = False,
    resolution: int = 50,
    n_mc: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if log_x:
        xs = np.logspace(np.log10(x_range[0]), np.log10(x_range[1]), resolution)
    else:
        xs = np.linspace(x_range[0], x_range[1], resolution)
    if log_y:
        ys = np.logspace(np.log10(y_range[0]), np.log10(y_range[1]), resolution)
    else:
        ys = np.linspace(y_range[0], y_range[1], resolution)

    Xg, Yg = np.meshgrid(xs, ys)
    points = []
    for i in range(resolution):
        for j in range(resolution):
            pt = {x_param: Xg[i, j], y_param: Yg[i, j]}
            pt.update(fixed_params)
            points.append(pt)

    df = pd.DataFrame(points)
    for col in BASE_FEATURES:
        if col not in df.columns:
            df[col] = fixed_params.get(col, 0.0)

    mean, std = pred.predict(df, n_samples=n_mc)
    return Xg, Yg, mean.reshape(resolution, resolution), std.reshape(resolution, resolution)


# ── Plotting ─────────────────────────────────────────────────────────
def plot_figure(surfaces: List[Dict], save_path: str):
    n = len(surfaces)
    fig, axes = plt.subplots(n, 2, figsize=(14, 5.5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for row, s in enumerate(surfaces):
        for col, (grid, cmap, label) in enumerate([
            (s["pred"], "viridis", "Predicted RRR"),
            (s["unc"], "YlOrRd", "Predictive std"),
        ]):
            ax = axes[row, col]
            levels = np.linspace(grid.min(), grid.max(), 20)
            im = ax.contourf(s["Xg"], s["Yg"], grid, levels=levels, cmap=cmap)
            if s.get("log_x"):
                ax.set_xscale("log")
            if s.get("log_y"):
                ax.set_yscale("log")
            ax.set_xlabel(s["xlabel"], fontsize=11)
            ax.set_ylabel(s["ylabel"], fontsize=11)
            subtitle = f"{'Predicted RRR' if col == 0 else 'Uncertainty'}: {s['title']}"
            ax.set_title(subtitle, fontsize=11)
            plt.colorbar(im, ax=ax)

    fig.suptitle("Parameter Response Surfaces — LLM-Constrained / Series\n"
                 "$f$ = 4 Hz, $d$ = 30 nm",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    OUT_DIR = PROJECT / "npj_ComputMater_Manuscript.assets/generated"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    TRIAL = 1670  # best all-around
    print(f"Loading trial {TRIAL}...")
    bnn, fp = load_bnn(TRIAL)
    predictor = MCDropoutPredictor(bnn, fp)

    fixed = {"Frequency": 4.0, "Thickness": 30.0}
    surfaces_defs = [
        {"x_param": "Oxygen pressure", "y_param": "Temperature",
         "xlabel": "$P_{\\mathrm{O_2}}$ (mbar)", "ylabel": "$T$ (°C)",
         "title": "$P_{\\mathrm{O_2}}$ vs $T$",
         "log_x": True, "log_y": False,
         "x_range": (1e-4, 1.0), "y_range": (500, 900)},
        {"x_param": "Oxygen pressure", "y_param": "Laser energy density",
         "xlabel": "$P_{\\mathrm{O_2}}$ (mbar)", "ylabel": "$F$ (J/cm²)",
         "title": "$P_{\\mathrm{O_2}}$ vs $F$",
         "log_x": True, "log_y": False,
         "x_range": (1e-4, 1.0), "y_range": (1.0, 3.0)},
    ]

    print("Generating response surfaces (MC-Dropout, 50 samples × 2500 grid points)...")
    all_surfaces = []
    for s in surfaces_defs:
        Xg, Yg, pred, unc = generate_surface(
            predictor, s["x_param"], s["y_param"], fixed,
            x_range=s["x_range"], y_range=s["y_range"],
            log_x=s["log_x"], log_y=s["log_y"],
            resolution=50, n_mc=50,
        )
        all_surfaces.append({**s, "Xg": Xg, "Yg": Yg, "pred": pred, "unc": unc})
        print(f"  {s['title']}: RRR ∈ [{pred.min():.2f}, {pred.max():.2f}], "
              f"std ∈ [{unc.min():.3f}, {unc.max():.3f}]")

    plot_figure(all_surfaces, str(OUT_DIR / "Fig7_response_surfaces.png"))
    print("Done.")
