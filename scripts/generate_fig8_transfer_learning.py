#!/usr/bin/env python3
"""Generate transfer-learning lift bar chart (Fig. 8) from fine-tune CSV.

Reads performance_comparison.csv and plots pre-fine-tune vs post-fine-tune R²
for each trial, highlighting the best-performing model.
No model loading required.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT / "LLM/XGB_BNN_series_hybrid_model"
OUT_DIR = PROJECT / "npj_ComputMater_Manuscript.assets/generated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12,
    "xtick.labelsize": 8, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.titlesize": 13, "font.family": "sans-serif",
    "mathtext.fontset": "stix",
})

# ── Load data ────────────────────────────────────────────────────────
csv_path = MODEL_DIR / "fine-tune/batch_finetuned_models/performance_comparison.csv"
df = pd.read_csv(csv_path)
df_success = df[df["status"] == "success"].copy()
df_success["delta_r2"] = df_success["experiment_data_r2"] - df_success["original_sto_r2"]

print(f"Total trials: {len(df)}, successful: {len(df_success)}")
print(f"Mean ΔR² = {df_success['delta_r2'].mean():.3f}")
print(f"Best trial: {df_success.loc[df_success['delta_r2'].idxmax(), 'solution']}, "
      f"ΔR² = {df_success['delta_r2'].max():.3f}")

# Select top 16 trials by ΔR² and sort
df_top = df_success.nlargest(16, "delta_r2").sort_values("delta_r2")

# ── Plot ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))

y_pos = np.arange(len(df_top))
bar_height = 0.35

# Pre-fine-tune (STO R²)
ax.barh(y_pos + bar_height / 2, df_top["original_sto_r2"], bar_height,
        color="#5D9CEC", alpha=0.85, label="Pre-fine-tune (STO $R^2$)")
# Post-fine-tune (Experiment R²)
ax.barh(y_pos - bar_height / 2, df_top["experiment_data_r2"], bar_height,
        color="#E66100", alpha=0.85, label="Post-fine-tune (Exp $R^2$)")

# ΔR² annotations
for i, (_, row) in enumerate(df_top.iterrows()):
    delta = row["delta_r2"]
    ax.text(max(row["original_sto_r2"], row["experiment_data_r2"]) + 0.02, i,
            f"+{delta:.2f}", va="center", fontsize=8, fontweight="bold",
            color="#2E7D32" if delta > 0.5 else "#333333")

ax.set_yticks(y_pos)
ax.set_yticklabels([s.replace("trial_", "T") for s in df_top["solution"]])
ax.set_xlabel("$R^2$", fontsize=12)
ax.set_title("Transfer-Learning Lift: Pre-fine-tune vs Post-fine-tune $R^2$\n"
             "LLM-Constrained / Series — Top 16 Trials",
             fontsize=13)
ax.legend(loc="lower right", framealpha=0.9)
ax.grid(True, alpha=0.2, axis="x")
ax.set_xlim(0, 1.15)

plt.tight_layout()
plt.savefig(str(OUT_DIR / "Fig8_transfer_learning_lift.png"), dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved {OUT_DIR / 'Fig8_transfer_learning_lift.png'}")

# ── Also: summary statistics box ─────────────────────────────────────
print(f"\nSummary (all successful trials, n={len(df_success)}):")
print(f"  Pre-fine-tune  STO R²:  mean={df_success['original_sto_r2'].mean():.3f}, "
      f"median={df_success['original_sto_r2'].median():.3f}")
print(f"  Post-fine-tune Exp R²: mean={df_success['experiment_data_r2'].mean():.3f}, "
      f"median={df_success['experiment_data_r2'].median():.3f}")
print(f"  ΔR²:                   mean={df_success['delta_r2'].mean():.3f}, "
      f"median={df_success['delta_r2'].median():.3f}")
print("Done.")
