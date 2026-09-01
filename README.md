# SRO-PLD-RRR: Language-model priors for sparse adaptation of experimental process models

<p align="center">
  <a href="README_zh.md">中文文档</a> | <strong>English</strong>
</p>

> A physics-constrained hybrid machine learning framework integrating LLM-encoded domain knowledge, transfer learning, and adaptive Bayesian optimization for SrRuO₃ thin film growth optimization via Pulsed Laser Deposition (PLD).

---

## Table of Contents

- [Background](#background)
- [Key Features](#key-features)
- [Workflow Overview](#workflow-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Algorithm Details](#algorithm-details)
- [Data Availability](#data-availability)
- [Release Scope](#release-scope)
- [LLM Model Availability](#llm-model-availability)
- [Citation](#citation)
- [License](#license)

---

## Background

SrRuO₃ (SRO) thin films grown by PLD exhibit exceptional sensitivity to process parameters — small variations in oxygen partial pressure, substrate temperature, laser energy density, pulse frequency, and film thickness can cause RRR (Residual Resistivity Ratio) values to vary by more than 2× across different labs under nominally identical conditions. This heterogeneity, driven by hidden variables (chamber history, target batch differences, substrate surface termination), makes conventional trial-and-error optimization extremely costly.

This project proposes a novel paradigm: **use a domain-adapted LLM as a physics constraint encoder rather than a numerical predictor**, pre-train a hybrid XGBoost-BNN surrogate on 311 public literature data points with LLM-generated ordinal ranking constraints, then transfer it to a local PLD system with only 16 Sobol calibration experiments.

---

## Key Features

- **Hybrid XGBoost-BNN Model**: XGBoost captures deterministic nonlinear mappings; BNN (MC-Dropout) quantifies uncertainty
- **LLM Physics Constraint Injection**: Domain-adapted DeepSeek-R1 (32B, QLoRA) generates ordinal parameter rankings; sequence-ranking loss embeds physical priors into BNN training — inspired by Physics-Informed Neural Networks (PINN)
- **Multi-objective Hyperparameter Optimization**: Optuna TPE with Sobol initialization, 3-objective Pareto frontier (STO R², mode-dependent secondary score, training stability)
- **Two-stage Transfer Learning**: STO-subset fine-tuning → knowledge-distillation adaptation to 16 local Sobol calibration experiments
- **Adaptive Bayesian Optimization**: EI → UCB → LogEI acquisition function switching across 150 iterations; thickness marginalization for fixed-thickness mode
- **Mode-specific Model Selection**: Different composite scoring weights for LLM-constrained (0.2, 0.1, 0.7) and unconstrained (0.3, 0.3, 0.4) modes

---

## Workflow Overview

```
Literature Data (322 samples) + LLM Domain Adaptation
        │
        ▼
┌─────────────────────────────┐
│  1. Hybrid Model Training   │  XGBoost + BNN + LLM Sequence-Ranking Loss
│     (Hyperparameter Optim.) │  Optuna TPE, 2048 trials, 3 objectives
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  2. Two-Stage Transfer      │  Stage 1: STO-subset fine-tuning
│     Learning                │  Stage 2: Knowledge distillation (16 Sobol)
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  3. Pareto Frontier         │  5-objective: STO R², secondary, stability,
│     + Grid Search Filter    │  fine-tuning R², retention R²
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  4. Model Selection         │  Mode-specific composite scoring
│     (within-mode ranking)   │  LLM: (0.2, 0.1, 0.7) / Tradition: (0.3, 0.3, 0.4)
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  5. Bayesian Optimization   │  Sobol sampling → EI → UCB → LogEI
│     → Top-10 Recipes        │  Thickness marginalization for fixed-d mode
└─────────────────────────────┘
```

---

## Project Structure

```
SRO-PLD-RRR/
├── optim_prediction_main.py        # Main pipeline entry point
├── model_optim.py                  # Hyperparameter optimization (Optuna)
├── fine-tune.py                    # Transfer learning fine-tuning
├── model_evaluation.py             # Model scoring & evaluation
├── calculate_pareto_solution.py    # Pareto frontier analysis
├── bayesian_optimization.py        # Bayesian optimization engine
├── consolidate&physics_evaluate.py # Results consolidation & LLM evaluation
├── requirements.txt                # Python dependencies
│
├── utils/
│   ├── data_processer.py           # Feature engineering & boundary penalty
│   ├── model_utils.py              # Shared model utilities
│   ├── model_optim_utils.py        # Optuna tuning pipeline helpers
│   ├── optim_prediction_utils.py   # Bayesian optimization utilities
│   └── tuning_config.py            # Hyperparameter search space config
│
├── data/
│   ├── converted_file.xlsx         # Preprocessed public literature dataset (311 samples)
│   ├── extracted_conditions_good.csv  # LLM-generated good parameter sequences
│   ├── extracted_conditions_bad.csv   # LLM-generated bad parameter sequences
│   └── sobol_samples_results.csv   # 16 Sobol calibration experiment results
│
├── LLM/                            # LLM-mode model definitions
│   └── XGB_BNN_{type}_hybrid_model/
│       └── model.py                # Model definition (attention/series/uncertainty_1/2)
│
└── tradition/                      # Traditional-mode model definitions
    └── XGB_BNN_{type}_hybrid_model/
        └── model.py
```

---

## Installation

**Requirements**: Python 3.10+, CUDA 11.8+ (recommended for GPU acceleration)

```bash
# Clone the repository
git clone https://github.com/luozj1020/SRO-PLD-RRR.git
cd SRO-PLD-RRR

# Install dependencies
pip install -r requirements.txt
```

**Hardware recommendations**:

| Stage | CPU | RAM | GPU | Time |
|-------|-----|-----|-----|------|
| Hyperparameter optimization (2048 trials) | 16-core | 32 GB | 8 GB (optional) | 6–12 h |
| Transfer learning fine-tuning | 8-core | 16 GB | 8 GB+ | 2–4 h |
| LLM evaluation | 4-core | 32 GB | RTX 4090 (48 GB) | 2–5 h/batch |
| Bayesian optimization (150 iterations) | 4-core | 8 GB | 4 GB | 1–3 h |

---

## Usage

### Step 1: Hyperparameter Optimization

```bash
# Traditional mode (no LLM physics constraints)
python model_optim.py --mode tradition --model_type series

# LLM mode (with physics constraint injection)
python model_optim.py --mode LLM --model_type series
```

### Step 2: Transfer Learning Fine-tuning

```bash
python fine-tune.py --mode LLM --model_type series
```

### Step 3: Pareto Frontier + Bayesian Optimization

```bash
# Fixed frequency + thickness marginalization (recommended for validation)
python optim_prediction_main.py \
    --mode LLM \
    --model_type series \
    --optim_mode fix_freq_thick \
    --fixed_frequency 4.0 \
    --thickness_mean 20.0 \
    --thickness_std 10.0 \
    --n_thickness_samples 20 \
    --use_gpu

# Fixed frequency only, optimize thickness
python optim_prediction_main.py \
    --mode LLM \
    --model_type series \
    --optim_mode fix_freq \
    --fixed_frequency 4.0 \
    --use_gpu
```

### Step 4: Model Evaluation

```bash
python model_evaluation.py --mode LLM --model_type series
python 'consolidate&physics_evaluate.py' --mode LLM --model_type series
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | `tradition` or `LLM` | required |
| `--model_type` | `attention`, `series`, `uncertainty_1`, `uncertainty_2` | required |
| `--optim_mode` | `optim_all`, `fix_freq`, `fix_thick`, `fix_freq_thick` | `fix_freq` |
| `--fixed_frequency` | Fixed laser frequency (Hz) | `4.0` |
| `--thickness_mean` | Mean of thickness distribution (nm) | `20.0` |
| `--thickness_std` | Std of thickness distribution (nm) | `10.0` |
| `--n_thickness_samples` | Monte-Carlo thickness draws per evaluation | `20` |
| `--use_gpu` | Enable GPU acceleration | `False` |

**Optimization modes**:

| Mode | Variables | Frequency | Thickness |
|------|-----------|-----------|-----------|
| `optim_all` | 5 (O₂, E, T, f, d) | free | free |
| `fix_freq` | 4 (O₂, E, T, d) | fixed | free |
| `fix_thick` | 4 (O₂, E, T, f) | free | marginalized |
| `fix_freq_thick` | 3 (O₂, E, T) | fixed | marginalized |

---

## Algorithm Details

### LLM as Physics Constraint Encoder

The LLM is **not** used as a numerical predictor or data generator. Instead, it generates ordinal rankings of parameter sequences (good/bad), which are embedded into BNN training via sequence-ranking loss:

$$\mathcal{L}_{physics} = \mathcal{L}_{intra\_good} + \mathcal{L}_{intra\_bad} + \mathcal{L}_{inter}$$

This approach avoids LLM hallucination risks while extracting qualitative physical knowledge.

### Mode-Dependent Secondary Objective

| Mode | Secondary Objective | Direction |
|------|--------------------|----|
| LLM-constrained | Training loss $\mathcal{L}_{total}$ | minimize |
| Unconstrained | 3-fold CV $R^2$ | maximize |

### Thickness Marginalization

In fixed-thickness mode, BNN uncertainty includes thickness sampling:

$$\sigma_{total} = \sqrt{\sigma_{model}^2 + \sigma_{thickness}^2}$$

where $\sigma_{model}$ is the mean model uncertainty across samples, and $\sigma_{thickness}$ is the variance of predictions across thickness draws from $N(d_0, \sigma_d)$.

### Composite Model Selection Score

$$\mathcal{S}_{final} = w_{pre} \cdot \mathcal{S}_{pretrain} + w_{phy} \cdot \mathcal{S}_{physical} + w_{ft} \cdot \mathcal{S}_{fine\text{-}tune} - w_D \cdot P_{bayesian} - w_E \cdot P_{optimal}$$

Weights are mode-specific:
- **LLM mode**: $(w_{pre}, w_{phy}, w_{ft}) = (0.2, 0.1, 0.7)$
- **Unconstrained mode**: $(w_{pre}, w_{phy}, w_{ft}) = (0.3, 0.3, 0.4)$

---

## Experimental Results

| Experiment | Best RRR | Method |
|------------|----------|--------|
| Sobol calibration (16 samples) | 6.35 | Sample 6 (high O₂) |
| Fixed-frequency deployment | 6.28 | LLM-constrained/Series |
| Fixed-frequency, fixed-thickness panel | 8.70 | LLM-constrained/Series |

---

## Data Availability

The public training release contains:

- `data/converted_file.xlsx`: 311 literature-derived SRO PLD records with a non-empty `paper` field;
- `data/extracted_conditions_good.csv`: the favourable ordinal sequence pool;
- `data/extracted_conditions_bad.csv`: the unfavourable ordinal sequence pool.

Experimental calibration and validation data underlying the reported figures and tables are provided in the Article and Supplementary Information. Eleven archived records from an earlier in-house PLD system are not public because they contain unpublished equipment-log parameters; they may be requested from the corresponding author for verification of the pretraining analysis.

## Release Scope

This repository is the public code and training-source snapshot associated with the manuscript. It intentionally excludes manuscript and Supplementary Information files, downloaded papers or extracted full text, raw instrument logs, server/account information, generated model checkpoints, and the 11 non-public archived records. GitHub release tags identify versioned submission snapshots.

`requirements.txt` specifies compatible software requirements rather than an exact environment lock. Generated models and numerical results can depend on the operating system, accelerator, and installed package versions.

## LLM Model Availability

The 32B base model and the continued-pretraining/QLoRA adapter are not stored in this Git repository. Set `--LLM_base_model_path` to a locally obtained DeepSeek-R1-Distill-Qwen-32B-compatible base model and `--LLM_finetuned_model_path` to the corresponding adapter directory when running the LLM evaluation workflow. The adapter is available to editors and referees from the corresponding authors on request.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@unpublished{luo2026language,
  title  = {Language model priors for sparse adaptation of experimental process models},
  author = {Luo, Zijin and Yao, Runze and Gan, Yulin and Wang, Yulong and Wang, Tianyang and Chen, Kai and Deng, Zhixiong and Liao, Zhaoliang},
  note   = {Manuscript submitted},
  year   = {2026},
  url    = {https://github.com/luozj1020/SRO-PLD-RRR}
}
```

---

## License

The source code in this repository is released under the [MIT License](LICENSE). Third-party models, papers, datasets, and other externally supplied materials remain subject to their respective licences and terms.
