# SRO-PLD-RRR: 语言模型先验用于实验过程模型的稀疏适配

<p align="center">
  <strong>中文文档</strong> | <a href="README.md">English</a>
</p>

> 一种物理约束的混合机器学习框架，集成LLM编码的领域知识、迁移学习和自适应贝叶斯优化，用于SrRuO₃薄膜脉冲激光沉积（PLD）生长参数优化。

---

## 目录

- [背景](#背景)
- [核心特性](#核心特性)
- [工作流程](#工作流程)
- [项目结构](#项目结构)
- [安装](#安装)
- [使用方法](#使用方法)
- [算法细节](#算法细节)
- [数据可用性](#数据可用性)
- [发布范围](#发布范围)
- [LLM模型可用性](#llm模型可用性)
- [引用](#引用)
- [许可](#许可)

---

## 背景

PLD生长的SrRuO₃（SRO）薄膜对工艺参数异常敏感——氧分压、衬底温度、激光能量密度、脉冲频率和薄膜厚度的微小变化可导致RRR（残余电阻率比）在不同实验室间产生2倍以上的差异。这种异质性由隐藏变量（腔体历史、靶材批次、衬底表面终端）驱动，使得传统的试错优化成本极高。

本项目提出一种新范式：**将领域适配的大语言模型（LLM）用作物理约束编码器而非数值预测器**，在311条公开文献数据上使用LLM生成的序数排序约束预训练混合XGBoost-BNN代理模型，然后通过16次Sobol校准实验将其迁移至本地PLD系统。

---

## 核心特性

- **混合XGBoost-BNN模型**：XGBoost捕获确定性非线性映射；BNN（MC-Dropout）量化不确定性
- **LLM物理约束注入**：领域适配的DeepSeek-R1（32B，QLoRA）生成序数参数排序；序列排序损失将物理先验嵌入BNN训练——受物理信息神经网络（PINN）启发
- **多目标超参数优化**：Optuna TPE + Sobol初始化，3目标帕累托前沿（STO R²、模式相关次级目标、训练稳定性）
- **两阶段迁移学习**：STO子集微调 → 知识蒸馏适配至16次Sobol校准实验
- **自适应贝叶斯优化**：EI → UCB → LogEI采集函数切换（150次迭代）；固定厚度模式下的厚度边际化
- **模式相关模型选择**：LLM约束模式（0.2, 0.1, 0.7）和无约束模式（0.3, 0.3, 0.4）使用不同的综合评分权重

---

## 工作流程

```
公开文献数据（311样本）+ LLM领域适配
        │
        ▼
┌─────────────────────────────┐
│  1. 混合模型训练            │  XGBoost + BNN + LLM序列排序损失
│     （超参数优化）          │  Optuna TPE, 2048次试验, 3目标
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  2. 两阶段迁移学习          │  阶段1：STO子集微调
│                             │  阶段2：知识蒸馏（16次Sobol）
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  3. 帕累托前沿              │  5目标：STO R²、次级、稳定性、
│     + 网格搜索过滤          │  微调R²、保持R²
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  4. 模型选择                │  模式相关综合评分
│     （模式内排序）          │  LLM: (0.2, 0.1, 0.7) / 传统: (0.3, 0.3, 0.4)
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  5. 贝叶斯优化              │  Sobol采样 → EI → UCB → LogEI
│     → Top-10工艺窗口        │  固定厚度模式下的厚度边际化
└─────────────────────────────┘
```

---

## 项目结构

```
SRO-PLD-RRR/
├── optim_prediction_main.py        # 主流程入口
├── model_optim.py                  # 超参数优化（Optuna）
├── fine-tune.py                    # 迁移学习微调
├── model_evaluation.py             # 模型评分与评估
├── calculate_pareto_solution.py    # 帕累托前沿分析
├── bayesian_optimization.py        # 贝叶斯优化引擎
├── consolidate&physics_evaluate.py # 结果整合与LLM评估
├── requirements.txt                # Python依赖
│
├── utils/
│   ├── data_processer.py           # 特征工程与边界惩罚
│   ├── model_utils.py              # 共享模型工具
│   ├── model_optim_utils.py        # Optuna调优管道辅助
│   ├── optim_prediction_utils.py   # 贝叶斯优化工具
│   └── tuning_config.py            # 超参数搜索空间配置
│
├── data/
│   ├── converted_file.xlsx         # 预处理公开文献数据集（311样本）
│   ├── extracted_conditions_good.csv  # LLM生成的优良参数序列
│   ├── extracted_conditions_bad.csv   # LLM生成的不良参数序列
│   └── sobol_samples_results.csv   # 16次Sobol校准实验结果
│
├── LLM/                            # LLM模式模型定义
│   └── XGB_BNN_{type}_hybrid_model/
│       └── model.py                # 模型定义（attention/series/uncertainty_1/2）
│
└── tradition/                      # 传统模式模型定义
    └── XGB_BNN_{type}_hybrid_model/
        └── model.py
```

---

## 安装

**要求**：Python 3.10+，CUDA 11.8+（推荐用于GPU加速）

```bash
# 克隆仓库
git clone https://github.com/luozj1020/SRO-PLD-RRR.git
cd SRO-PLD-RRR

# 安装依赖
pip install -r requirements.txt
```

**硬件建议**：

| 阶段 | CPU | 内存 | GPU | 时间 |
|------|-----|------|-----|------|
| 超参数优化（2048次试验） | 16核 | 32 GB | 8 GB（可选） | 6–12小时 |
| 迁移学习微调 | 8核 | 16 GB | 8 GB+ | 2–4小时 |
| LLM评估 | 4核 | 32 GB | RTX 4090（48 GB） | 2–5小时/批次 |
| 贝叶斯优化（150次迭代） | 4核 | 8 GB | 4 GB | 1–3小时 |

---

## 使用方法

### 步骤1：超参数优化

```bash
# 传统模式（无LLM物理约束）
python model_optim.py --mode tradition --model_type series

# LLM模式（带物理约束注入）
python model_optim.py --mode LLM --model_type series
```

### 步骤2：迁移学习微调

```bash
python fine-tune.py --mode LLM --model_type series
```

### 步骤3：帕累托前沿 + 贝叶斯优化

```bash
# 固定频率 + 厚度边际化（推荐用于验证）
python optim_prediction_main.py \
    --mode LLM \
    --model_type series \
    --optim_mode fix_freq_thick \
    --fixed_frequency 4.0 \
    --thickness_mean 20.0 \
    --thickness_std 10.0 \
    --n_thickness_samples 20 \
    --use_gpu

# 仅固定频率，优化厚度
python optim_prediction_main.py \
    --mode LLM \
    --model_type series \
    --optim_mode fix_freq \
    --fixed_frequency 4.0 \
    --use_gpu
```

### 步骤4：模型评估

```bash
python model_evaluation.py --mode LLM --model_type series
python 'consolidate&physics_evaluate.py' --mode LLM --model_type series
```

### 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode` | `tradition` 或 `LLM` | 必填 |
| `--model_type` | `attention`, `series`, `uncertainty_1`, `uncertainty_2` | 必填 |
| `--optim_mode` | `optim_all`, `fix_freq`, `fix_thick`, `fix_freq_thick` | `fix_freq` |
| `--fixed_frequency` | 固定激光频率（Hz） | `4.0` |
| `--thickness_mean` | 厚度分布均值（nm） | `20.0` |
| `--thickness_std` | 厚度分布标准差（nm） | `10.0` |
| `--n_thickness_samples` | 每次评估的蒙特卡洛厚度采样数 | `20` |
| `--use_gpu` | 启用GPU加速 | `False` |

**优化模式**：

| 模式 | 优化变量 | 频率 | 厚度 |
|------|----------|------|------|
| `optim_all` | 5个（O₂, E, T, f, d） | 自由 | 自由 |
| `fix_freq` | 4个（O₂, E, T, d） | 固定 | 自由 |
| `fix_thick` | 4个（O₂, E, T, f） | 自由 | 边际化 |
| `fix_freq_thick` | 3个（O₂, E, T） | 固定 | 边际化 |

---

## 算法细节

### LLM作为物理约束编码器

LLM**不**用作数值预测器或数据生成器。相反，它生成参数序列的序数排序（优/劣），通过序列排序损失嵌入BNN训练：

$$\mathcal{L}_{physics} = \mathcal{L}_{intra\_good} + \mathcal{L}_{intra\_bad} + \mathcal{L}_{inter}$$

这种方法避免了LLM幻觉风险，同时提取定性物理知识。

### 模式相关次级目标

| 模式 | 次级目标 | 方向 |
|------|----------|------|
| LLM约束模式 | 训练损失 $\mathcal{L}_{total}$ | 最小化 |
| 无约束模式 | 3折交叉验证 $R^2$ | 最大化 |

### 厚度边际化

在固定厚度模式下，BNN不确定性包含厚度采样：

$$\sigma_{total} = \sqrt{\sigma_{model}^2 + \sigma_{thickness}^2}$$

其中$\sigma_{model}$为各采样的模型标准差平均值，$\sigma_{thickness}$为从$N(d_0, \sigma_d)$采样的预测均值方差。

### 综合模型选择评分

$$\mathcal{S}_{final} = w_{pre} \cdot \mathcal{S}_{pretrain} + w_{phy} \cdot \mathcal{S}_{physical} + w_{ft} \cdot \mathcal{S}_{fine\text{-}tune} - w_D \cdot P_{bayesian} - w_E \cdot P_{optimal}$$

权重为模式相关：
- **LLM模式**：$(w_{pre}, w_{phy}, w_{ft}) = (0.2, 0.1, 0.7)$
- **无约束模式**：$(w_{pre}, w_{phy}, w_{ft}) = (0.3, 0.3, 0.4)$

---

## 实验结果

| 实验 | 最佳RRR | 方法 |
|------|---------|------|
| Sobol校准（16样本） | 6.35 | 样品6（高氧压） |
| 固定频率部署 | 6.28 | LLM约束/Series |
| 固定频率、固定厚度面板 | 8.70 | LLM约束/Series |

---

## 数据可用性

公开的训练数据包括：

- `data/converted_file.xlsx`：311条 `paper` 字段非空的SRO PLD文献来源记录；
- `data/extracted_conditions_good.csv`：favourable ordinal sequence pool；
- `data/extracted_conditions_bad.csv`：unfavourable ordinal sequence pool。

支撑论文图表的实验校准和验证数据在正文及补充信息中提供。早期组内PLD系统的11条归档记录含有未公开的设备日志参数，因此不予公开；如需核验预训练分析，可向通讯作者合理申请。

## 发布范围

本仓库是与投稿论文配套的公开代码和训练来源快照。仓库有意不包含主文和补充信息文件、下载的论文或全文提取内容、原始设备日志、服务器/账号信息、生成的模型检查点和11条非公开历史记录。GitHub release tag用于标识已版本化的投稿快照。

`requirements.txt` 描述兼容的软件要求，而非完全锁定的运行环境。生成模型和数值结果可能受操作系统、加速器和实际安装软件版本影响。

## LLM模型可用性

32B基础模型和持续预训练/QLoRA adapter不存储在本Git仓库中。运行LLM评估流程时，请将 `--LLM_base_model_path` 设为本地获取的DeepSeek-R1-Distill-Qwen-32B兼容基础模型，并将 `--LLM_finetuned_model_path` 设为相应adapter目录。编辑和审稿人可向通讯作者申请获取adapter。

---

## 引用

如果您在研究中使用此代码，请引用：

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

## 许可

本仓库源代码以 [MIT License](LICENSE) 开源。第三方模型、论文、数据集及其他外部材料仍适用其各自的许可和使用条款。
