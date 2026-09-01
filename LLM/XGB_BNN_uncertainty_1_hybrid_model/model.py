import logging
import os
import sys
import ast
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, Subset, Dataset
from tqdm import tqdm

# 添加项目根目录到系统路径
current_script_path = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))
sys.path.insert(0, root_dir)
# 导入辅助模块
from utils.model_utils import ModelVisualizer, create_experiment_report, setup_seed
from utils.data_processer import (
    ClusteringBasedWeightCalculator,
    EnhancedFeatureProcessor,
    calculate_boundary_penalty_weights,
    compute_fixed_covariance_matrix,
    DataAugmentationSuite,
    BOUND_1,
    BOUND_2
)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# 设置matplotlib字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def uncertainty_based_filtering(
        X, y, model,
        uncertainty_threshold_percentile=85,
        min_samples_to_keep=200
):
    """
    Remove only the most uncertain predictions based on BNN uncertainty estimates.

    Args:
        X (np.ndarray or torch.Tensor): Input features.
        y (np.ndarray or torch.Tensor): Target values.
        model (HybridModel): The fitted model containing the BNN.
        uncertainty_threshold_percentile (float): Percentile to set the uncertainty threshold (e.g., 85).
        min_samples_to_keep (int): Minimum number of samples to keep after filtering.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Filtered X and y arrays.
    """
    if not model.bnn_model or not model.is_fitted:
        logger.warning("Model or BNN not fitted, cannot perform uncertainty filtering. Returning original data.")
        return X, y

    # Use the model's predict method to get uncertainties
    _, uncertainties = model.predict(pd.DataFrame(X, columns=model.base_features) if isinstance(X, np.ndarray) else X)
    threshold = np.percentile(uncertainties, uncertainty_threshold_percentile)

    keep_mask = uncertainties < threshold
    num_kept = np.sum(keep_mask)

    # Ensure minimum sample size
    if num_kept < min_samples_to_keep:
        # logger.info(f"Number of samples after uncertainty filtering ({num_kept}) is below min_samples_to_keep ({min_samples_to_keep}). Adjusting.")
        # Keep the indices of the min_samples_to_keep samples with the lowest uncertainty
        indices_to_keep = np.argsort(uncertainties)[:min_samples_to_keep]
        keep_mask = np.zeros(len(uncertainties), dtype=bool)
        keep_mask[indices_to_keep] = True

    # logger.info(f"Uncertainty filtering: Kept {np.sum(keep_mask)}/{len(keep_mask)} samples.")

    if isinstance(X, pd.DataFrame):
        filtered_X = X.iloc[keep_mask]
        filtered_y = y.iloc[keep_mask] if isinstance(y, pd.Series) else y[keep_mask]
    else:  # Assume numpy arrays
        filtered_X = X[keep_mask]
        filtered_y = y[keep_mask]

    return filtered_X, filtered_y


@dataclass
class ModelConfig:
    """Hybrid model configuration with enhanced STO training options and boundary penalty"""
    # XGBoost parameters
    xgb_params: Dict = field(default_factory=lambda: {
        'n_estimators': 977,
        'learning_rate': 6.8921e-02,
        'max_depth': 10,
        'reg_alpha': 6.5598e-01,
        'reg_lambda': 1.9378,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'tree_method': 'hist' if torch.cuda.is_available() else 'auto',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'n_jobs': -1
    })
    # 添加掩码配置
    mask_params: Dict = field(default_factory=lambda: {
        'use_mask': True,  # 是否使用掩码机制
        'mask_attention_layers': [64, 32],  # 掩码注意力层维度
        'mask_dropout': 0.1,  # 掩码注意力dropout
    })
    # Hidden layer dimension exponent for BNN
    first_hidden_dims_pow: int = 7

    # BNN parameters (dynamically calculated)
    @property
    def bnn_params(self) -> Dict:
        base_dim = 2 ** self.first_hidden_dims_pow
        return {
            'hidden_dims': [base_dim, base_dim // 2, base_dim // 4],
            'dropout_rates': [2.6680e-01, 2.2358e-01, 2.3204e-01],
            'use_batchnorm': False,
            'use_layernorm': True,
            'activation': 'silu',
            'use_mask': self.mask_params['use_mask'],  # 添加掩码配置
        }

    # Training parameters
    training_params: Dict = field(default_factory=lambda: {
        'bnn_epochs': 1000,  # Modified: Reduced from 1000
        'batch_size': 64,
        'learning_rate': 5e-3,  # Modified: Reduced from 7.9547e-03
        'weight_decay': 5e-3,  # Modified: Reduced from 1e-2
        'clip_grad_norm': 1.0,
        'patience': 50,  # Modified: Increased from 10
        'min_delta_loss': 1e-7,
        'min_delta_r2': 1e-5,
        'scheduler_factor': 0.8,
        'scheduler_patience': 10,
        'verbose_epoch': 10,
        'stability_window': 10,
        'stability_threshold': 0.01,
        'min_epochs': 100,  # Modified: Increased from 50
        'warmup_epochs': 30,
        # 自适应Huber loss参数
        'initial_huber_delta': 1.0,  # 初始delta值
        'final_huber_delta': 0.3,  # 最终delta值
        # Uncertainty pruning parameters
        'uncertainty_pruning': True,
        'uncertainty_threshold_percentile': 85,
        'uncertainty_pruning_min_samples': 200,
        'uncertainty_pruning_frequency': 50,  # Prune every N epochs
    })
    # Data augmentation parameters
    augmentation_params: Dict = field(default_factory=lambda: {
        'max_iterations': 10,  # Modified: Reduced from 20
        'strategy_weights': {
            'gaussian_noise': 0.25,  # Modified: Reduced from 0.35
            'interpolation': 0.40,  # Modified: Increased from 0.30
            'knn': 0.20,
            'smote_like': 0.15,
        }
    })
    # Weight calculation parameters
    weight_params: Dict = field(default_factory=lambda: {
        'gaussian_sigma': 1.0,  # Modified: Increased from 0.8
        'missing_penalty_rate': 0.97,  # Modified: Increased from 0.95
        'penalty_sharpness': 1.2,  # Modified: Reduced from 1.5
        'use_mahalanobis': True,
        'weight_threshold': 0.05,  # Modified: Reduced from 0.1
        'apply_threshold_to_val': False,  # Added: Verify this logic in fit
    })
    weight_calculation: Dict = field(default_factory=lambda: {
        # Weight calculation method for TRAINING data
        # Options: 'clustering', 'boundary', 'hybrid'
        'method': 'hybrid',
        # Clustering parameters (used if method includes clustering)
        'clustering': {
            'n_clusters': None,  # Auto-detect if None
            'min_cluster_size': 5,
            'noise_percentile': 15,
            'methods': ['kmeans', 'density', 'residual'],  # Clustering algorithms to use
        },
        # How to handle validation/test weights
        # Options: 'distance_to_clusters', 'boundary_only', 'uniform'
        'validation_method': 'boundary_only',
    })
    # Output control
    output_control: Dict = field(default_factory=lambda: {
        'mode': 'detailed',
        'show_progress_bar': True,
        'show_epoch_details': True,
        'show_parameter_info': True,
        'show_early_stopping_info': True,
        'show_final_summary': True,
        'log_level': 'INFO',
    })
    # Enhanced STO training options
    sto_training: Dict = field(default_factory=lambda: {
        'train_sto_only': False,
        'transfer_learning': True,
        'freeze_layers': True,  # Modified: Changed from True to False
        'fine_tune_epochs': 1000,  # Modified: Reduced from 500
        'xgb_fine_tune': False,
        'xgb_fine_tune_epochs': 100,
        'xgb_fine_tune_lr_ratio': 0.1,
        'sto_sample_weight_multiplier': 1.2,
        'other_weight': 0.9,
        'fine_tune_patience': 50,  # Modified: Reduced from 80
        'fine_tune_min_epochs': 200,
        'fine_tune_min_delta_loss': 1e-8,
        'fine_tune_min_delta_r2': 1e-7,
        'fine_tune_stability_window': 30,
        'fine_tune_stability_threshold': 0.0005,
        'fine_tune_warmup_epochs': 80,
        'fine_tune_lr_scale': 0.1,  # Modified: Increased from 0.08
        'fine_tune_weight_decay_scale': 3.0,
        'fine_tune_momentum_decay': 0.03,
        'weight_change_penalty': 1e-4,
        'adaptive_lr_reduction': True,
        'lr_reduction_factor': 0.8,
        'lr_reduction_patience': 25,
        'gradient_accumulation_steps': 16,
        'min_lr_ratio': 5e-5,
        'stability_check_frequency': 3,
        # Added: Huber loss delta parameter for fine-tuning
        'fine_tune_initial_huber_delta': 0.8,  # fine-tune阶段初始delta
        'fine_tune_final_huber_delta': 0.2,  # fine-tune阶段最终delta
        # Uncertainty pruning parameters for fine-tuning
        'fine_tune_uncertainty_pruning': True,
        'fine_tune_uncertainty_threshold_percentile': 80,  # Potentially stricter for fine-tuning
        'fine_tune_uncertainty_pruning_min_samples': 100,
        'fine_tune_uncertainty_pruning_frequency': 25,
    })
    # Sequence contrast loss parameters
    sequence_loss: Dict = field(default_factory=lambda: {
        'enabled': False,
        'good_data_path': '',
        'bad_data_path': '',
        'intra_good_penalty': 0.1,
        'intra_bad_penalty': 0.1,
        'inter_penalty': 0.01,
        'sequence_batch_size': 128
    })

    @property
    def sequence_loss_enabled(self):
        return self.sequence_loss.get('enabled', False)


class enable_dropout:
    """临时启用dropout的上下文管理器"""

    def __init__(self, model):
        self.model = model
        self.original_training = model.training

    def __enter__(self):
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.train(self.original_training)


class BNN(nn.Module):
    """Bayesian Neural Network"""

    def __init__(self, input_dim: int, config: Dict, use_mask: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = config.get('hidden_dims', [128, 64, 32])
        self.dropout_rates = config.get('dropout_rates', [0.15, 0.1, 0.05])
        self.use_batchnorm = config.get('use_batchnorm', False)
        self.use_layernorm = config.get('use_layernorm', True)
        self.activation_type = config.get('activation', 'silu')
        self.use_mask = use_mask

        # 掩码注意力机制
        if self.use_mask:
            self.mask_attention = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.SiLU(),
                nn.Linear(input_dim // 2, input_dim),
                nn.Sigmoid()
            )

        # 根据是否使用掩码调整输入维度
        effective_input_dim = input_dim * 2 if use_mask else input_dim
        self.network = self._build_network(effective_input_dim)
        self._initialize_weights()

    def _build_network(self, input_dim: int) -> nn.Sequential:
        """构建网络架构"""
        layers = []
        prev_dim = input_dim
        activation_fn = self._get_activation_function()

        for i, (h_dim, drop_rate) in enumerate(zip(self.hidden_dims, self.dropout_rates)):
            layers.append(nn.Linear(prev_dim, h_dim))

            if self.use_layernorm:
                layers.append(nn.LayerNorm(h_dim))
            elif self.use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))

            layers.append(activation_fn())
            layers.append(nn.Dropout(drop_rate))
            prev_dim = h_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)

    def _get_activation_function(self):
        """Get activation function class"""
        activations = {
            'relu': nn.ReLU,
            'silu': nn.SiLU,
            'gelu': nn.GELU,
            'elu': nn.ELU,
            'leaky_relu': nn.LeakyReLU
        }
        return activations.get(self.activation_type, nn.SiLU)

    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.activation_type in ['relu', 'leaky_relu']:
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                    with torch.no_grad():
                        module.weight.data *= 0.5
                else:
                    nn.init.xavier_normal_(module.weight, gain=0.3)
                nn.init.constant_(module.bias, 0.001)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播，支持掩码机制"""

        # 检查输入是否包含NaN或Inf
        if torch.any(torch.isnan(x)):
            logger.warning("Input x contains NaN values")
            x = torch.nan_to_num(x, nan=0.0)
        if torch.any(torch.isinf(x)):
            logger.warning("Input x contains Inf values")
            x = torch.clamp(x, min=-1e6, max=1e6)

        if self.use_mask:
            # 检查掩码是否包含NaN或Inf
            if torch.any(torch.isnan(mask)):
                logger.warning("Mask contains NaN values")
                mask = torch.nan_to_num(mask, nan=0.0)
            if torch.any(torch.isinf(mask)):
                logger.warning("Mask contains Inf values")
                mask = torch.clamp(mask, min=0.0, max=1.0)

            # 应用掩码注意力机制
            attention_weights = self.mask_attention(mask)
            # logger.info(f"Attention weights shape: {attention_weights.shape}")

            weighted_x = x * attention_weights
            # logger.info(f"Weighted x shape: {weighted_x.shape}")

            # 将加权特征和掩码连接起来
            x_combined = torch.cat([weighted_x, mask], dim=1)
            # logger.info(f"Combined x shape: {x_combined.shape}")
        else:
            x_combined = x
            # logger.info(f"Using original x shape: {x_combined.shape}")

        # 检查网络第一层的期望输入
        first_layer = self.network[0]
        if isinstance(first_layer, nn.Linear):
            # logger.info(f"First layer expects input dim: {first_layer.in_features}")

            # 检查输入维度是否匹配
            if x_combined.shape[1] != first_layer.in_features:
                logger.error(f"Dimension mismatch! Input has {x_combined.shape[1]} features, "
                             f"but first layer expects {first_layer.in_features} features")

                # 自动调整维度
                if x_combined.shape[1] < first_layer.in_features:
                    # 填充零
                    padding = torch.zeros(x_combined.shape[0],
                                          first_layer.in_features - x_combined.shape[1],
                                          device=x_combined.device)
                    x_combined = torch.cat([x_combined, padding], dim=1)
                    logger.warning(f"Padded input from {x_combined.shape[1]} to {first_layer.in_features} features")
                else:
                    # 截断
                    x_combined = x_combined[:, :first_layer.in_features]
                    logger.warning(f"Truncated input from {x_combined.shape[1]} to {first_layer.in_features} features")

        return self.network(x_combined)

    def mc_predict(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                   n_samples: int = 50) -> np.ndarray:
        """
        Monte Carlo预测，增强掩码处理
        """
        self.eval()

        # 设备转移
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=device)
        elif x.device != device:
            x = x.to(device)

        # 增强掩码处理逻辑
        if mask is None:
            # 创建默认掩码
            mask = torch.ones_like(x, device=device)
            logger.debug("No mask provided in mc_predict, using default mask (all ones)")
        else:
            if isinstance(mask, np.ndarray):
                mask = torch.tensor(mask, dtype=torch.float32, device=device)
            elif mask.device != device:
                mask = mask.to(device)

        # 确保掩码形状匹配
        if mask.shape != x.shape:
            logger.warning(f"Mask shape {mask.shape} doesn't match input shape {x.shape}. Adjusting mask.")
            # 更智能的形状调整
            if mask.shape[0] != x.shape[0]:
                if mask.shape[0] == 1:  # 广播单个样本的掩码
                    mask = mask.repeat(x.shape[0], 1)
                else:
                    # 截断或填充以匹配批次大小
                    min_batch = min(mask.shape[0], x.shape[0])
                    mask = mask[:min_batch]
                    x = x[:min_batch]

            if mask.shape[1] != x.shape[1]:
                min_features = min(mask.shape[1], x.shape[1])
                mask = mask[:, :min_features]
                x = x[:, :min_features]

        # MC采样，启用dropout
        with torch.no_grad():
            # 启用dropout
            for module in self.modules():
                if isinstance(module, nn.Dropout):
                    module.train()

            batch_size = min(500, max(50, n_samples))
            all_predictions = []

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                current_batch_size = end_idx - start_idx

                # 重复输入以匹配批次大小
                x_batch = x.repeat(current_batch_size, 1)
                mask_batch = mask.repeat(current_batch_size, 1)

                # 关键修复：使用掩码进行预测
                pred_batch = self(x_batch, mask_batch)  # 传递掩码
                batch_predictions = pred_batch.view(current_batch_size, -1).T.cpu().numpy()
                all_predictions.append(batch_predictions)

            predictions = np.concatenate(all_predictions, axis=1)

        return predictions


class OrdinalDataset(Dataset):
    """Sequence contrast dataset"""

    def __init__(self, processor, good_data_path, bad_data_path, base_features):  # 添加base_features参数
        self.processor = processor
        self.base_features = base_features  # 存储base_features
        try:
            self.good_data = pd.read_csv(good_data_path)
            self.bad_data = pd.read_csv(bad_data_path)
            self.good_sequences = self._preprocess_data(self.good_data)
            self.bad_sequences = self._preprocess_data(self.bad_data)
        except Exception as e:
            logger.error(f"Sequence data loading failed: {str(e)}")
            self.good_sequences = []
            self.bad_sequences = []

    def _preprocess_data(self, data):
        sequences = []
        feature_columns = self.base_features  # 使用存储的base_features
        for idx in range(len(data)):
            sequence = []
            for col in [f'condition_{i}' for i in range(1, 6)]:
                try:
                    if col in data.columns and not pd.isna(data[col].iloc[idx]):
                        # Parse the condition values
                        condition_values = np.array(ast.literal_eval(data[col].iloc[idx]), dtype=np.float32)
                        # Create a DataFrame row for processing
                        condition_df = pd.DataFrame([condition_values], columns=feature_columns)
                        # Process using the pre-trained processor
                        processed_condition = self.processor.transform(condition_df)[0]
                        sequence.append(processed_condition)
                    else:
                        # Use zeros if condition is missing
                        sequence.append(np.zeros(len(feature_columns), dtype=np.float32))
                except Exception as e:
                    logger.warning(f"Error processing condition: {str(e)}")
                    sequence.append(np.zeros(len(feature_columns), dtype=np.float32))
            sequences.append(np.stack(sequence))
        return sequences

    def __len__(self):
        return len(self.good_sequences) + len(self.bad_sequences)

    def __getitem__(self, idx):
        if idx < len(self.good_sequences):
            return torch.FloatTensor(self.good_sequences[idx]), 0
        else:
            return torch.FloatTensor(self.bad_sequences[idx - len(self.good_sequences)]), 1


def sequence_ranking_loss(bnn_model, residual_stats, sequences, labels, config):
    """Sequence contrast loss calculation"""
    batch_size, seq_len, feature_dim = sequences.shape
    flat_sequences = sequences.reshape(-1, feature_dim)

    device = next(bnn_model.parameters()).device
    flat_sequences_device = flat_sequences.to(device)
    sequence_mask = torch.ones_like(flat_sequences_device, device=device)

    bnn_residuals = bnn_model(flat_sequences_device, sequence_mask).reshape(batch_size, seq_len)
    residual_std = torch.as_tensor(residual_stats['std'], dtype=bnn_residuals.dtype, device=device)
    residual_mean = torch.as_tensor(residual_stats['mean'], dtype=bnn_residuals.dtype, device=device)
    final_preds = bnn_residuals * residual_std + residual_mean

    # Calculate losses
    intra_good_loss = torch.tensor(0.0, device=sequences.device)
    intra_bad_loss = torch.tensor(0.0, device=sequences.device)
    inter_loss = torch.tensor(0.0, device=sequences.device)

    good_mask = labels == 0
    bad_mask = labels == 1
    good_seqs = final_preds[good_mask]
    bad_seqs = final_preds[bad_mask]

    # Intra-good sequence loss
    if good_seqs.numel() > 0 and good_seqs.shape[1] > 1:
        good_diffs = good_seqs[:, :-1] - good_seqs[:, 1:]
        intra_good_loss = torch.sum(torch.relu(good_diffs)) * config.sequence_loss['intra_good_penalty']

    # Intra-bad sequence loss
    if bad_seqs.numel() > 0 and bad_seqs.shape[1] > 1:
        bad_diffs = bad_seqs[:, 1:] - bad_seqs[:, :-1]
        intra_bad_loss = torch.sum(torch.relu(bad_diffs)) * config.sequence_loss['intra_bad_penalty']

    # Inter-sequence loss
    if good_seqs.numel() > 0 and bad_seqs.numel() > 0:
        min_bad = bad_seqs.min(dim=1).values
        max_good = good_seqs.max(dim=1).values
        inter_diff = min_bad.unsqueeze(1) - max_good.unsqueeze(0)
        inter_loss = torch.sum(torch.relu(inter_diff)) * config.sequence_loss['inter_penalty']

    total_loss = intra_good_loss + intra_bad_loss + inter_loss
    return total_loss, intra_good_loss, intra_bad_loss, inter_loss


class HybridModel:
    """Hybrid XGBoost + BNN model with embedded processor"""

    def __init__(self, base_features: List[str], config: ModelConfig):
        self.config = config
        self.base_features = base_features
        self.input_dim = len(base_features)

        # 初始化XGBoost模型
        self.xgb_model = xgb.XGBRegressor(**config.xgb_params)
        self.bnn_model = None

        # 初始化processor（在fit时会拟合）
        self.processor = EnhancedFeatureProcessor(
            base_features=base_features,
            scaler_type='robust',
            interpolation_method='knn',
            n_neighbors=3
        )

        # 初始化数据增强器
        self.augmenter = DataAugmentationSuite(feature_processor=self.processor)

        # 训练历史和统计信息
        self.history = {
            'loss': [], 'r2': [], 'val_loss': [], 'val_r2': [],
            'learning_rates': [], 'feature_importance': [],
            'sequence_loss': [], 'intra_good_loss': [],
            'intra_bad_loss': [], 'inter_loss': []
        }
        self.residual_stats = {}
        self.is_fitted = False
        self.fixed_covariance_inv = None

        # Early stopping相关
        self._reset_early_stopping()
        self.last_lr = None
        self.cluster_calculator = None  # Will be set during fit if using clustering

        # Use MaskAwareBNN instead of regular BNN
        self.bnn_model = None

        self.bnn_var_mean = None
        self.bnn_var_std = None

    def _reset_early_stopping(self):
        """重置early stopping状态"""
        self.best_loss = float('inf')
        self.best_r2 = -float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        self.loss_history = []
        self.r2_history = []
        self.epochs_since_improvement = 0

    def adaptive_huber_delta(self, epoch, max_epochs, initial_delta=1.0, final_delta=0.3):
        """计算当前epoch的自适应Huber delta值"""
        progress = epoch / max_epochs
        delta = initial_delta * (1 - progress) + final_delta * progress
        return delta

    def _prepare_data_with_augmentation(self, X_raw: pd.DataFrame, y: pd.Series,
                                        sto_flags: Optional[np.ndarray] = None,
                                        target_size: int = None) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """数据增强流程，忽略权重较小的数据"""
        # Modified: Use conservative target size based on original size
        if target_size is None:
            original_train_size = len(X_raw)
            if original_train_size < 200:
                target_size = original_train_size * 3
            elif original_train_size < 500:
                target_size = original_train_size * 2
            else:
                target_size = int(original_train_size * 1.5)
        else:
            original_train_size = len(X_raw)

        max_iterations = self.config.augmentation_params['max_iterations']
        strategy_weights = self.config.augmentation_params['strategy_weights']
        weight_threshold = self.config.weight_params.get('weight_threshold', 0.05)

        # logger.info(f"开始数据增强流程，原始训练集大小: {original_train_size}")
        # logger.info(f"目标大小: {target_size}, 权重阈值: {weight_threshold}")

        # 拟合augmenter（仅在原始训练集上）
        augmenter_df = X_raw.copy()
        augmenter_df[self.config.output_control.get('target_column', 'rrr')] = y.values
        # 添加STO标记列到augmenter_df，用于数据增强
        if sto_flags is not None:
            augmenter_df['STO_flag'] = sto_flags
        else:
            augmenter_df['STO_flag'] = 0
        self.augmenter.fit(augmenter_df, self.config.output_control.get('target_column', 'rrr'))

        current_X = X_raw.copy()
        current_y = y.copy()
        if sto_flags is not None:
            current_sto = sto_flags.copy()
        else:
            current_sto = np.zeros(len(y))

        iteration = 0
        consecutive_no_progress = 0  # 跟踪连续无进展的迭代
        while len(current_X) < target_size and iteration < max_iterations:
            iteration += 1
            # logger.info(f"--- 数据增强迭代 {iteration} ---")
            # logger.info(f"当前训练集大小: {len(current_X)}")

            # 计算当前数据的权重（使用固定协方差矩阵）
            X_raw_values = current_X[self.base_features].values
            weights = calculate_boundary_penalty_weights(
                X_raw_values,
                self.base_features,
                gaussian_sigma=self.config.weight_params['gaussian_sigma'],
                missing_penalty_rate=self.config.weight_params['missing_penalty_rate'],
                penalty_sharpness=self.config.weight_params['penalty_sharpness'],
                use_mahalanobis=self.config.weight_params['use_mahalanobis'],
                fixed_covariance_inv=self.fixed_covariance_inv,
                weight_threshold=weight_threshold
            )

            # 移除权重低于阈值的样本
            before_size = len(current_X)
            valid_mask = weights >= weight_threshold
            current_X = current_X[valid_mask].reset_index(drop=True)
            current_y = current_y[valid_mask].reset_index(drop=True)
            current_sto = current_sto[valid_mask]
            removed = before_size - len(current_X)
            # logger.info(f"移除 {removed} 个权重低于{weight_threshold}的样本，当前大小: {len(current_X)}")

            # 检查是否需要继续增强
            if len(current_X) >= target_size:
                # logger.info("训练集已达到目标大小")
                break

            # Early stopping mechanism based on removal
            if removed == 0:
                consecutive_no_progress += 1
                if consecutive_no_progress >= 3:
                    # logger.info("连续3次迭代无改善（无样本被移除），提前停止")
                    break
            else:
                consecutive_no_progress = 0

            # 生成增强样本 - 修正：确保不会超过目标大小
            samples_needed = min(target_size - len(current_X), 200)
            # logger.info(f"需要增强 {samples_needed} 个样本")

            # 调整策略权重动态
            adjusted_weights = strategy_weights.copy()
            if iteration > 5:
                adjusted_weights['gaussian_noise'] *= 0.7
                adjusted_weights['interpolation'] *= 1.3
                # 重新归一化
                total = sum(adjusted_weights.values())
                adjusted_weights = {k: v / total for k, v in adjusted_weights.items()}

            augmented_df = self.augmenter.ensemble_augmentation(
                n_samples=samples_needed,
                strategy_weights=adjusted_weights
            )

            if len(augmented_df) > 0:
                # 提取特征和目标
                aug_X = augmented_df[self.base_features]
                aug_y = augmented_df[self.config.output_control.get('target_column', 'rrr')]
                # 关键修改：将所有增强数据标记为非STO
                aug_sto = np.zeros(len(aug_X))  # 增强的数据标记为非STO
                # 记录增强数据的STO标记信息
                # logger.info(f"增强生成 {len(aug_X)} 个样本，全部标记为非STO")

                # 合并数据 - 修正：确保不会超过目标大小
                current_size_before = len(current_X)
                current_X = pd.concat([current_X, aug_X], ignore_index=True)
                current_y = pd.concat([current_y, aug_y], ignore_index=True)
                current_sto = np.concatenate([current_sto, aug_sto])

                # 如果合并后超过了目标大小，进行截断
                if len(current_X) > target_size:
                    excess = len(current_X) - target_size
                    current_X = current_X.head(target_size)
                    current_y = current_y.head(target_size)
                    current_sto = current_sto[:target_size]
                    # logger.info(f"截断 {excess} 个超额样本，保持目标大小: {target_size}")

                # logger.info(f"增强后训练集大小: {len(current_X)}")

                # 记录当前STO分布
                sto_count = np.sum(current_sto == 1)
                non_sto_count = np.sum(current_sto == 0)
                # logger.info(f"当前STO分布: STO样本={sto_count}, 非STO样本={non_sto_count}")
            else:
                logger.warning("增强生成0个样本")
                break

        # logger.info(f"数据增强完成，最终训练集大小: {len(current_X)}")
        # 最终STO分布统计
        final_sto_count = np.sum(current_sto == 1)
        final_non_sto_count = np.sum(current_sto == 0)
        # logger.info(f"最终STO分布: STO样本={final_sto_count} ({final_sto_count / len(current_sto) * 100:.1f}%), "
        # f"非STO样本={final_non_sto_count} ({final_non_sto_count / len(current_sto) * 100:.1f}%)")

        return current_X, current_y, current_sto

    def _calculate_validation_weights_from_clusters(
            self,
            X_val: np.ndarray,
            cluster_calculator: ClusteringBasedWeightCalculator
    ) -> np.ndarray:
        """
        Calculate validation weights based on distance to training clusters
        This PREVENTS data leakage - we don't cluster validation data!
        Args:
            X_val: Validation feature matrix (raw, not processed)
            cluster_calculator: Fitted clustering calculator from training
        Returns:
            weights: Validation weights based on proximity to training clusters
        """
        # Get training cluster information
        if 'kmeans' not in cluster_calculator.cluster_info:
            # Fallback to uniform weights if clustering info not available
            logger.warning("No clustering info available, using uniform validation weights")
            return np.ones(len(X_val))

        kmeans_info = cluster_calculator.cluster_info['kmeans']
        train_centers = kmeans_info['centers']

        # Scale validation data using training scaler
        X_val_scaled = cluster_calculator.scaler.transform(X_val)

        # Calculate distance to nearest training cluster center
        distances_to_centers = np.array([
            np.min(np.linalg.norm(X_val_scaled - center, axis=1))
            for center in train_centers
        ]).T  # Shape: (n_val_samples,)

        # Convert distances to weights
        # Samples close to training clusters get high weight
        # Samples far from all training clusters get low weight
        median_distance = np.median(distances_to_centers)
        weights = np.exp(-distances_to_centers / (2 * median_distance))
        # Normalize
        weights = weights / np.mean(weights)
        weights = np.clip(weights, 0.3, 2.0)  # Less aggressive clipping for validation

        return weights

    def fit(self, X: pd.DataFrame, y: pd.Series,
            val_X: Optional[pd.DataFrame] = None,
            val_y: Optional[pd.Series] = None,
            sto_column: str = 'Substrate') -> 'HybridModel':
        """
        拟合模型
        Parameters:
        -----------
        X : pd.DataFrame
            原始训练数据（包含所有列）
        y : pd.Series
            目标变量
        val_X : pd.DataFrame, optional
            验证集特征
        val_y : pd.Series, optional
            验证集目标
        sto_column : str
            用于识别STO样本的列名
        """
        # 1. 提取STO标签 - 修改为只有'STO'才标记为STO
        if sto_column in X.columns:
            sto_flags = (X[sto_column] == 'STO').astype(int).values
        else:
            sto_flags = np.zeros(len(X))
            logger.warning(f"Column '{sto_column}' not found")

        X_features = X[self.base_features].copy()

        # *** Key Change 1: 在原始训练集上拟合processor（数据增强之前） ***
        self.processor.fit(X_features)

        # *** Key Change 2: 计算固定协方差矩阵使用原始训练集 ***
        X_raw_values = X_features.values
        self.fixed_covariance_inv = compute_fixed_covariance_matrix(
            X_raw_values, self.base_features, BOUND_1, BOUND_2
        )

        # 3. 数据增强（如果启用）
        original_train_size = len(X_features)
        augmentation_flags = np.zeros(len(X_features))  # 0表示原始数据
        if original_train_size < 200:
            target_multiplier = 5
        elif original_train_size < 500:
            target_multiplier = 2
        else:
            target_multiplier = 1.5
        dynamic_target_size = int(original_train_size * target_multiplier)

        if dynamic_target_size > original_train_size:
            X_augmented, y_augmented, sto_augmented = self._prepare_data_with_augmentation(
                X_features, y, sto_flags, target_size=dynamic_target_size
            )
            augmentation_flags = np.concatenate([
                np.zeros(original_train_size),  # 原始数据
                np.ones(len(X_augmented) - original_train_size)  # 增强数据
            ])
        else:
            X_augmented = X_features
            y_augmented = y
            sto_augmented = sto_flags

        # 4. Calculate training weights
        X_train_raw = X_augmented[self.base_features].values
        weight_calculation_method = self.config.weight_calculation.get('method', 'boundary')
        # logger.info(f"Calculating training weights using method: {weight_calculation_method}")

        if weight_calculation_method == 'clustering':
            # Pure clustering-based weights
            clustering_config = self.config.weight_calculation['clustering']
            cluster_calculator = ClusteringBasedWeightCalculator(
                n_clusters=clustering_config.get('n_clusters'),
                min_cluster_size=max(5, len(X_train_raw) // 50),
                noise_percentile=clustering_config.get('noise_percentile', 15),
                methods=clustering_config.get('methods', ['kmeans', 'density', 'residual']),
                feature_names=self.base_features
            )
            # *** KEY FIX: Use processor to impute NaN before clustering ***
            X_train_for_clustering = self.processor.transform(X_augmented[self.base_features])
            train_weights, diagnostics = cluster_calculator.fit_calculate_weights(
                X_train_for_clustering,  # Use imputed data!
                y_augmented.values,
                return_diagnostics=True
            )
            # logger.info(f"  Clustering detected {diagnostics['n_clusters_used']} clusters")
            # logger.info(f"  Low weight samples: {diagnostics['low_weight_samples']['count_below_0.5']} "
            # f"({diagnostics['low_weight_samples']['percentage_below_0.5']:.1f}%)")
            # Store calculator for validation weight calculation
            self.cluster_calculator = cluster_calculator

        elif weight_calculation_method == 'hybrid':
            # Hybrid: clustering + boundary penalty
            clustering_config = self.config.weight_calculation['clustering']
            cluster_calculator = ClusteringBasedWeightCalculator(
                n_clusters=clustering_config.get('n_clusters'),
                min_cluster_size=max(5, len(X_train_raw) // 50),
                noise_percentile=clustering_config.get('noise_percentile', 15),
                methods=clustering_config.get('methods', ['kmeans', 'density', 'residual']),
                feature_names=self.base_features
            )
            # *** KEY FIX: Use processor to impute NaN before clustering ***
            X_train_for_clustering = self.processor.transform(X_augmented[self.base_features])
            # Calculate both types of weights
            cluster_weights, diagnostics = cluster_calculator.fit_calculate_weights(
                X_train_for_clustering,  # Use imputed data!
                y_augmented.values,
                return_diagnostics=True
            )
            # Boundary weights still use raw data (designed to handle NaN)
            boundary_weights = calculate_boundary_penalty_weights(
                X_train_raw,  # Raw data is OK here
                self.base_features,
                gaussian_sigma=self.config.weight_params['gaussian_sigma'],
                missing_penalty_rate=self.config.weight_params['missing_penalty_rate'],
                penalty_sharpness=self.config.weight_params['penalty_sharpness'],
                use_mahalanobis=self.config.weight_params['use_mahalanobis'],
                fixed_covariance_inv=self.fixed_covariance_inv,
                weight_threshold=0.0  # Don't threshold yet
            )
            # Ensemble: geometric mean
            train_weights = np.sqrt(cluster_weights * boundary_weights)
            # Normalize
            train_weights = train_weights / np.mean(train_weights)
            train_weights = np.clip(train_weights, 0.05, 3.0)

            # logger.info(f"  Hybrid weighting: clustering + boundary penalty")
            # logger.info(f"  Clustering detected {diagnostics['n_clusters_used']} clusters")
            # logger.info(f"  Weight correlation: {np.corrcoef(cluster_weights, boundary_weights)[0, 1]:.3f}")
            # Store calculator for validation
            self.cluster_calculator = cluster_calculator

        else:  # 'boundary' (default/original)
            # Original boundary penalty weights only (handles NaN internally)
            weight_threshold = self.config.weight_params.get('weight_threshold', 0.05)
            train_weights = calculate_boundary_penalty_weights(
                X_train_raw,  # Raw data is OK for boundary weights
                self.base_features,
                gaussian_sigma=self.config.weight_params['gaussian_sigma'],
                missing_penalty_rate=self.config.weight_params['missing_penalty_rate'],
                penalty_sharpness=self.config.weight_params['penalty_sharpness'],
                use_mahalanobis=self.config.weight_params['use_mahalanobis'],
                fixed_covariance_inv=self.fixed_covariance_inv,
                weight_threshold=weight_threshold
            )
            self.cluster_calculator = None

        # Apply weight threshold (common for all methods)
        weight_threshold = self.config.weight_params.get('weight_threshold', 0.05)
        before_removal_count = len(X_augmented)
        valid_mask = train_weights >= weight_threshold
        X_augmented = X_augmented[valid_mask].reset_index(drop=True)
        y_augmented = y_augmented[valid_mask].reset_index(drop=True)
        sto_augmented = sto_augmented[valid_mask]
        train_weights = train_weights[valid_mask]
        augmentation_flags = augmentation_flags[valid_mask]
        removed_count = before_removal_count - len(X_augmented)
        # logger.info(f"  Removed {removed_count} samples with weight < {weight_threshold}")

        # 5. Handle validation set weights (if provided)
        val_weights = None
        val_sto_flags = None
        if val_X is not None and val_y is not None:
            if sto_column in val_X.columns:
                val_sto_flags = (val_X[sto_column] == 'STO').astype(int).values
            else:
                val_sto_flags = np.zeros(len(val_X))

            val_X_features = val_X[self.base_features]
            val_X_raw = val_X_features.values

            # Determine validation weighting method
            validation_method = self.config.weight_calculation.get('validation_method', 'boundary_only')
            # logger.info(f"Calculating validation weights using method: {validation_method}")

            if validation_method == 'distance_to_clusters' and self.cluster_calculator is not None:
                # *** KEY FIX: Impute NaN for validation data before distance calculation ***
                val_X_processed = self.processor.transform(val_X_features)
                # Use distance to training clusters (NO DATA LEAKAGE)
                val_weights = self._calculate_validation_weights_from_clusters(
                    val_X_processed,  # Use imputed data!
                    self.cluster_calculator
                )
                # logger.info(f"  Validation weights from cluster distances")
                # logger.info(f"  Weight range: [{np.min(val_weights):.3f}, {np.max(val_weights):.3f}]")

            elif validation_method == 'boundary_only' or self.cluster_calculator is None:
                # Use boundary penalty only (handles NaN internally)
                val_weights = calculate_boundary_penalty_weights(
                    val_X_raw,  # Raw data OK for boundary weights
                    self.base_features,
                    gaussian_sigma=self.config.weight_params['gaussian_sigma'],
                    missing_penalty_rate=self.config.weight_params['missing_penalty_rate'],
                    penalty_sharpness=self.config.weight_params['penalty_sharpness'],
                    use_mahalanobis=self.config.weight_params['use_mahalanobis'],
                    fixed_covariance_inv=self.fixed_covariance_inv,
                    weight_threshold=0.0
                )
                # logger.info(f"  Validation weights from boundary penalty")
            else:  # 'uniform'
                # No weighting for validation
                val_weights = np.ones(len(val_X_raw))
                # logger.info(f"  Uniform validation weights (all = 1.0)")

        # 6. 训练XGBoost（保留原始缺失模式）
        X_train_xgb = X_augmented[self.base_features].values
        self.xgb_model.fit(X_train_xgb, y_augmented.values, sample_weight=train_weights)
        self.history['feature_importance'].append({
            'xgb': self.xgb_model.feature_importances_,
            'epoch': 0
        })

        # 7. 为BNN准备完全插补的数据
        # *** Key Change 3: 使用processor转换增强后的训练数据 ***
        X_train_processed = self.processor.transform(X_augmented[self.base_features])

        # 8. 训练BNN
        if self.config.sto_training['transfer_learning']:
            # Pretrain阶段
            pretrain_early_stopped = self._train_bnn(
                X_train_xgb, X_train_processed, y_augmented.values,
                stage='pretrain',
                sto_flags=sto_augmented,
                sample_weights=train_weights
            )
            self.pretrain_early_stopped = pretrain_early_stopped
            self.pretrain_model_state = {
                name: param.clone().detach().cpu()
                for name, param in self.bnn_model.named_parameters()
            }

            # Fine-tune阶段（如果有STO样本）
            if self.config.sto_training.get('freeze_layers', False):
                total_layers = len(list(self.bnn_model.network))
                freeze_ratio = 0.6
                freeze_count = int(total_layers * freeze_ratio)
                for i, layer in enumerate(self.bnn_model.network):
                    if i < freeze_count:
                        for param in layer.parameters():
                            param.requires_grad = False

            sto_indices = np.where(sto_augmented == 1)[0]
            if len(sto_indices) > 0:
                X_sto_xgb = X_train_xgb[sto_indices]
                X_sto_processed = X_train_processed[sto_indices]
                y_sto = y_augmented.values[sto_indices]
                weights_sto = train_weights[sto_indices]

                self._train_bnn(
                    X_sto_xgb, X_sto_processed, y_sto,
                    stage='fine_tune',
                    sto_flags=np.ones(len(y_sto)),
                    sample_weights=weights_sto
                )
        else:
            # 标准训练
            self._train_bnn(
                X_train_xgb, X_train_processed, y_augmented.values,
                stage='normal',
                sto_flags=sto_augmented,
                sample_weights=train_weights
            )

        self.train_augmentation_flags = augmentation_flags
        self.train_X_augmented = X_augmented
        self.train_y_augmented = y_augmented
        self.train_weights = train_weights
        self.train_sto_flags = sto_augmented

        self.is_fitted = True
        return self

    def _train_bnn(self, X_raw: np.ndarray, X_processed: np.ndarray, y: np.ndarray,
                   stage: str = 'normal', sto_flags: Optional[np.ndarray] = None,
                   sample_weights: Optional[np.ndarray] = None):
        """训练BNN模型，支持掩码"""

        # 获取掩码（从processor中获取）
        X_processed, train_mask = self.processor.transform(
            pd.DataFrame(X_raw, columns=self.base_features),
            return_mask=True
        )

        # 添加维度检查
        # logger.info(f"X_processed shape: {X_processed.shape}")
        # logger.info(f"train_mask shape: {train_mask.shape}")

        # Initialize BNN if needed
        if self.bnn_model is None:
            processed_dim = X_processed.shape[1]
            # logger.info(f"Initializing BNN with input_dim: {processed_dim}")

            self.bnn_model = BNN(
                input_dim=processed_dim,  # 使用处理后的特征维度
                config=self.config.bnn_params,
                use_mask=True
            ).to(device)

            # 检查网络第一层的输入维度
            first_layer = self.bnn_model.network[0]

        if stage == 'fine_tune':
            training_params = self.config.training_params.copy()
            training_params.update({
                'bnn_epochs': self.config.sto_training['fine_tune_epochs'],
                'batch_size': max(16, min(64, len(y) // 2)),
                'patience': self.config.sto_training['fine_tune_patience'],
                'min_epochs': self.config.sto_training['fine_tune_min_epochs'],
                'warmup_epochs': self.config.sto_training['fine_tune_warmup_epochs'],
                'clip_grad_norm': 0.3,
                'min_delta_loss': 5e-9,
                'min_delta_r2': 5e-8,
                # 添加自适应Huber loss参数
                'initial_huber_delta': self.config.sto_training.get('fine_tune_initial_huber_delta', 1.0),
                'final_huber_delta': self.config.sto_training.get('fine_tune_final_huber_delta', 0.3),
                # Uncertainty pruning parameters for fine-tune
                'uncertainty_pruning': self.config.sto_training.get('fine_tune_uncertainty_pruning', True),
                'uncertainty_threshold_percentile': self.config.sto_training.get(
                    'fine_tune_uncertainty_threshold_percentile', 80),
                'uncertainty_pruning_min_samples': self.config.sto_training.get(
                    'fine_tune_uncertainty_pruning_min_samples', 100),
                'uncertainty_pruning_frequency': self.config.sto_training.get('fine_tune_uncertainty_pruning_frequency',
                                                                              25),
            })
            self._current_sample_size = len(y)
        else:
            training_params = self.config.training_params
            training_params.update({
                'initial_huber_delta': training_params.get('initial_huber_delta', 1.0),
                'final_huber_delta': training_params.get('final_huber_delta', 0.3),
            })

        # 计算残差统计
        xgb_pred = self.xgb_model.predict(X_raw)
        residuals = y - xgb_pred
        residual_q25, residual_q75 = np.percentile(residuals, [25, 75])
        iqr = residual_q75 - residual_q25
        lower_bound = residual_q25 - 3 * iqr
        upper_bound = residual_q75 + 3 * iqr
        residuals_clipped = np.clip(residuals, lower_bound, upper_bound)
        residual_median = np.median(residuals_clipped)
        residual_mad = np.median(np.abs(residuals_clipped - residual_median))
        residual_robust_std = max(residual_mad * 1.4826, 1e-6)
        self.residual_stats = {'mean': residual_median, 'std': residual_robust_std}
        normalized_residuals = (residuals_clipped - residual_median) / residual_robust_std

        if np.any(np.isnan(normalized_residuals)) or np.any(np.isinf(normalized_residuals)):
            logger.warning("Normalized residuals contain NaN/inf, using robust fallback")
            normalized_residuals = np.clip(normalized_residuals, -10, 10)

        # 初始化BNN模型
        if self.bnn_model is None:
            self.bnn_model = BNN(input_dim=X_processed.shape[1], config=self.config.bnn_params).to(device)

        # 配置优化器和调度器
        if stage == 'fine_tune':
            self._reset_early_stopping()
            base_lr = training_params['learning_rate']
            sample_ratio = len(y) / getattr(self, '_pretrain_sample_size', len(y))
            if sample_ratio < 0.05:
                lr_scale = 0.005
            elif sample_ratio < 0.1:
                lr_scale = 0.01
            elif sample_ratio < 0.3:
                lr_scale = 0.03
            else:
                lr_scale = 0.05
            initial_lr = base_lr * lr_scale
            weight_decay = training_params['weight_decay'] * 2.0
            trainable_params = [p for p in self.bnn_model.parameters() if p.requires_grad]
            optimizer = optim.AdamW(
                trainable_params,
                lr=initial_lr,
                weight_decay=weight_decay,
                betas=(0.85, 0.98),
                eps=1e-7,
                amsgrad=False
            )
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=training_params['warmup_epochs']
            )
        else:
            initial_lr = training_params['learning_rate']
            optimizer = optim.AdamW(
                self.bnn_model.parameters(),
                lr=initial_lr,
                weight_decay=training_params['weight_decay'],
                betas=(0.9, 0.999),
                eps=1e-8,
                amsgrad=True  # Modified: amsgrad was True by default, kept as True
            )
            warmup_scheduler = None

        if stage == 'pretrain':
            self.pretrain_optimizer_state = optimizer.state_dict()

        if stage == 'fine_tune':
            scheduler_main = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.7,
                patience=30,
                min_lr=initial_lr * 0.01,
                threshold=1e-9,
                threshold_mode='abs',
                cooldown=15,
            )
            scheduler_backup = None
        else:
            scheduler_main = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.8, patience=10, min_lr=1e-7,
                threshold=1e-6, threshold_mode='abs', cooldown=2
            )
            scheduler_backup = None

        # 修改数据准备部分，包含掩码
        if sample_weights is not None:
            weight_tensor = torch.FloatTensor(sample_weights)
            full_dataset = TensorDataset(
                torch.FloatTensor(X_processed),
                torch.FloatTensor(normalized_residuals),
                torch.FloatTensor(train_mask),  # 添加掩码
                weight_tensor
            )
        else:
            full_dataset = TensorDataset(
                torch.FloatTensor(X_processed),
                torch.FloatTensor(normalized_residuals),
                torch.FloatTensor(train_mask)  # 添加掩码
            )

        # Updated: Use Huber loss instead of MSE loss
        huber_delta = training_params.get('huber_delta', 1.0)
        criterion = nn.SmoothL1Loss(reduction='none', beta=huber_delta)

        # 准备验证数据 - 修正：确保包含掩码
        X_tensor_train = torch.FloatTensor(X_processed).to(device)
        mask_tensor_train = torch.FloatTensor(train_mask).to(device)  # 添加掩码张量

        if torch.any(torch.isnan(X_tensor_train)) or torch.any(torch.isinf(X_tensor_train)):
            logger.warning("Input tensor contains NaN/inf values, applying robust preprocessing")
            X_tensor_train = torch.nan_to_num(X_tensor_train, nan=0.0, posinf=1e6, neginf=-1e6)
            for i in range(X_tensor_train.shape[1]):
                feature_col = X_tensor_train[:, i]
                q25, q75 = torch.quantile(feature_col, torch.tensor([0.25, 0.75], device=device))
                iqr = q75 - q25
                if iqr > 1e-6:
                    feature_col = torch.clamp(feature_col, q25 - 3 * iqr, q75 + 3 * iqr)
                    X_tensor_train[:, i] = feature_col

        # 输出控制
        output_config = self.config.output_control
        mode = output_config['mode']
        show_progress_bar = output_config['show_progress_bar'] and mode != 'silent'
        show_epoch_details = output_config['show_epoch_details'] and mode != 'silent'
        show_parameter_info = output_config['show_parameter_info'] and mode != 'silent'
        show_early_stopping_info = output_config['show_early_stopping_info'] and mode != 'silent'
        show_final_summary = output_config['show_final_summary'] and mode != 'silent'
        verbose_epoch = training_params['verbose_epoch']
        total_epochs = training_params['bnn_epochs']

        # Sequence data loader
        sequence_dataloader = None
        sequence_loss_enabled = self.config.sequence_loss.get('enabled', False)
        if sequence_loss_enabled:
            try:
                sequence_dataset = OrdinalDataset(
                    self.processor,
                    self.config.sequence_loss['good_data_path'],
                    self.config.sequence_loss['bad_data_path'],
                    self.base_features  # 传入base_features
                )
                if len(sequence_dataset) > 0:
                    sequence_dataloader = DataLoader(
                        sequence_dataset,
                        batch_size=self.config.sequence_loss.get('sequence_batch_size', 32),
                        shuffle=True,
                        pin_memory=True
                    )
                    logger.info(f"Sequence contrast loss enabled, loaded {len(sequence_dataset)} sequences")
                else:
                    logger.warning("Sequence dataset empty, disabling contrast loss")
                    sequence_loss_enabled = False
            except Exception as e:
                logger.error(f"Sequence data loading failed: {str(e)}")
                sequence_loss_enabled = False

        # 记录序列损失启用状态到history
        self.history['sequence_loss_enabled'] = sequence_loss_enabled

        # if show_parameter_info:
        # logger.info("=" * 60)
        # logger.info(f"Starting BNN training ({stage} stage)")
        # logger.info("=" * 60)
        # logger.info(f"Training parameters:")
        # logger.info(f"  Samples: {len(y)} (STO: {np.sum(sto_flags) if sto_flags is not None else 'N/A'})")
        # logger.info(f"  Total epochs: {total_epochs}")
        # logger.info(f"  Batch size: {training_params['batch_size']}")
        # logger.info(f"  Learning rate: {initial_lr:.2e}")
        # logger.info(f"  Early stopping patience: {training_params['patience']}")
        #             if sequence_loss_enabled:
        #                 logger.info(f"  Sequence contrast loss enabled")
        #             logger.info("=" * 60)

        if show_progress_bar:
            epoch_iterator = tqdm(range(total_epochs), desc=f"Training BNN ({stage})", ncols=100)
        else:
            epoch_iterator = range(total_epochs)

        # Sequence data iterator
        sequence_iterator = None
        if sequence_loss_enabled:
            sequence_iterator = iter(sequence_dataloader)

        self._reset_early_stopping()
        early_stopped = False
        loss_history_short = []

        # --- Uncertainty Pruning Variables ---
        active_indices = np.arange(len(y))  # Initially, all indices are active
        dataloader = DataLoader(
            Subset(full_dataset, active_indices),  # Use Subset with active indices
            batch_size=training_params['batch_size'],
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            shuffle=True
        )
        # --- End Uncertainty Pruning Variables ---

        # 训练循环
        for epoch in epoch_iterator:
            # --- Uncertainty Pruning Logic ---
            if training_params.get('uncertainty_pruning', False) and \
                    epoch > 0 and epoch % training_params.get('uncertainty_pruning_frequency', 50) == 0:

                self.bnn_model.eval()
                with torch.no_grad():
                    # 获取当前活动样本的特征和掩码
                    current_active_X = X_processed[active_indices]
                    current_active_mask = train_mask[active_indices]  # 确保获取对应的掩码

                    current_active_X_tensor = torch.FloatTensor(current_active_X).to(device)
                    current_active_mask_tensor = torch.FloatTensor(current_active_mask).to(device)

                    # 关键修复：传递掩码参数
                    current_mc_predictions = self.bnn_model.mc_predict(
                        current_active_X_tensor,
                        mask=current_active_mask_tensor,  # 传递掩码
                        n_samples=30
                    )

                # 【修复】计算每个样本的不确定性（标准差），而不是直接使用MC预测结果
                if current_mc_predictions.ndim == 2:
                    # 形状为 (n_samples, n_mc)，计算每个样本的MC预测的标准差
                    current_uncertainties = np.std(current_mc_predictions, axis=1)
                else:
                    # 如果已经是单值，直接使用或转换为适当形状
                    logger.warning(f"Unexpected MC predictions shape: {current_mc_predictions.shape}")
                    if current_mc_predictions.ndim == 1:
                        current_uncertainties = current_mc_predictions
                    else:
                        # 尝试降维
                        current_uncertainties = current_mc_predictions.reshape(current_mc_predictions.shape[0], -1).std(
                            axis=1)

                # Map back to original dataset indices
                original_active_uncertainties = np.full(len(y), np.inf)  # Initialize with inf for inactive
                original_active_uncertainties[active_indices] = current_uncertainties

                # Perform filtering on the *full* original dataset indices
                current_threshold = np.percentile(current_uncertainties,
                                                  training_params.get('uncertainty_threshold_percentile', 85))
                new_active_mask = original_active_uncertainties < current_threshold

                # Ensure minimum sample size from the *original* dataset
                min_samples = training_params.get('uncertainty_pruning_min_samples', 200)
                if np.sum(new_active_mask) < min_samples:
                    # logger.info(f"Number of active samples ({np.sum(new_active_mask)}) below minimum ({min_samples}). Adjusting.")
                    # Get the indices of the *original* dataset that are currently active
                    current_active_orig_indices = active_indices
                    # Get the uncertainties for *current* active samples
                    current_active_uncs = original_active_uncertainties[current_active_orig_indices]
                    # Find the indices of the min_samples_to_keep lowest uncertainty samples within the current active set
                    sorted_current_active_idx = np.argsort(current_active_uncs)[:min_samples]
                    # Get the corresponding *original* indices
                    final_active_orig_indices = current_active_orig_indices[sorted_current_active_idx]
                    # Create new mask based on original dataset size
                    new_active_mask = np.zeros(len(y), dtype=bool)
                    new_active_mask[final_active_orig_indices] = True

                # Update active_indices
                new_active_indices = np.where(new_active_mask)[0]
                num_pruned = len(active_indices) - len(new_active_indices)
                # logger.info(f"Epoch {epoch}: Pruned {num_pruned} samples. Active samples: {len(new_active_indices)}.")

                if len(new_active_indices) != len(active_indices):
                    active_indices = new_active_indices
                    # Recreate dataloader with new active indices
                    dataloader = DataLoader(
                        Subset(full_dataset, active_indices),
                        batch_size=training_params['batch_size'],
                        num_workers=0,
                        pin_memory=torch.cuda.is_available(),
                        shuffle=True
                    )
                else:
                    # logger.info(f"Epoch {epoch}: No new samples pruned.")
                    pass
            # --- End Uncertainty Pruning Logic ---

            # 每个epoch计算自适应的Huber delta
            initial_delta = training_params['initial_huber_delta']
            final_delta = training_params['final_huber_delta']
            current_delta = self.adaptive_huber_delta(epoch, total_epochs, initial_delta, final_delta)
            # 创建当前epoch的Huber损失函数
            criterion = nn.SmoothL1Loss(beta=current_delta, reduction='none')

            self.bnn_model.train()
            epoch_loss = 0.0
            sequence_loss_total = 0.0
            intra_good_loss_total = 0.0
            intra_bad_loss_total = 0.0
            inter_loss_total = 0.0
            num_samples = 0

            if stage == 'fine_tune':
                accumulation_steps = min(
                    self.config.sto_training['gradient_accumulation_steps'],
                    max(4, len(y) // 4)
                )
            else:
                accumulation_steps = 2

            batch_losses = []
            valid_batch_count = 0
            for batch_idx, batch in enumerate(dataloader):
                if len(batch) == 4:  # 有权重的情况
                    inputs, targets, batch_mask, batch_weights = batch
                    batch_weights = batch_weights.to(device)
                else:  # 没有权重的情况
                    inputs, targets, batch_mask = batch
                    batch_weights = None

                inputs = inputs.to(device)
                targets = targets.to(device)
                batch_mask = batch_mask.to(device)  # 掩码转移到设备

                outputs = self.bnn_model(inputs, batch_mask)
                if torch.any(torch.isnan(outputs)) or torch.any(torch.isinf(outputs)):
                    logger.warning(f"Model outputs contain NaN/inf at epoch {epoch}, batch {batch_idx}")
                    for param in self.bnn_model.parameters():
                        if torch.any(torch.isnan(param)) or torch.any(torch.isinf(param)):
                            param.data.normal_(0, 0.01)
                    continue

                # 使用当前epoch的自适应Huber loss
                loss_per_sample = criterion(outputs.squeeze(), targets)
                if torch.any(torch.isnan(loss_per_sample)) or torch.any(torch.isinf(loss_per_sample)):
                    logger.warning(f"Loss contains NaN/inf at epoch {epoch}, batch {batch_idx}")
                    continue

                if batch_weights is not None:
                    # Note: batch_weights now corresponds to the active subset
                    loss = (loss_per_sample * batch_weights).mean()
                else:
                    loss = loss_per_sample.mean()

                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e6:
                    logger.warning(f"Abnormal loss detected: {loss.item()}")
                    continue

                batch_losses.append(loss.item())
                if stage == 'fine_tune':
                    l2_reg = sum(torch.norm(p, 2) for p in self.bnn_model.parameters() if p.requires_grad)
                    if not (torch.isnan(l2_reg) or torch.isinf(l2_reg)):
                        reg_strength_l2 = 5e-6
                        reg_loss = reg_strength_l2 * l2_reg
                        if not (torch.isnan(reg_loss) or torch.isinf(reg_loss)):
                            loss = loss + reg_loss

                    if (hasattr(self, 'pretrain_model_state') and
                            epoch > training_params.get('warmup_epochs', 80)):
                        weight_change_penalty = 0
                        current_params = dict(self.bnn_model.named_parameters())
                        for name, param in current_params.items():
                            if param.requires_grad and name in self.pretrain_model_state:
                                pretrain_param = self.pretrain_model_state[name]
                                if pretrain_param.shape == param.shape:
                                    weight_change = torch.norm(param - pretrain_param.to(param.device), 2)
                                    if not (torch.isnan(weight_change) or torch.isinf(weight_change)):
                                        weight_change_penalty += weight_change
                        penalty_strength = 5e-6 * (100.0 / max(50, len(y)))
                        if weight_change_penalty > 0 and not (
                                torch.isnan(weight_change_penalty) or torch.isinf(weight_change_penalty)):
                            penalty = penalty_strength * weight_change_penalty
                            if not (torch.isnan(penalty) or torch.isinf(penalty)):
                                loss = loss + penalty
                    if sequence_loss_enabled and sequence_iterator is not None:
                        try:
                            sequence_batch = next(sequence_iterator)
                        except StopIteration:
                            sequence_iterator = iter(sequence_dataloader)
                            sequence_batch = next(sequence_iterator)

                        seq_inputs, seq_labels = sequence_batch
                        seq_inputs, seq_labels = seq_inputs.to(device), seq_labels.to(device)

                        seq_loss, intra_good_l, intra_bad_l, inter_l = sequence_ranking_loss(
                            self.bnn_model, self.residual_stats, seq_inputs, seq_labels, self.config
                        )

                        # Fixed: Use current loss instead of avg_loss (which isn't calculated yet)
                        # Adaptive weight based on current batch loss
                        current_loss_value = loss.item() if isinstance(loss, torch.Tensor) else loss
                        seq_loss_weight = min(0.5, max(0.1, 10.0 / (current_loss_value + 1e-8)))
                        loss = loss + seq_loss_weight * seq_loss

                        # Accumulate sequence losses for logging
                        sequence_loss_total += seq_loss.item()
                        intra_good_loss_total += intra_good_l.item()
                        intra_bad_loss_total += intra_bad_l.item()
                        inter_loss_total += inter_l.item()

                loss = loss / accumulation_steps
                loss.backward()
                epoch_loss += loss.item() * inputs.size(0) * accumulation_steps
                num_samples += inputs.size(0)
                valid_batch_count += 1

                if (batch_idx + 1) % accumulation_steps == 0:
                    grad_norm = 0.2 if stage == 'fine_tune' else 0.5
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in self.bnn_model.parameters() if p.requires_grad], grad_norm
                    )
                    if torch.isnan(total_norm) or torch.isinf(total_norm) or total_norm > 50:
                        logger.warning(f"Large gradient norm detected: {total_norm}")
                        for param in self.bnn_model.parameters():
                            if param.requires_grad and param.grad is not None:
                                param.grad.data.clamp_(-1.0, 1.0)

                    optimizer.step()
                    optimizer.zero_grad()

            if (batch_idx + 1) % accumulation_steps != 0:
                grad_norm = 0.2 if stage == 'fine_tune' else 0.5
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.bnn_model.parameters() if p.requires_grad], grad_norm
                )
                optimizer.step()
                optimizer.zero_grad()

            if num_samples == 0:
                logger.warning(f"No valid samples processed in epoch {epoch}")
                continue
            if valid_batch_count == 0:
                logger.warning(f"No valid batches processed in epoch {epoch}")
                continue

            avg_loss = epoch_loss / num_samples
            # 计算平均序列损失
            avg_sequence_loss = sequence_loss_total / valid_batch_count if valid_batch_count > 0 else 0.0
            avg_intra_good_loss = intra_good_loss_total / valid_batch_count if valid_batch_count > 0 else 0.0
            avg_intra_bad_loss = intra_bad_loss_total / valid_batch_count if valid_batch_count > 0 else 0.0
            avg_inter_loss = inter_loss_total / valid_batch_count if valid_batch_count > 0 else 0.0

            # 增强的loss历史管理
            loss_history_short.append(avg_loss)
            if len(loss_history_short) > 30:  # 增加历史长度
                loss_history_short.pop(0)

            # Loss spike detection and recovery
            if len(loss_history_short) >= 10 and stage == 'fine_tune':
                recent_losses = loss_history_short[-5:]
                prev_losses = loss_history_short[-10:-5]
                recent_avg = np.mean(recent_losses)
                prev_avg = np.mean(prev_losses)
                if recent_avg > prev_avg * 3 and avg_loss > 0.5:
                    logger.warning(f"Loss spike detected at epoch {epoch}, applying recovery")
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.3
                    if self.best_model_state is not None:
                        self.bnn_model.load_state_dict(self.best_model_state)

            # Evaluate on training set (only active samples contribute to metrics here)
            self.bnn_model.eval()
            with torch.no_grad():
                # Use only the currently active samples for R2 calculation
                active_X_tensor = X_tensor_train[active_indices]  # Use active indices
                active_mask = mask_tensor_train[active_indices]  # 获取对应的掩码
                bnn_output = self.bnn_model(active_X_tensor, active_mask).squeeze()
                if torch.any(torch.isnan(bnn_output)) or torch.any(torch.isinf(bnn_output)):
                    logger.warning(f"BNN output contains NaN/inf at epoch {epoch}")
                    train_r2 = -1.0
                else:
                    active_xgb_pred = torch.tensor(xgb_pred[active_indices], device=device)  # Use active indices
                    active_y = torch.tensor(y[active_indices], dtype=torch.float32,
                                            device=device)  # Use active indices
                    active_sample_weights = None
                    if sample_weights is not None:
                        active_sample_weights = torch.tensor(sample_weights[active_indices], dtype=torch.float32,
                                                             device=device)  # Use active indices

                    bnn_pred = bnn_output * self.residual_stats['std'] + self.residual_stats['mean']
                    hybrid_pred = active_xgb_pred + bnn_pred
                    train_r2 = self.weighted_r2_score_tensor(active_y, hybrid_pred, active_sample_weights)
                    if np.isnan(train_r2) or np.isinf(train_r2):
                        train_r2 = -1.0

            # Update scheduler
            if stage == 'fine_tune':
                if warmup_scheduler is not None and epoch < training_params['warmup_epochs']:
                    warmup_scheduler.step()
                else:
                    if scheduler_main is not None:
                        scheduler_main.step(avg_loss)
            else:
                if warmup_scheduler is not None and epoch < training_params['warmup_epochs']:
                    warmup_scheduler.step()
                elif isinstance(scheduler_main, optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    scheduler_main.step()
                else:
                    scheduler_main.step(avg_loss)

            self.history['loss'].append(avg_loss)
            self.history['r2'].append(train_r2)
            self.history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            self.history['sequence_loss'].append(avg_sequence_loss)
            self.history['intra_good_loss'].append(avg_intra_good_loss)
            self.history['intra_bad_loss'].append(avg_intra_bad_loss)
            self.history['inter_loss'].append(avg_inter_loss)
            self.history['sequence_loss_enabled'] = sequence_loss_enabled

            if show_progress_bar:
                epoch_iterator.set_postfix({'loss': f'{avg_loss:.4f}', 'R2': f'{train_r2:.4f}',
                                            'lr': f"{optimizer.param_groups[0]['lr']:.2e}",
                                            'seq_loss': f'{avg_sequence_loss:.4f}'})  # 在进度条中显示序列损失

            # if show_epoch_details and (epoch % verbose_epoch == 0 or epoch == total_epochs - 1):
            # logger.info(f"Epoch {epoch + 1}/{total_epochs} - "
            #            f"Loss: {avg_loss:.6f}, R²: {train_r2:.6f}, "f"Seq Loss: {avg_sequence_loss:.6f}, "
            #            f"LR: {optimizer.param_groups[0]['lr']:.2e}, Active: {len(active_indices)}")

            if self._should_stop_early(avg_loss, train_r2, epoch, stage, len(y)):
                early_stopped = True
                # if show_early_stopping_info:
                # logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        if stage == 'pretrain':
            self._pretrain_sample_size = len(y)

        final_lr = optimizer.param_groups[0]['lr']
        self.last_lr = final_lr

        # if show_final_summary:
        # logger.info("=" * 60)
        # logger.info(f"BNN training ({stage} stage) completed")
        # logger.info("=" * 60)
        # logger.info(f"Final loss: {avg_loss:.6f}")
        # logger.info(f"Final R²: {train_r2:.6f}")
        # logger.info(f"Final sequence loss: {avg_sequence_loss:.6f}")  # 记录最终序列损失
        # logger.info(f"Final learning rate: {final_lr:.2e}")
        # if early_stopped:
        # logger.info("Training was early stopped")
        # logger.info("=" * 60)

        if self.best_model_state is not None:
            self.bnn_model.load_state_dict(self.best_model_state)

        if stage != 'fine_tune':  # Only store during pretrain or normal training
            try:
                # Calculate variance statistics on training data
                self.bnn_model.eval()
                X_tensor = torch.tensor(X_processed, dtype=torch.float32, device=device)
                residuals = self.bnn_model.mc_predict(X_tensor, n_samples=50)
                residuals = residuals * self.residual_stats['std'] + self.residual_stats['mean']
                residual_var = residuals.var(axis=1)
                self.bnn_var_mean = np.mean(residual_var)
                self.bnn_var_std = np.std(residual_var)
                # logger.info(f"BNN variance statistics - mean: {self.bnn_var_mean:.6f}, std: {self.bnn_var_std:.6f}")
            except Exception as e:
                logger.warning(f"Failed to calculate BNN variance statistics: {str(e)}")
                self.bnn_var_mean = None
                self.bnn_var_std = None

        return early_stopped

    @staticmethod
    def weighted_r2_score_tensor(y_true: torch.Tensor, y_pred: torch.Tensor,
                                 sample_weights: Optional[torch.Tensor] = None) -> float:
        """计算加权R²分数"""
        if sample_weights is not None and sample_weights.device != y_true.device:
            sample_weights = sample_weights.to(y_true.device)

        if sample_weights is not None:
            sum_weights = torch.sum(sample_weights)
            if sum_weights == 0:
                y_wmean = torch.mean(y_true)
            else:
                y_wmean = torch.sum(sample_weights * y_true) / sum_weights
        else:
            y_wmean = torch.mean(y_true)

        if sample_weights is not None:
            ss_res = torch.sum(sample_weights * (y_true - y_pred) ** 2)
            ss_tot = torch.sum(sample_weights * (y_true - y_wmean) ** 2)
            if ss_tot == 0:
                return 1.0 if ss_res == 0 else 0.0
        else:
            ss_res = torch.sum((y_true - y_pred) ** 2)
            ss_tot = torch.sum((y_true - y_wmean) ** 2)
            if ss_tot == 0:
                return 1.0 if ss_res == 0 else 0.0

        r2 = 1.0 - (ss_res / ss_tot)
        return r2.item()

    def _should_stop_early(self, current_loss: float, current_r2: float, epoch: int,
                           stage: str, sample_size: int) -> bool:
        """Early stopping逻辑（复用原有代码）"""
        training_params = self.config.training_params
        if stage == 'fine_tune':
            if sample_size < 50:
                warmup_epochs = 120
                min_epochs = 250
                patience = 120
                min_delta_r2 = 5e-8
                stability_window = 40
                stability_threshold = 0.0002
            elif sample_size < 100:
                warmup_epochs = 100
                min_epochs = 200
                patience = 100
                min_delta_r2 = 1e-7
                stability_window = 35
                stability_threshold = 0.0003
            elif sample_size < 200:
                warmup_epochs = 80
                min_epochs = 180
                patience = 80
                min_delta_r2 = 2e-7
                stability_window = 30
                stability_threshold = 0.0005
            else:
                warmup_epochs = self.config.sto_training['fine_tune_warmup_epochs']
                min_epochs = self.config.sto_training['fine_tune_min_epochs']
                patience = self.config.sto_training['fine_tune_patience']
                min_delta_r2 = self.config.sto_training['fine_tune_min_delta_r2']
                stability_window = self.config.sto_training['fine_tune_stability_window']
                stability_threshold = self.config.sto_training['fine_tune_stability_threshold']
        else:
            warmup_epochs = training_params['warmup_epochs']
            min_epochs = training_params['min_epochs']
            patience = training_params['patience']
            min_delta_loss = training_params['min_delta_loss']
            min_delta_r2 = training_params['min_delta_r2']
            stability_window = training_params['stability_window']
            stability_threshold = training_params['stability_threshold']

        if epoch < warmup_epochs or epoch < min_epochs:
            return False

        if stage == 'fine_tune':
            self.r2_history.append(current_r2)
            max_history = stability_window * 6
            if len(self.r2_history) > max_history:
                self.r2_history = self.r2_history[-max_history:]
        else:
            self.loss_history.append(current_loss)
            self.r2_history.append(current_r2)
            max_history = stability_window * 6
            if len(self.loss_history) > max_history:
                self.loss_history = self.loss_history[-max_history:]
            if len(self.r2_history) > max_history:
                self.r2_history = self.r2_history[-max_history:]

        if stage == 'fine_tune':
            r2_improved = current_r2 > (self.best_r2 + min_delta_r2)
            if r2_improved:
                self.best_r2 = current_r2
                self.best_model_state = self.bnn_model.state_dict().copy()
                self.patience_counter = 0
                self.epochs_since_improvement = 0
            else:
                self.patience_counter += 1
                self.epochs_since_improvement += 1
        else:
            r2_improved = current_r2 > (self.best_r2 + min_delta_r2)
            loss_improved = current_loss < (self.best_loss - min_delta_loss)
            if r2_improved or loss_improved:
                if r2_improved:
                    self.best_r2 = current_r2
                if loss_improved:
                    self.best_loss = current_loss
                    self.best_model_state = self.bnn_model.state_dict().copy()
                self.patience_counter = 0
                self.epochs_since_improvement = 0
            else:
                self.patience_counter += 1
                self.epochs_since_improvement += 1

        if stage == 'fine_tune' and len(self.r2_history) >= stability_window:
            recent_r2s = self.r2_history[-stability_window:]
            r2_cv = np.std(recent_r2s) / (np.abs(np.mean(recent_r2s)) + 1e-8)
            r2_trend = np.polyfit(range(len(recent_r2s)), recent_r2s, 1)[0]
            is_overfitting = (r2_trend < -5e-7 and
                              self.epochs_since_improvement > stability_window)
            if is_overfitting:
                self.patience_counter += min(10, sample_size // 5)
            convergence_threshold = stability_threshold * (50.0 / max(30, sample_size))
            is_converged = (r2_cv < convergence_threshold and
                            abs(r2_trend) < 1e-7 and
                            self.epochs_since_improvement > stability_window * 2)
            if is_converged:
                self.patience_counter += 5
        elif stage != 'fine_tune' and len(self.loss_history) >= stability_window:
            recent_losses = self.loss_history[-stability_window:]
            recent_r2s = self.r2_history[-stability_window:]
            loss_cv = np.std(recent_losses) / (np.abs(np.mean(recent_losses)) + 1e-8)
            r2_cv = np.std(recent_r2s) / (np.abs(np.mean(recent_r2s)) + 1e-8)
            loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
            r2_trend = np.polyfit(range(len(recent_r2s)), recent_r2s, 1)[0]

            is_overfitting = (loss_trend > 5e-7 and r2_trend < -5e-7 and
                              self.epochs_since_improvement > stability_window)
            if is_overfitting:
                self.patience_counter += min(10, sample_size // 5)

            convergence_threshold = stability_threshold * (50.0 / max(30, sample_size))
            is_converged = (loss_cv < convergence_threshold and r2_cv < convergence_threshold and
                            abs(loss_trend) < 1e-7 and abs(r2_trend) < 1e-7 and
                            self.epochs_since_improvement > stability_window * 2)
            if is_converged:
                self.patience_counter += 5

        dynamic_patience = patience
        if stage == 'fine_tune':
            if sample_size < 30:
                dynamic_patience = int(patience * 3.0)
            elif sample_size < 50:
                dynamic_patience = int(patience * 2.5)
            elif sample_size < 100:
                dynamic_patience = int(patience * 2.0)
            elif sample_size < 200:
                dynamic_patience = int(patience * 1.5)

        return (self.patience_counter >= dynamic_patience and epoch >= min_epochs)

    def predict(self, X: pd.DataFrame, n_samples: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测方法 - 修复掩码处理
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        # 提取特征
        X_features = X[self.base_features]

        # XGBoost预测
        X_raw = X_features.values
        xgb_pred = self.xgb_model.predict(X_raw)

        # BNN预测 - 确保正确获取掩码
        try:
            # 获取处理后的特征和掩码
            X_processed, mask = self.processor.transform(X[self.base_features], return_mask=True)

            # 调试信息
            # logger.info(f"X_processed shape: {X_processed.shape}")
            # logger.info(f"Mask shape: {mask.shape}")

            # 确保掩码是浮点类型且与输入形状匹配
            mask = mask.astype(np.float32)
            if mask.shape != X_processed.shape:
                logger.warning(
                    f"Mask shape {mask.shape} doesn't match input shape {X_processed.shape}. Creating default mask.")
                mask = np.ones_like(X_processed, dtype=np.float32)

        except Exception as e:
            logger.warning(f"Error getting mask from processor: {e}. Using default mask.")
            X_processed = self.processor.transform(X[self.base_features])
            mask = np.ones_like(X_processed, dtype=np.float32)

        # 转换为张量
        X_tensor = torch.tensor(X_processed, dtype=torch.float32, device=device)
        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device)

        # 数据清理
        if torch.any(torch.isnan(X_tensor)) or torch.any(torch.isinf(X_tensor)):
            logger.warning("Input tensor contains NaN/inf values")
            X_tensor = torch.nan_to_num(X_tensor, nan=0.0, posinf=1e6, neginf=-1e6)

        # BNN预测 - 传递掩码
        residuals = self.bnn_model.mc_predict(X_tensor, mask=mask_tensor, n_samples=n_samples)
        if np.any(np.isnan(residuals)) or np.any(np.isinf(residuals)):
            logger.warning("BNN residual predictions contain NaN/inf values")

        # 反标准化残差
        residual_std = self.residual_stats.get('std', 1.0)
        residual_mean = self.residual_stats.get('mean', 0.0)
        if np.isnan(residual_std) or np.isinf(residual_std) or residual_std == 0:
            residual_std = 1.0
        if np.isnan(residual_mean) or np.isinf(residual_mean):
            residual_mean = 0.0
        residuals = residuals * residual_std + residual_mean

        # 计算统计量
        residual_mean = residuals.mean(axis=1)
        residual_var = residuals.var(axis=1)
        if np.any(np.isnan(residual_mean)) or np.any(np.isinf(residual_mean)):
            logger.warning("Residual mean contains NaN/inf values")
            residual_mean = np.nan_to_num(residual_mean, nan=0.0)
        if np.any(np.isnan(residual_var)) or np.any(np.isinf(residual_var)):
            logger.warning("Residual variance contains NaN/inf values")
            residual_var = np.nan_to_num(residual_var, nan=1.0)

        # Dynamic weighting based on uncertainty
        bnn_var_mean = getattr(self, 'bnn_var_mean', None)
        bnn_var_std = getattr(self, 'bnn_var_std', None)

        if bnn_var_mean is not None and bnn_var_std is not None and bnn_var_std > 1e-8:
            # Normalize variance and apply sigmoid weighting
            normalized_var = (residual_var - bnn_var_mean) / (bnn_var_std + 1e-8)
            weights = 1 / (1 + np.exp(np.clip(normalized_var, -10, 10)))  # Clip to avoid overflow
        else:
            # Fallback: simple inverse variance weighting
            weights = 1 / (1 + residual_var)

        # Apply weighting to residual mean
        weighted_residual_mean = residual_mean * weights

        # Combine predictions
        try:
            hybrid_pred = xgb_pred + weighted_residual_mean
            if np.any(np.isnan(hybrid_pred)) or np.any(np.isinf(hybrid_pred)):
                logger.warning("Final hybrid predictions contain NaN/inf values")
                hybrid_pred = np.nan_to_num(hybrid_pred, nan=xgb_pred)
        except Exception as e:
            logger.error(f"Final prediction combination failed: {str(e)}")
            hybrid_pred = xgb_pred

        residual_std_out = np.sqrt(residual_var)  # standard deviation of residuals for each instance

        return hybrid_pred, residual_std_out

    def diagnose_model_performance(self, X_train, y_train, X_val, y_val):
        """诊断模型性能问题"""
        # logger.info("=" * 60)
        # logger.info("开始模型性能诊断")
        # logger.info("=" * 60)

        # 1. 检查训练集和验证集的分布差异
        train_pred, train_unc = self.predict(X_train)
        val_pred, val_unc = self.predict(X_val)

        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)

        # logger.info(f"训练集 R²: {train_r2:.4f}")
        # logger.info(f"验证集 R²: {val_r2:.4f}")
        # logger.info(f"过拟合程度: {train_r2 - val_r2:.4f}")

        # 2. 检查权重分布
        X_train_raw = X_train[self.base_features].values
        X_val_raw = X_val[self.base_features].values
        train_weights = calculate_boundary_penalty_weights(
            X_train_raw, self.base_features,
            fixed_covariance_inv=self.fixed_covariance_inv,
            weight_threshold=0.0  # Use 0.0 to get raw weights
        )
        val_weights = calculate_boundary_penalty_weights(
            X_val_raw, self.base_features,
            fixed_covariance_inv=self.fixed_covariance_inv,
            weight_threshold=0.0  # Use 0.0 to get raw weights
        )

        # logger.info(f"权重分布诊断:")
        # logger.info(f"训练集权重 - 均值: {train_weights.mean():.4f}, "
        # f"中位数: {np.median(train_weights):.4f}, "
        # f"低于0.1: {(train_weights < 0.1).sum()}/{len(train_weights)}")
        # logger.info(f"验证集权重 - 均值: {val_weights.mean():.4f}, "
        # f"中位数: {np.median(val_weights):.4f}, "
        # f"低于0.1: {(val_weights < 0.1).sum()}/{len(val_weights)}")

        # 3. 检查预测不确定性
        # logger.info(f"不确定性诊断:")
        # logger.info(f"训练集不确定性 - 均值: {train_unc.mean():.4f}, "
        # f"中位数: {np.median(train_unc):.4f}")
        # logger.info(f"验证集不确定性 - 均值: {val_unc.mean():.4f}, "
        # f"中位数: {np.median(val_unc):.4f}")

        # 4. 特征重要性检查
        # if hasattr(self.xgb_model, 'feature_importances_'):
        # importance = self.xgb_model.feature_importances_
        # logger.info(f"XGBoost特征重要性:")
        # for feat, imp in zip(self.base_features, importance):
        # logger.info(f"  {feat}: {imp:.4f}")

        # logger.info("=" * 60)

        return {
            'train_r2': train_r2,
            'val_r2': val_r2,
            'overfit_degree': train_r2 - val_r2,
            'train_weight_stats': {
                'mean': train_weights.mean(),
                'median': np.median(train_weights),
                'low_weight_ratio': (train_weights < 0.1).sum() / len(train_weights)
            },
            'val_weight_stats': {
                'mean': val_weights.mean(),
                'median': np.median(val_weights),
                'low_weight_ratio': (val_weights < 0.1).sum() / len(val_weights)
            }
        }

    def save_model(self, path: str):
        """保存模型"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # 保存XGBoost模型和参数
        self.xgb_model.save_model(str(path / 'xgb_model.xgb'))
        # 额外保存XGBoost参数
        xgb_params = self.xgb_model.get_params()
        joblib.dump(xgb_params, str(path / 'xgb_params.joblib'))

        # 保存BNN
        if self.bnn_model is not None:
            torch.save(self.bnn_model.state_dict(), str(path / 'bnn_model.pth'))

        # 保存processor
        if self.processor is not None:
            joblib.dump(self.processor, str(path / 'processor.joblib'))

        # 保存固定协方差矩阵
        if self.fixed_covariance_inv is not None:
            joblib.dump(self.fixed_covariance_inv, str(path / 'fixed_covariance_inv.joblib'))

        # Save cluster calculator if it exists
        if self.cluster_calculator is not None:
            joblib.dump(self.cluster_calculator, str(path / 'cluster_calculator.joblib'))

        # 保存模型参数
        joblib.dump({
            'config': self.config,
            'residual_stats': self.residual_stats,
            'input_dim': self.input_dim,
            'is_fitted': self.is_fitted,
            'base_features': self.base_features,
            'bnn_var_mean': getattr(self, 'bnn_var_mean', None),  # Add this
            'bnn_var_std': getattr(self, 'bnn_var_std', None)  # Add this
        }, str(path / 'model_params.joblib'))

        # logger.info(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path: str) -> 'HybridModel':
        """加载模型"""
        path = Path(path)

        # 加载参数
        params = joblib.load(str(path / 'model_params.joblib'))

        # 创建模型实例
        model = cls(params['base_features'], params['config'])
        model.residual_stats = params['residual_stats']
        model.is_fitted = params['is_fitted']
        model.base_features = params.get('base_features', [])
        model.bnn_var_mean = params.get('bnn_var_mean', None)
        model.bnn_var_std = params.get('bnn_var_std', None)

        # 加载XGBoost
        model.xgb_model = xgb.XGBRegressor()
        # 先加载保存的参数
        if (path / 'xgb_params.joblib').exists():
            saved_params = joblib.load(str(path / 'xgb_params.joblib'))
            model.xgb_model.set_params(**saved_params)
        # 然后加载模型
        model.xgb_model.load_model(str(path / 'xgb_model.xgb'))

        # 验证参数
        loaded_params = model.xgb_model.get_params()
        for key in ['learning_rate', 'n_estimators', 'max_depth']:
            if loaded_params.get(key) is None:
                logger.warning(f"Loaded XGBoost parameter {key} is still None after loading saved params")

        # 加载processor
        if (path / 'processor.joblib').exists():
            model.processor = joblib.load(str(path / 'processor.joblib'))

        # 加载固定协方差矩阵
        if (path / 'fixed_covariance_inv.joblib').exists():
            model.fixed_covariance_inv = joblib.load(str(path / 'fixed_covariance_inv.joblib'))

        # 加载BNN
        if (path / 'bnn_model.pth').exists():
            processed_dim = len(model.processor.selected_features) if model.processor else params['input_dim']
            model.bnn_model = BNN(input_dim=processed_dim, config=params['config'].bnn_params)
            model.bnn_model.load_state_dict(torch.load(str(path / 'bnn_model.pth'), map_location=device))
            model.bnn_model.to(device)

        # Load cluster calculator if it exists
        if (path / 'cluster_calculator.joblib').exists():
            model.cluster_calculator = joblib.load(str(path / 'cluster_calculator.joblib'))
        else:
            model.cluster_calculator = None

        # logger.info(f"Model loaded from {path}")
        return model


def create_experiment_pipeline():
    """创建实验流水线"""

    def run_experiment(
            data_path: str,
            base_features: List[str],
            target_column: str,
            sto_column: str = 'Substrate',
            config: Optional[ModelConfig] = None,
            output_dir: str = "./experiment_results",
            val_size: float = 0.4,
            random_state: int = 42,
            sequence_good_path: str = "",
            sequence_bad_path: str = ""
    ):
        """
        运行完整实验流程，包含全数据集+增强数据的评估
        """
        setup_seed(random_state)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        config = config or ModelConfig()
        config.output_control['target_column'] = target_column

        # Configure sequence loss
        if sequence_good_path and sequence_bad_path and os.path.exists(sequence_good_path) and os.path.exists(
                sequence_bad_path):
            config.sequence_loss['enabled'] = True
            config.sequence_loss['good_data_path'] = sequence_good_path
            config.sequence_loss['bad_data_path'] = sequence_bad_path
            logger.info("Sequence contrast loss enabled")
        else:
            config.sequence_loss['enabled'] = False
            logger.info("Sequence contrast loss disabled - paths not found or empty")

        # 清空并创建输出目录
        import shutil
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. 读取数据
        data = pd.read_excel(data_path)

        # 2. 数据分割
        if sto_column in data.columns:
            data['STO_binary'] = (data[sto_column] == 'STO').astype(int)
            binary_distribution = data['STO_binary'].value_counts()
            stratify = data['STO_binary'] if binary_distribution.min() >= 2 else None
        else:
            stratify = None

        train_data, val_data = train_test_split(
            data,
            test_size=val_size,
            random_state=random_state,
            stratify=stratify
        )

        # 清理临时列
        if 'STO_binary' in train_data.columns:
            train_data = train_data.drop('STO_binary', axis=1)
        if 'STO_binary' in val_data.columns:
            val_data = val_data.drop('STO_binary', axis=1)

        # 3. 准备训练和验证数据
        X_train = train_data[base_features + [sto_column]].copy()
        y_train = train_data[target_column]
        X_val = val_data[base_features + [sto_column]].copy()
        y_val = val_data[target_column]

        # 4. 训练模型
        model = HybridModel(base_features=base_features, config=config)
        model.fit(X_train, y_train, val_X=X_val, val_y=y_val, sto_column=sto_column)

        # 5. 诊断性能
        diagnostic_results = model.diagnose_model_performance(X_train, y_train, X_val, y_val)

        # 6. 评估训练集（含增强数据）
        full_train_pred, full_train_uncertainty = model.predict(model.train_X_augmented)
        full_train_r2 = r2_score(
            model.train_y_augmented,
            full_train_pred,
            sample_weight=model.train_weights
        )

        # 7. 评估原始训练集
        original_mask = model.train_augmentation_flags == 0
        original_train_performance = {}
        if np.sum(original_mask) > 0:
            original_train_pred = full_train_pred[original_mask]
            original_train_y = model.train_y_augmented.values[original_mask]
            original_train_weights = model.train_weights[original_mask]
            original_train_r2 = r2_score(
                original_train_y,
                original_train_pred,
                sample_weight=original_train_weights
            )
            original_train_performance = {
                'r2_score': original_train_r2,
                'mae': mean_absolute_error(original_train_y, original_train_pred, sample_weight=original_train_weights),
                'rmse': np.sqrt(
                    mean_squared_error(original_train_y, original_train_pred, sample_weight=original_train_weights))
            }

        # 8. 评估验证集
        val_pred, val_uncertainty = model.predict(X_val)
        val_sto_flags = (X_val[sto_column] == 'STO').astype(int).values if sto_column in X_val.columns else None

        X_val_raw = X_val[base_features].values
        val_weights = calculate_boundary_penalty_weights(
            X_val_raw, base_features,
            gaussian_sigma=config.weight_params['gaussian_sigma'],
            missing_penalty_rate=config.weight_params['missing_penalty_rate'],
            penalty_sharpness=config.weight_params['penalty_sharpness'],
            use_mahalanobis=config.weight_params['use_mahalanobis'],
            fixed_covariance_inv=model.fixed_covariance_inv
        )

        val_performance = {
            'full': {
                'r2_score': r2_score(y_val, val_pred, sample_weight=val_weights),
                'mae': mean_absolute_error(y_val, val_pred, sample_weight=val_weights),
                'rmse': np.sqrt(mean_squared_error(y_val, val_pred, sample_weight=val_weights)),
                'mean_uncertainty': np.mean(val_uncertainty),
                'std_uncertainty': np.std(val_uncertainty),
                'uncertainty_95_percentile': np.percentile(val_uncertainty, 95)
            }
        }

        # 9. 关键修改：评估全数据集+训练集增强数据
        # logger.info("开始对全数据集+训练集增强数据进行评估...")

        # 合并全数据集（训练集+验证集）和增强训练集
        X_full = pd.concat([X_train, X_val], ignore_index=True)
        y_full = pd.concat([y_train, y_val], ignore_index=True)

        # 增强训练集（来自model）
        X_augmented = model.train_X_augmented
        y_augmented = model.train_y_augmented

        # 合并数据
        X_combined = pd.concat([X_full, X_augmented], ignore_index=True)
        y_combined = pd.concat([y_full, y_augmented], ignore_index=True)

        # 提取STO标签
        if sto_column in X_combined.columns:
            combined_sto_flags = (X_combined[sto_column] == 'STO').astype(int).values
        else:
            combined_sto_flags = np.zeros(len(X_combined))

        # 对合并后的数据集进行预测
        combined_pred, combined_uncertainty = model.predict(X_combined)

        # 计算合并数据集的权重
        X_combined_raw = X_combined[base_features].values
        combined_weights = calculate_boundary_penalty_weights(
            X_combined_raw, base_features,
            gaussian_sigma=config.weight_params['gaussian_sigma'],
            missing_penalty_rate=config.weight_params['missing_penalty_rate'],
            penalty_sharpness=config.weight_params['penalty_sharpness'],
            use_mahalanobis=config.weight_params['use_mahalanobis'],
            fixed_covariance_inv=model.fixed_covariance_inv
        )

        # 计算合并数据集的性能指标
        combined_performance = {
            'full': {
                'r2_score': r2_score(y_combined, combined_pred, sample_weight=combined_weights),
                'mae': mean_absolute_error(y_combined, combined_pred, sample_weight=combined_weights),
                'rmse': np.sqrt(mean_squared_error(y_combined, combined_pred, sample_weight=combined_weights)),
                'mean_uncertainty': np.mean(combined_uncertainty),
                'std_uncertainty': np.std(combined_uncertainty),
                'uncertainty_95_percentile': np.percentile(combined_uncertainty, 95),
                'total_samples': len(y_combined),
                'original_samples': len(X_full),
                'augmented_samples': len(X_augmented)
            }
        }

        # STO样本评估
        if np.sum(combined_sto_flags) > 0:
            sto_mask = combined_sto_flags == 1
            combined_performance['sto'] = {
                'r2_score': r2_score(y_combined[sto_mask], combined_pred[sto_mask],
                                     sample_weight=combined_weights[sto_mask]),
                'mae': mean_absolute_error(y_combined[sto_mask], combined_pred[sto_mask],
                                           sample_weight=combined_weights[sto_mask]),
                'rmse': np.sqrt(mean_squared_error(y_combined[sto_mask], combined_pred[sto_mask],
                                                   sample_weight=combined_weights[sto_mask])),
                'mean_uncertainty': np.mean(combined_uncertainty[sto_mask]),
                'std_uncertainty': np.std(combined_uncertainty[sto_mask]),
                'uncertainty_95_percentile': np.percentile(combined_uncertainty[sto_mask], 95),
                'num_samples': np.sum(sto_mask)
            }

        # 10. 可视化
        mode = config.output_control['mode']
        if mode != 'silent':
            visualizer = ModelVisualizer()

            # 训练历史
            if hasattr(model, 'history') and model.history:
                fig1 = visualizer.plot_training_history(
                    model.history,
                    save_path=str(output_path / "training_history.png"),
                    feature_names=base_features
                )
                plt.close(fig1)

            # 合并数据集预测图
            fig_combined = visualizer.plot_predictions(
                y_combined, combined_pred, combined_uncertainty,
                full_r2=combined_performance['full']['r2_score'],
                sto_r2=combined_performance.get('sto', {}).get('r2_score'),
                save_path=str(output_path / "combined_dataset_predictions.png"),
                sto_flags=combined_sto_flags,
                sample_weights=combined_weights,
                augmentation_flags=np.concatenate([
                    np.zeros(len(X_full)),  # 原始数据
                    np.ones(len(X_augmented))  # 增强数据
                ])
            )
            plt.close(fig_combined)
            # logger.info("合并数据集预测图已保存")

        # 11. 保存模型和结果
        model.save_model(str(output_path / "model"))

        # 构建完整的结果字典
        results = {
            # 添加训练集和验证集数据
            'X_train': X_train,  # 原始训练集特征
            'y_train': y_train,  # 原始训练集目标
            'X_val': X_val,  # 验证集特征
            'y_val': y_val,  # 验证集目标
            'train_performance': {
                'full': {
                    'r2_score': full_train_r2,
                    'mae': mean_absolute_error(model.train_y_augmented, full_train_pred,
                                               sample_weight=model.train_weights),
                    'rmse': np.sqrt(mean_squared_error(model.train_y_augmented, full_train_pred,
                                                       sample_weight=model.train_weights)),
                    'mean_uncertainty': np.mean(full_train_uncertainty),
                    'std_uncertainty': np.std(full_train_uncertainty),
                    'uncertainty_95_percentile': np.percentile(full_train_uncertainty, 95),
                    'total_samples': len(model.train_y_augmented),
                    'original_samples': np.sum(original_mask),
                    'augmented_samples': np.sum(model.train_augmentation_flags == 1)
                },
                'original_only': original_train_performance
            },
            'val_performance': val_performance,
            'combined_dataset_performance': combined_performance,
            'diagnostic_results': diagnostic_results,
            'config': config,
            'feature_names': base_features,
            'feature_importance': model.history.get('feature_importance', []),
            'X_combined': X_combined,  # 保存合并后的特征数据
            'y_combined': y_combined,  # 保存合并后的目标数据
            'combined_pred': combined_pred,  # 保存预测结果
            'combined_uncertainty': combined_uncertainty  # 保存不确定性
        }

        joblib.dump(results, str(output_path / "results.joblib"))

        # 12. 生成实验报告
        create_experiment_report(results, output_path)

        return model, results

    return run_experiment


if __name__ == "__main__":
    # 配置参数
    DATA_PATH = os.path.join(root_dir, './data/converted_file.xlsx')
    BASE_FEATURES = ['Oxygen pressure', 'Laser energy density', 'Temperature', 'Frequency', 'Thickness']
    TARGET_COLUMN = 'rrr'
    STO_COLUMN = 'Substrate'
    # Sequence data paths
    SEQUENCE_GOOD_PATH = os.path.join(root_dir, './data/extracted_conditions_good.csv')
    SEQUENCE_BAD_PATH = os.path.join(root_dir, './data/extracted_conditions_bad.csv')

    # 创建配置
    custom_config = ModelConfig()
    custom_config.output_control['mode'] = 'detailed'

    # 使用混合权重方法
    custom_config.weight_calculation['method'] = 'hybrid'
    custom_config.weight_calculation['clustering']['methods'] = ['kmeans', 'density', 'residual']
    custom_config.weight_calculation['validation_method'] = 'boundary_only'

    # 创建并运行实验
    run_experiment = create_experiment_pipeline()
    try:
        model, results = run_experiment(
            data_path=DATA_PATH,
            base_features=BASE_FEATURES,
            target_column=TARGET_COLUMN,
            sto_column=STO_COLUMN,
            config=custom_config,
            output_dir="./model_results",
            val_size=0.5,
            random_state=42,
            sequence_good_path=SEQUENCE_GOOD_PATH,
            sequence_bad_path=SEQUENCE_BAD_PATH
        )

        # 打印四组评估结果
        print("\n" + "=" * 80)
        print("实验完成！四组评估结果对比")
        print("=" * 80)

        # 1. 训练集性能（含增强数据）
        train_full = results['train_performance']['full']
        print(f"1. 训练集性能 (含增强数据):")
        print(f"   R²: {train_full['r2_score']:.4f}")
        print(f"   MAE: {train_full['mae']:.4f}")
        print(f"   RMSE: {train_full['rmse']:.4f}")
        print(f"   样本数: {train_full.get('total_samples', 'N/A')}")
        print(f"   原始样本: {train_full.get('original_samples', 'N/A')}")
        print(f"   增强样本: {train_full.get('augmented_samples', 'N/A')}")

        # 2. 验证集性能
        val_full = results['val_performance']['full']
        print(f"\n2. 验证集性能:")
        print(f"   R²: {val_full['r2_score']:.4f}")
        print(f"   MAE: {val_full['mae']:.4f}")
        print(f"   RMSE: {val_full['rmse']:.4f}")
        print(f"   样本数: {len(results['X_val']) if 'X_val' in results else 'N/A'}")
        # 3. 全数据集性能（训练集+验证集，不含增强数据）
        # 需要先计算这个指标
        if 'X_train' in results and 'X_val' in results:
            # 合并训练集和验证集（不含增强数据）
            X_full = pd.concat([results['X_train'], results['X_val']], ignore_index=True)
            y_full = pd.concat([results['y_train'], results['y_val']], ignore_index=True)

            # 预测
            full_pred, full_uncertainty = model.predict(X_full)

            # 计算权重
            X_full_raw = X_full[BASE_FEATURES].values
            full_weights = calculate_boundary_penalty_weights(
                X_full_raw, BASE_FEATURES,
                gaussian_sigma=custom_config.weight_params['gaussian_sigma'],
                missing_penalty_rate=custom_config.weight_params['missing_penalty_rate'],
                penalty_sharpness=custom_config.weight_params['penalty_sharpness'],
                use_mahalanobis=custom_config.weight_params['use_mahalanobis'],
                fixed_covariance_inv=model.fixed_covariance_inv,
                weight_threshold=0.0
            )

            # 计算性能指标
            full_r2 = r2_score(y_full, full_pred, sample_weight=full_weights)
            full_mae = mean_absolute_error(y_full, full_pred, sample_weight=full_weights)
            full_rmse = np.sqrt(mean_squared_error(y_full, full_pred, sample_weight=full_weights))

            print(f"\n3. 全数据集性能 (训练+验证，不含增强):")
            print(f"   R²: {full_r2:.4f}")
            print(f"   MAE: {full_mae:.4f}")
            print(f"   RMSE: {full_rmse:.4f}")
            print(f"   样本数: {len(y_full)}")
        else:
            print(f"\n3. 全数据集性能 (训练+验证，不含增强): 数据不可用")

        # 4. 全数据集+增强数据性能
        if 'combined_dataset_performance' in results:
            combined_full = results['combined_dataset_performance']['full']
            print(f"\n4. 全数据集+增强数据性能:")
            print(f"   R²: {combined_full['r2_score']:.4f}")
            print(f"   MAE: {combined_full['mae']:.4f}")
            print(f"   RMSE: {combined_full['rmse']:.4f}")
            print(f"   总样本数: {combined_full.get('total_samples', 'N/A')}")
            print(f"   原始样本: {combined_full.get('original_samples', 'N/A')}")
            print(f"   增强样本: {combined_full.get('augmented_samples', 'N/A')}")
        else:
            print(f"\n4. 全数据集+增强数据性能: 数据不可用")

        # 性能对比分析
        print("\n" + "=" * 80)
        print("性能对比分析")
        print("=" * 80)

        # 创建对比表格
        comparison_data = []

        # 训练集
        comparison_data.append([
            "训练集 (含增强)",
            f"{train_full['r2_score']:.4f}",
            f"{train_full['mae']:.4f}",
            f"{train_full['rmse']:.4f}",
            f"{train_full.get('total_samples', 'N/A')}"
        ])

        # 验证集
        comparison_data.append([
            "验证集",
            f"{val_full['r2_score']:.4f}",
            f"{val_full['mae']:.4f}",
            f"{val_full['rmse']:.4f}",
            f"{len(results.get('X_val', [])) if 'X_val' in results else 'N/A'}"
        ])

        # 全数据集 (训练+验证)
        if 'X_train' in results and 'X_val' in results:
            comparison_data.append([
                "全数据集 (训练+验证)",
                f"{full_r2:.4f}",
                f"{full_mae:.4f}",
                f"{full_rmse:.4f}",
                f"{len(results['X_train']) + len(results['X_val'])}"
            ])
        else:
            comparison_data.append([
                "全数据集 (训练+验证)",
                "N/A",
                "N/A",
                "N/A",
                "N/A"
            ])

        # 全数据集+增强
        if 'combined_dataset_performance' in results:
            comparison_data.append([
                "全数据集+增强",
                f"{combined_full['r2_score']:.4f}",
                f"{combined_full['mae']:.4f}",
                f"{combined_full['rmse']:.4f}",
                f"{combined_full.get('total_samples', 'N/A')}"
            ])
        else:
            comparison_data.append([
                "全数据集+增强",
                "N/A",
                "N/A",
                "N/A",
                "N/A"
            ])

        # 打印对比表格
        print(f"{'数据集':<20} {'R²':<8} {'MAE':<8} {'RMSE':<8} {'样本数':<8}")
        print("-" * 60)
        for row in comparison_data:
            print(f"{row[0]:<20} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8}")

        # 过拟合分析
        overfit_degree = train_full['r2_score'] - val_full['r2_score']
        print(f"\n过拟合分析:")
        print(f"  训练集R² - 验证集R²: {overfit_degree:.4f}")
        if overfit_degree > 0.1:
            print(f"  警告: 可能存在明显过拟合")
        elif overfit_degree < -0.05:
            print(f"  注意: 验证集性能优于训练集，可能数据分布不均")
        else:
            print(f"  模型泛化能力良好")

        # 增强效果分析
        if 'combined_dataset_performance' in results and 'X_train' in results and 'X_val' in results:
            enhancement_effect = combined_full['r2_score'] - full_r2
            print(f"\n增强效果分析:")
            print(f"  增强后R² - 增强前R²: {enhancement_effect:.4f}")
            if enhancement_effect > 0.02:
                print(f"  数据增强对模型性能有积极影响")
            elif enhancement_effect < -0.02:
                print(f"  数据增强可能对模型性能产生了负面影响")
            else:
                print(f"  数据增强对模型性能影响不明显")

        print("=" * 80)

        # ========== 绘制四组数据对应的图 ==========
        print("\n开始绘制四组数据的预测图...")

        # 获取可视化器
        visualizer = ModelVisualizer()

        # 1. 训练集预测图（含增强数据）
        if hasattr(model, 'train_X_augmented') and hasattr(model, 'train_y_augmented'):
            train_pred, train_unc = model.predict(model.train_X_augmented)
            train_sto_flags = model.train_sto_flags if hasattr(model, 'train_sto_flags') else None
            train_weights = model.train_weights if hasattr(model, 'train_weights') else None
            train_aug_flags = model.train_augmentation_flags if hasattr(model, 'train_augmentation_flags') else None

            fig_train = visualizer.plot_predictions(
                model.train_y_augmented.values, train_pred, train_unc,
                full_r2=train_full['r2_score'],
                sto_r2=None,  # 可以在需要时计算STO的R²
                save_path="./model_results/train_set_predictions.png",
                sto_flags=train_sto_flags,
                sample_weights=train_weights,
                augmentation_flags=train_aug_flags
            )
            plt.close(fig_train)
            print("✓ 训练集预测图已保存")

        # 2. 验证集预测图
        if 'X_val' in results and 'y_val' in results:
            val_pred, val_unc = model.predict(results['X_val'])
            val_sto_flags = (results['X_val'][STO_COLUMN] == 'STO').astype(int).values if STO_COLUMN in results[
                'X_val'].columns else None

            fig_val = visualizer.plot_predictions(
                results['y_val'].values, val_pred, val_unc,
                full_r2=val_full['r2_score'],
                sto_r2=None,
                save_path="./model_results/validation_set_predictions.png",
                sto_flags=val_sto_flags,
                sample_weights=None  # 验证集可能没有权重
            )
            plt.close(fig_val)
            print("✓ 验证集预测图已保存")

        # 3. 全数据集预测图（不含增强）
        if 'X_train' in results and 'X_val' in results and 'y_train' in results and 'y_val' in results:
            # 合并训练集和验证集
            X_full = pd.concat([results['X_train'], results['X_val']], ignore_index=True)
            y_full = pd.concat([results['y_train'], results['y_val']], ignore_index=True)

            full_pred, full_unc = model.predict(X_full)
            full_sto_flags = (X_full[STO_COLUMN] == 'STO').astype(int).values if STO_COLUMN in X_full.columns else None

            fig_full = visualizer.plot_predictions(
                y_full.values, full_pred, full_unc,
                full_r2=full_r2,
                sto_r2=None,
                save_path="./model_results/full_dataset_predictions.png",
                sto_flags=full_sto_flags,
                sample_weights=None
            )
            plt.close(fig_full)
            print("✓ 全数据集预测图已保存")

        # 4. 全数据集+增强数据预测图
        if 'X_combined' in results and 'y_combined' in results:
            combined_pred = results['combined_pred']
            combined_unc = results['combined_uncertainty']
            combined_sto_flags = (results['X_combined'][STO_COLUMN] == 'STO').astype(int).values if STO_COLUMN in \
                                                                                                    results[
                                                                                                        'X_combined'].columns else None

            # 创建增强标记（原始数据为0，增强数据为1）
            total_original_samples = len(results['X_train']) + len(results['X_val'])
            num_augmented_samples = len(results['X_combined']) - total_original_samples
            combined_aug_flags = np.concatenate([
                np.zeros(total_original_samples),
                np.ones(num_augmented_samples)
            ]) if hasattr(model, 'train_X_augmented') else None

            fig_combined = visualizer.plot_predictions(
                results['y_combined'].values, combined_pred, combined_unc,
                full_r2=combined_full['r2_score'],
                sto_r2=None,
                save_path="./model_results/combined_dataset_predictions.png",
                sto_flags=combined_sto_flags,
                sample_weights=None,
                augmentation_flags=combined_aug_flags
            )
            plt.close(fig_combined)
            print("✓ 全数据集+增强数据预测图已保存")

        print("\n所有图表已保存到 ./model_results/ 目录")
        print("=" * 80)

    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise
