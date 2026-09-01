import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import joblib
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass, field
from tqdm import tqdm
import sys
import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import glob
import argparse

from utils.data_processer import EnhancedFeatureProcessor, calculate_boundary_penalty_weights

# 添加项目根目录到系统路径
current_script_path = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))
sys.path.insert(0, root_dir)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    # Hidden layer dimension exponent for BNN
    first_hidden_dims_pow: int = 7

    @property
    def bnn_params(self) -> Dict:
        base_dim = 2 ** self.first_hidden_dims_pow
        return {
            'hidden_dims': [base_dim, base_dim // 2, base_dim // 4],
            'dropout_rates': [2.6680e-01, 2.2358e-01, 2.3204e-01],
            'use_batchnorm': False,
            'use_layernorm': True,
            'activation': 'silu'
        }

    # Training parameters
    training_params: Dict = field(default_factory=lambda: {
        'bnn_epochs': 1000,
        'batch_size': 64,
        'learning_rate': 7.9547e-03,
        'weight_decay': 1e-2,
        'clip_grad_norm': 1.0,
        'patience': 10,
        'min_delta_loss': 1e-7,
        'min_delta_r2': 1e-5,
        'scheduler_factor': 0.8,
        'scheduler_patience': 10,
        'verbose_epoch': 10,
        'stability_window': 10,
        'stability_threshold': 0.01,
        'min_epochs': 50,
        'warmup_epochs': 30,
    })
    # Weight configuration
    weight_config: Dict = field(default_factory=lambda: {
        'weight_column': None,
        'weight_condition': None,
        'custom_weights': None
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
        'freeze_layers': True,
        'fine_tune_epochs': 1200,
        'xgb_fine_tune': False,
        'xgb_fine_tune_epochs': 100,
        'xgb_fine_tune_lr_ratio': 0.1,
        'sto_sample_weight_multiplier': 1.2,
        'other_weight': 0.9,
        'fine_tune_patience': 80,
        'fine_tune_min_epochs': 200,
        'fine_tune_min_delta_loss': 1e-8,
        'fine_tune_min_delta_r2': 1e-7,
        'fine_tune_stability_window': 30,
        'fine_tune_stability_threshold': 0.0005,
        'fine_tune_warmup_epochs': 80,
        'fine_tune_lr_scale': 0.08,
        'fine_tune_weight_decay_scale': 3.0,
        'fine_tune_momentum_decay': 0.03,
        'weight_change_penalty': 1e-4,
        'adaptive_lr_reduction': True,
        'lr_reduction_factor': 0.8,
        'lr_reduction_patience': 25,
        'gradient_accumulation_steps': 16,
        'min_lr_ratio': 5e-5,
        'stability_check_frequency': 3,
    })


# --- 新增: 自适应知识蒸馏损失 ---
class AdaptiveKDLoss(nn.Module):
    """自适应知识蒸馏损失 - 教师影响随epoch递减"""

    def __init__(self, temperature: float = 3.0,
                 initial_alpha: float = 0.8,  # 初始教师权重
                 final_alpha: float = 0.2,  # 最终教师权重
                 decay_epochs: int = 500):  # 衰减周期
        super().__init__()
        self.temperature = temperature
        self.initial_alpha = initial_alpha
        self.final_alpha = final_alpha
        self.decay_epochs = decay_epochs
        self.current_epoch = 0
        self.mse_loss = nn.MSELoss(reduction='none')

    def update_epoch(self, epoch: int):
        """更新当前epoch，自动调整alpha"""
        self.current_epoch = epoch

    def get_current_alpha(self):
        """计算当前的alpha值"""
        if self.current_epoch >= self.decay_epochs:
            return self.final_alpha

        # 线性衰减
        progress = self.current_epoch / self.decay_epochs
        alpha = self.initial_alpha - (self.initial_alpha - self.final_alpha) * progress
        return alpha

    def forward(self, student_output, teacher_output, true_labels, sample_weights=None):
        # 获取当前alpha
        current_alpha = self.get_current_alpha()

        # 软目标损失
        soft_loss = self.mse_loss(
            student_output / self.temperature,
            teacher_output / self.temperature
        )

        # 硬目标损失
        hard_loss = self.mse_loss(student_output, true_labels)

        if sample_weights is not None:
            soft_loss = (soft_loss * sample_weights).mean()
            hard_loss = (hard_loss * sample_weights).mean()
        else:
            soft_loss = soft_loss.mean()
            hard_loss = hard_loss.mean()

        # 组合损失（alpha随训练递减）
        total_loss = (current_alpha * soft_loss * (self.temperature ** 2) +
                      (1 - current_alpha) * hard_loss)

        return total_loss, soft_loss, hard_loss, current_alpha


# --- 修改: 微调配置参数 ---
@dataclass
class FineTuneConfig:
    """微调配置参数 (基础配置)"""
    # 新增STO权重配置
    sto_weight_multiplier: float = 1.5  # STO数据权重倍数
    # 其他参数保持不变...
    # XGBoost微调参数
    xgb_fine_tune: bool = False
    xgb_learning_rate_ratio: float = 0.05
    xgb_n_estimators: int = 50
    # BNN微调参数
    bnn_epochs: int = 1000
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    # 知识蒸馏参数
    distillation_alpha: float = 0.7
    distillation_temperature: float = 3.0
    soft_target_weight: float = 0.8
    # Early stopping参数
    patience: int = 100
    min_delta_loss: float = 1e-9
    min_epochs: int = 100
    warmup_epochs: int = 50
    # 正则化参数
    l2_regularization: float = 1e-5
    weight_change_penalty: float = 5e-5
    dropout_boost: float = 0.1
    # 学习率调度
    scheduler_patience: int = 50
    scheduler_factor: float = 0.7
    min_lr: float = 1e-7
    # 梯度相关
    clip_grad_norm: float = 0.1
    gradient_accumulation_steps: int = 4
    # 数据增强
    noise_std: float = 0.01
    label_smoothing: float = 0.02
    use_data_augmentation: bool = True
    n_augment: int = 2
    # 输出控制
    verbose: bool = True
    save_best_only: bool = True


# --- 新增: 小样本专用微调配置 ---
@dataclass
class SmallSampleFineTuneConfig(FineTuneConfig):
    """针对16个样本的优化配置"""
    # STO权重配置
    sto_weight_multiplier: float = 1.5  # 保持与原数据STO相同的权重倍数
    # XGBoost微调 - 更保守
    xgb_fine_tune: bool = True
    # BNN微调 - 极度保守
    bnn_epochs: int = 2000
    batch_size: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 5e-4
    # 知识蒸馏 - 高度依赖教师
    distillation_alpha: float = 0.9
    distillation_temperature: float = 4.0
    # Early stopping - 非常严格
    patience: int = 200
    min_delta_loss: float = 1e-10
    min_epochs: int = 300
    warmup_epochs: int = 150
    # 正则化 - 极强
    l2_regularization: float = 1e-4
    weight_change_penalty: float = 1e-4
    dropout_boost: float = 0.15
    # 梯度相关 - 极度保守
    clip_grad_norm: float = 0.05
    gradient_accumulation_steps: int = 8
    # 学习率调度 - 温和衰减
    scheduler_patience: int = 80
    scheduler_factor: float = 0.85
    min_lr: float = 1e-8
    # 数据增强
    use_data_augmentation: bool = False


# --- 修改: 知识蒸馏损失函数 ---
class KnowledgeDistillationLoss(nn.Module):
    """知识蒸馏损失函数 (保留原版作为备选)"""

    def __init__(self, temperature: float = 3.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor,
                true_labels: torch.Tensor, sample_weights: Optional[torch.Tensor] = None):
        """
        计算知识蒸馏损失
        Args:
            student_output: 学生模型(微调后)的输出
            teacher_output: 教师模型(原始模型)的输出
            true_labels: 真实标签
            sample_weights: 样本权重
        """
        # 软目标损失（蒸馏损失）- 使用温度缩放
        soft_loss = self.mse_loss(student_output / self.temperature,
                                  teacher_output / self.temperature)
        # 硬目标损失（真实标签损失）
        hard_loss = self.mse_loss(student_output, true_labels)
        # 组合损失
        if sample_weights is not None:
            soft_loss = (soft_loss * sample_weights).mean()
            hard_loss = (hard_loss * sample_weights).mean()
        else:
            soft_loss = soft_loss.mean()
            hard_loss = hard_loss.mean()
        # 温度平方用于缩放软损失（标准KD做法）
        total_loss = (self.alpha * soft_loss * (self.temperature ** 2) +
                      (1 - self.alpha) * hard_loss)
        return total_loss, soft_loss, hard_loss


# --- 新增: Layer-wise学习率衰减函数 ---
def get_layer_wise_lr(model: nn.Module, base_lr: float, decay_factor: float = 0.95):
    """
    对于小样本,越深的层学习率越小
    """
    params = []
    n_layers = len(list(model.named_parameters()))

    for idx, (name, param) in enumerate(model.named_parameters()):
        if param.requires_grad:
            # 从输出层到输入层,学习率递减
            layer_lr = base_lr * (decay_factor ** (n_layers - idx - 1))
            params.append({
                'params': param,
                'lr': layer_lr,
                'name': name
            })

    return params


# --- 修改: 模型微调器 ---
class ModelFineTuner:
    def __init__(self, model_path: str, config: FineTuneConfig):
        self.config = config
        self.model_path = Path(model_path)

        # Load teacher model
        logger.info(f"Loading teacher model from {model_path}")
        try:
            self.teacher_model = HybridModel.load_model(str(self.model_path))
            # Validate XGBoost parameters
            teacher_xgb_params = self.teacher_model.xgb_model.get_params()
            if teacher_xgb_params.get('learning_rate') is None:
                logger.warning("Teacher model XGBoost learning_rate is None, setting to default 0.1")
                self.teacher_model.xgb_model.set_params(learning_rate=0.1)
            if teacher_xgb_params.get('n_estimators') is None:
                logger.warning("Teacher model XGBoost n_estimators is None, setting to default 100")
                self.teacher_model.xgb_model.set_params(n_estimators=100)
        except Exception as e:
            logger.error(f"Failed to load teacher model: {e}")
            raise

        # *** NEW: Store teacher model's weight calculation configuration ***
        self.teacher_weight_config = {
            'method': self.teacher_model.config.weight_calculation.get('method', 'boundary'),
            'clustering_config': self.teacher_model.config.weight_calculation.get('clustering', {}),
            'weight_params': self.teacher_model.config.weight_params.copy(),
            'fixed_covariance_inv': self.teacher_model.fixed_covariance_inv
        }

        # *** NEW: Store teacher's cluster calculator if it exists ***
        self.teacher_cluster_calculator = self.teacher_model.cluster_calculator

        logger.info(f"Teacher model weight calculation method: {self.teacher_weight_config['method']}")

        # Create student model (deep copy)
        try:
            self.student_model = HybridModel.load_model(str(self.model_path))
            student_xgb_params = self.student_model.xgb_model.get_params()
            if student_xgb_params.get('learning_rate') is None:
                logger.warning("Student model XGBoost learning_rate is None, setting to default 0.1")
                self.student_model.xgb_model.set_params(learning_rate=0.1)
            if student_xgb_params.get('n_estimators') is None:
                logger.warning("Student model XGBoost n_estimators is None, setting to default 100")
                self.student_model.xgb_model.set_params(n_estimators=100)
        except Exception as e:
            logger.error(f"Failed to load student model: {e}")
            raise

        # Save original BNN weights for regularization
        self.original_bnn_state = {
            name: param.clone().detach().cpu()
            for name, param in self.student_model.bnn_model.named_parameters()
        }

        # Freeze teacher model
        self.teacher_model.bnn_model.eval()
        for param in self.teacher_model.bnn_model.parameters():
            param.requires_grad = False

        # Training history
        self.history = {
            'loss': [], 'distill_loss': [], 'hard_loss': [],
            'r2': [], 'learning_rates': []
        }

        # Early stopping
        self.best_loss = float('inf')
        self.best_r2 = -float('inf')
        self.patience_counter = 0
        self.best_model_state = None

    def _calculate_weights_like_teacher(self, X_raw: np.ndarray, sto_flags: np.ndarray) -> np.ndarray:
        """
        Calculate weights using the same method as the teacher model during training.
        This ensures consistency in evaluation metrics.

        Args:
            X_raw: Raw feature matrix
            sto_flags: STO flags (1 for STO, 0 for non-STO)

        Returns:
            weights: Sample weights calculated using teacher's method
        """
        # Get teacher model's weight calculation config
        teacher_config = self.teacher_model.config

        # Calculate boundary weights using teacher's fixed covariance
        weights = calculate_boundary_penalty_weights(
            X_raw,
            self.teacher_model.base_features,
            gaussian_sigma=teacher_config.weight_params['gaussian_sigma'],
            missing_penalty_rate=teacher_config.weight_params['missing_penalty_rate'],
            penalty_sharpness=teacher_config.weight_params['penalty_sharpness'],
            use_mahalanobis=teacher_config.weight_params['use_mahalanobis'],
            fixed_covariance_inv=self.teacher_model.fixed_covariance_inv,
            weight_threshold=0.0  # Don't threshold yet
        )

        # Apply STO weight multiplier (from teacher model's training)
        if np.sum(sto_flags == 1) > 0:
            sto_mask = sto_flags == 1
            # Use the same STO multiplier as during teacher training
            # This should match the multiplier used in the original model training
            sto_weight_multiplier = teacher_config.sto_training.get('sto_sample_weight_multiplier', 1.2)
            weights[sto_mask] *= sto_weight_multiplier

        # Normalize weights (same as teacher model)
        weights = weights / np.mean(weights)
        weights = np.clip(weights, 0.05, 3.0)

        return weights

    def _add_input_noise(self, X: torch.Tensor) -> torch.Tensor:
        """添加输入噪声作为数据增强"""
        if self.config.noise_std > 0 and self.student_model.bnn_model.training:
            noise = torch.randn_like(X) * self.config.noise_std
            return X + noise
        return X

    def _smooth_labels(self, y: torch.Tensor) -> torch.Tensor:
        """标签平滑"""
        if self.config.label_smoothing > 0:
            noise = torch.randn_like(y) * self.config.label_smoothing
            return y + noise
        return y

    def fine_tune_xgboost(self, X_raw: np.ndarray, y: np.ndarray,
                          sample_weights: Optional[np.ndarray] = None):
        """使用知识蒸馏微调XGBoost"""
        if not self.config.xgb_fine_tune:
            logger.info("Skipping XGBoost fine-tuning")
            return
        logger.info("Fine-tuning XGBoost with knowledge distillation")
        # 获取教师模型的预测作为软目标
        teacher_pred = self.teacher_model.xgb_model.predict(X_raw)
        # 创建混合目标：alpha * teacher_pred + (1-alpha) * true_labels
        alpha = self.config.distillation_alpha
        soft_targets = alpha * teacher_pred + (1 - alpha) * y
        # 获取原始学习率 - 添加默认值处理
        original_params = self.student_model.xgb_model.get_params()
        original_lr = original_params.get('learning_rate', 0.1)
        # 如果original_lr为None，使用默认值
        if original_lr is None:
            original_lr = 0.1
            logger.warning("Original learning rate is None, using default value 0.1")
        # 获取原始n_estimators - 添加默认值处理
        original_n_estimators = original_params.get('n_estimators', 100)
        # 如果original_n_estimators为None，使用默认值
        if original_n_estimators is None:
            original_n_estimators = 100
            logger.warning("Original n_estimators is None, using default value 100")
        # 设置微调参数
        fine_tune_lr = original_lr * self.config.xgb_learning_rate_ratio
        # 使用增量训练（warm start）
        logger.info(f"XGBoost fine-tune LR: {fine_tune_lr:.6f}")
        logger.info(f"Adding {self.config.xgb_n_estimators} new trees")
        # 创建新的XGBoost模型继续训练
        self.student_model.xgb_model.set_params(
            learning_rate=fine_tune_lr,
            n_estimators=original_n_estimators + self.config.xgb_n_estimators
        )
        # 在软目标上训练
        self.student_model.xgb_model.fit(
            X_raw, soft_targets,
            sample_weight=sample_weights,
            xgb_model=self.student_model.xgb_model.get_booster()
        )
        # 恢复原始学习率参数（保存时用）
        self.student_model.xgb_model.set_params(learning_rate=original_lr)
        logger.info("XGBoost fine-tuning completed")

    def fine_tune_bnn_small_sample(self, X_raw: np.ndarray, X_processed: np.ndarray,
                                   y: np.ndarray, sample_weights: Optional[np.ndarray] = None):
        """
        Optimized fine-tuning for small samples WITH data augmentation.
        """
        logger.info("Starting small-sample fine-tuning with data augmentation")

        # --- 1. Data Augmentation ---
        original_size = len(y)

        # 从教师模型获取权重参数
        teacher_weight_params = self.teacher_model.config.weight_params

        # 准备数据用于增强
        X_raw_df = pd.DataFrame(X_raw, columns=self.student_model.base_features)
        y_attention = pd.Series(y, name='target')

        # 确定目标大小
        if original_size < 30:
            target_multiplier = 5
        elif original_size < 50:
            target_multiplier = 4
        elif original_size < 100:
            target_multiplier = 3
        else:
            target_multiplier = 2

        target_size = int(original_size * target_multiplier)
        logger.info(f"Original size: {original_size}, Target size: {target_size}")

        # 执行数据增强
        X_augmented, y_augmented, sto_augmented = self.student_model._prepare_data_with_augmentation(
            X_raw_df, y_attention,
            sto_flags=np.ones(original_size),  # 所有都是STO
            target_size=target_size
        )

        logger.info(f"After augmentation: {len(y_augmented)} samples")
        logger.info(f"Augmented samples: {len(y_augmented) - original_size}")

        # --- 2. Calculate boundary weights on augmented data ---
        X_aug_raw = X_augmented[self.student_model.base_features].values

        # 使用教师模型的权重参数计算权重
        weights_aug = calculate_boundary_penalty_weights(
            X_aug_raw,
            self.student_model.base_features,
            gaussian_sigma=teacher_weight_params['gaussian_sigma'],
            missing_penalty_rate=teacher_weight_params['missing_penalty_rate'],
            penalty_sharpness=teacher_weight_params['penalty_sharpness'],
            use_mahalanobis=teacher_weight_params['use_mahalanobis'],
            fixed_covariance_inv=self.teacher_model.fixed_covariance_inv,
            weight_threshold=0.0
        )

        # 应用STO权重倍数
        weights_aug *= self.config.sto_weight_multiplier

        # 归一化权重
        weights_aug = weights_aug / np.mean(weights_aug)
        weights_aug = np.clip(weights_aug, 0.05, 3.0)

        logger.info(f"Weight statistics - mean: {np.mean(weights_aug):.3f}, "
                    f"min: {np.min(weights_aug):.3f}, max: {np.max(weights_aug):.3f}")

        # --- 3. Calculate residuals on augmented data ---
        xgb_pred = self.student_model.xgb_model.predict(X_aug_raw)
        residuals = y_augmented.values - xgb_pred
        residual_stats = self.teacher_model.residual_stats
        normalized_residuals = (residuals - residual_stats['mean']) / residual_stats['std']
        normalized_residuals = np.clip(normalized_residuals, -10, 10)

        # --- 4. Process augmented data ---
        X_aug_processed = self.student_model.processor.transform(X_augmented[self.student_model.base_features])

        # --- 5. Prepare dataset WITH MASK ---
        # CRITICAL FIX: Get mask from processor
        X_aug_processed_with_mask, aug_mask = self.student_model.processor.transform(
            X_augmented[self.student_model.base_features],
            return_mask=True
        )

        # Ensure mask is float32 and same shape as processed data
        aug_mask = aug_mask.astype(np.float32)
        if aug_mask.shape != X_aug_processed_with_mask.shape:
            logger.warning(
                f"Mask shape {aug_mask.shape} doesn't match processed data {X_aug_processed_with_mask.shape}")
            aug_mask = np.ones_like(X_aug_processed_with_mask, dtype=np.float32)

        dataset = TensorDataset(
            torch.FloatTensor(X_aug_processed_with_mask),
            torch.FloatTensor(normalized_residuals),
            torch.FloatTensor(aug_mask),  # Add mask to dataset
            torch.FloatTensor(weights_aug)
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False
        )

        # --- 6. Layer-wise learning rate optimizer ---
        param_groups = get_layer_wise_lr(
            self.student_model.bnn_model,
            base_lr=self.config.learning_rate,
            decay_factor=0.9
        )

        optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # --- 7. Learning rate schedulers ---
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.001,
            end_factor=1.0,
            total_iters=self.config.warmup_epochs
        )

        cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=50,
            T_mult=2,
            eta_min=self.config.min_lr
        )

        # --- 8. Adaptive knowledge distillation loss ---
        kd_criterion = AdaptiveKDLoss(
            temperature=self.config.distillation_temperature,
            initial_alpha=0.8,  # 开始时80%依赖教师
            final_alpha=0.2,  # 结束时只20%依赖教师
            decay_epochs=500
        )

        # --- 9. Get teacher predictions on augmented data WITH MASK ---
        X_aug_tensor = torch.FloatTensor(X_aug_processed_with_mask).to(device)
        y_aug_tensor = torch.FloatTensor(normalized_residuals).to(device)
        aug_mask_tensor = torch.FloatTensor(aug_mask).to(device)  # Add mask tensor

        with torch.no_grad():
            self.teacher_model.bnn_model.eval()
            # Pass mask to teacher model
            teacher_predictions = self.teacher_model.bnn_model(X_aug_tensor, aug_mask_tensor)

        # --- 10. Training loop ---
        best_loss = float('inf')
        best_r2 = -float('inf')
        patience_counter = 0

        epoch_iterator = tqdm(range(self.config.bnn_epochs), desc="Fine-tuning (Small Sample + Augmentation)")

        for epoch in epoch_iterator:
            kd_criterion.update_epoch(epoch)
            self.student_model.bnn_model.train()
            epoch_loss = 0.0
            epoch_distill_loss = 0.0
            epoch_hard_loss = 0.0
            num_samples = 0

            optimizer.zero_grad()
            accumulated_steps = 0

            for batch_idx, batch_data in enumerate(dataloader):
                # CRITICAL FIX: Unpack mask from batch
                inputs, targets, batch_mask, batch_weights = batch_data
                inputs = inputs.to(device)
                targets = targets.to(device)
                batch_mask = batch_mask.to(device)  # Add mask to device
                batch_weights = batch_weights.to(device)

                # Forward pass WITH MASK
                student_output = self.student_model.bnn_model(inputs, batch_mask).squeeze()

                # Get corresponding teacher predictions
                batch_start = batch_idx * self.config.batch_size
                batch_end = min(batch_start + len(inputs), len(teacher_predictions))
                teacher_output = teacher_predictions[batch_start:batch_end].squeeze()

                # Calculate loss
                total_loss, soft_loss, hard_loss, current_alpha = kd_criterion(
                    student_output, teacher_output, targets, batch_weights
                )

                # Weight change penalty
                weight_penalty = 0.0
                for name, param in self.student_model.bnn_model.named_parameters():
                    if name in self.original_bnn_state:
                        original_param = self.original_bnn_state[name].to(device)
                        weight_penalty += torch.norm(param - original_param, 2)

                # L2 regularization
                l2_reg = sum(torch.norm(p, 2) for p in self.student_model.bnn_model.parameters())

                # Total loss
                loss = (total_loss +
                        self.config.weight_change_penalty * weight_penalty +
                        self.config.l2_regularization * l2_reg)

                # Gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

                epoch_loss += total_loss.item() * inputs.size(0)
                epoch_distill_loss += soft_loss.item() * inputs.size(0)
                epoch_hard_loss += hard_loss.item() * inputs.size(0)
                num_samples += inputs.size(0)
                accumulated_steps += 1

                # Update after accumulation
                if accumulated_steps >= self.config.gradient_accumulation_steps:
                    torch.nn.utils.clip_grad_norm_(
                        self.student_model.bnn_model.parameters(),
                        self.config.clip_grad_norm
                    )
                    optimizer.step()
                    optimizer.zero_grad()
                    accumulated_steps = 0

            # Handle remaining gradients
            if accumulated_steps > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.student_model.bnn_model.parameters(),
                    self.config.clip_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad()

            avg_loss = epoch_loss / num_samples
            avg_distill_loss = epoch_distill_loss / num_samples
            avg_hard_loss = epoch_hard_loss / num_samples

            # Evaluation WITH MASK
            self.student_model.bnn_model.eval()
            with torch.no_grad():
                train_pred = self.student_model.bnn_model(X_aug_tensor, aug_mask_tensor).squeeze()
                train_r2 = self._compute_r2(y_aug_tensor, train_pred)

            # Learning rate scheduling
            if epoch < self.config.warmup_epochs:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']

            # Update progress bar
            epoch_iterator.set_postfix({
                'loss': f'{avg_loss:.6f}',
                'R2': f'{train_r2:.4f}',
                'lr': f'{current_lr:.2e}'
            })

            # Early stopping
            improved = False
            if train_r2 > best_r2 + 1e-6:
                best_r2 = train_r2
                best_loss = avg_loss
                self.best_model_state = self.student_model.bnn_model.state_dict().copy()
                patience_counter = 0
                improved = True
            else:
                patience_counter += 1

            # Check overfitting
            if epoch > self.config.min_epochs:
                if patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            # Record history
            self.history['loss'].append(avg_loss)
            self.history['distill_loss'].append(avg_distill_loss)
            self.history['hard_loss'].append(avg_hard_loss)
            self.history['r2'].append(train_r2)
            self.history['learning_rates'].append(current_lr)

        # Restore best model
        if self.best_model_state is not None:
            self.student_model.bnn_model.load_state_dict(self.best_model_state)
            logger.info(f"Restored best model with R²: {best_r2:.6f}")

        logger.info("BNN fine-tuning completed")

    def _compute_r2(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """计算R²分数"""
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        r2 = 1.0 - (ss_res / ss_tot)
        return r2.item()

    @staticmethod
    def weighted_r2_score(y_true: np.ndarray, y_pred: np.ndarray,
                          sample_weights: Optional[np.ndarray] = None) -> float:
        """
        计算加权R²分数（与原模型model.py一致）
        """
        if sample_weights is not None:
            sum_weights = np.sum(sample_weights)
            if sum_weights == 0:
                y_wmean = np.mean(y_true)
            else:
                y_wmean = np.sum(sample_weights * y_true) / sum_weights

            ss_res = np.sum(sample_weights * (y_true - y_pred) ** 2)
            ss_tot = np.sum(sample_weights * (y_true - y_wmean) ** 2)
        else:
            y_wmean = np.mean(y_true)
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - y_wmean) ** 2)

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0

        r2 = 1.0 - (ss_res / ss_tot)
        return r2

    def _check_early_stopping(self, current_loss: float, current_r2: float, epoch: int) -> bool:
        """检查是否应该早停"""
        if epoch < self.config.min_epochs:
            return False
        # R²改进检查
        r2_improved = current_r2 > (self.best_r2 + self.config.min_delta_loss)
        if r2_improved:
            self.best_r2 = current_r2
            self.best_loss = current_loss
            self.best_model_state = self.student_model.bnn_model.state_dict().copy()
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        return self.patience_counter >= self.config.patience

    def fine_tune(self, X_raw: np.ndarray, X_processed: np.ndarray, y: np.ndarray,
                  sample_weights: Optional[np.ndarray] = None) -> 'HybridModel':
        """Complete fine-tuning process"""
        logger.info("=" * 60)
        logger.info("Starting model fine-tuning with knowledge distillation")
        logger.info("=" * 60)
        logger.info(f"Fine-tuning samples: {len(y)}")
        logger.info(f"Distillation alpha: {self.config.distillation_alpha}")
        logger.info(f"Temperature: {self.config.distillation_temperature}")
        logger.info(f"Using data augmentation: {self.config.use_data_augmentation}")
        logger.info(f"XGBoost fine-tuning: {self.config.xgb_fine_tune}")
        logger.info("=" * 60)

        # Fine-tune XGBoost
        self.fine_tune_xgboost(X_raw, y, sample_weights)

        # CRITICAL FIX: Get processed data WITH MASK before fine-tuning BNN
        X_processed_with_mask, train_mask = self.student_model.processor.transform(
            pd.DataFrame(X_raw, columns=self.student_model.base_features),
            return_mask=True
        )

        # Choose fine-tuning method based on sample size
        if len(y) <= 50:
            logger.info("Detected small sample size, using optimized fine-tuning method.")
            self.fine_tune_bnn_small_sample(X_raw, X_processed_with_mask, y, sample_weights)
        else:
            logger.info("Using original fine-tuning method.")
            self.fine_tune_bnn_original(X_raw, X_processed_with_mask, y, sample_weights)

        logger.info("=" * 60)
        logger.info("Fine-tuning completed!")
        logger.info("=" * 60)
        return self.student_model

    def save_finetuned_model(self, save_path: str):
        """保存微调后的模型"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        try:
            # 保存学生模型
            self.student_model.save_model(str(save_path))

            # 保存微调历史
            history_path = save_path / "finetune_history.joblib"
            joblib.dump(self.history, str(history_path))

            # 保存配置
            config_path = save_path / "finetune_config.joblib"
            joblib.dump(self.config.__dict__, str(config_path))

            # 保存教师模型信息（用于参考）
            teacher_info = {
                'model_path': str(self.model_path),
                'model_type': 'HybridModel'
            }
            teacher_path = save_path / "teacher_model_info.joblib"
            joblib.dump(teacher_info, str(teacher_path))

            logger.info(f"Fine-tuned model successfully saved to {save_path}")
            logger.info(f" - Model files: {save_path}")
            logger.info(f" - Training history: {history_path}")
            logger.info(f" - Configuration: {config_path}")

        except Exception as e:
            logger.error(f"Failed to save fine-tuned model: {e}")
            raise

    def plot_training_history(self, save_path: Optional[str] = None):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        # Loss curves
        axes[0, 0].plot(self.history['loss'], label='Total Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        # Distillation vs Hard loss
        axes[0, 1].plot(self.history['distill_loss'], label='Distillation Loss', linewidth=2)
        axes[0, 1].plot(self.history['hard_loss'], label='Hard Loss', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Distillation vs Hard Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        # R² score
        axes[1, 0].plot(self.history['r2'], label='R² Score', linewidth=2, color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].set_title('R² Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        # Learning rate
        axes[1, 1].plot(self.history['learning_rates'], label='Learning Rate', linewidth=2, color='orange')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        return fig


def plot_prediction_comparison(model, X_original_raw, y_original,
                               X_new_raw, y_new, output_path, base_features,
                               teacher_model=None, original_weights=None, experiment_weights=None):
    """绘制原数据和新数据的预测结果对比图,在一张图中显示"""

    # 首先确保输入数据是合适的格式
    if X_original_raw is not None and isinstance(X_original_raw, np.ndarray):
        X_original_raw_df = pd.DataFrame(X_original_raw, columns=base_features)
    elif X_original_raw is not None:
        X_original_raw_df = X_original_raw
    else:
        X_original_raw_df = None

    if isinstance(X_new_raw, np.ndarray):
        X_new_raw_df = pd.DataFrame(X_new_raw, columns=base_features)
    else:
        X_new_raw_df = X_new_raw

    # 创建一个大图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 科研论文标准配色方案
    colors = {
        'original_non_sto': '#4E79A7',  # 柔和的蓝色
        'original_sto': '#98DF8A',  # 淡绿色
        'new_data': '#E15759',  # 红色(科研论文常用的红色)
        'ideal_line': '#666666',  # 中灰色
        'hist_original': '#AEC7E8',  # 浅蓝色(直方图)
        'hist_new': '#FF9898',  # 浅红色(直方图,对应Experiment Data)
        'teacher_model': '#D6D6D6'  # 浅灰色(基础模型预测效果)
    }

    # 设置科研论文风格的参数
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 10,
        'figure.titlesize': 13
    })

    # 获取原数据的STO标签
    original_data_raw_path = './data/converted_file.xlsx'
    original_data_raw_df = pd.read_excel(original_data_raw_path, engine='openpyxl')

    # 列名映射
    column_mapping = {
        'oxygen_pressure': 'Oxygen pressure',
        'laser_energy_density': 'Laser energy density',
        'temperature': 'Temperature',
        'frequency': 'Frequency',
        'thickness(measure)': 'Thickness',
        'rrr': 'RRR'
    }
    existing_columns = original_data_raw_df.columns
    mapping_to_apply = {k: v for k, v in column_mapping.items() if k in existing_columns}
    original_data_raw_df = original_data_raw_df.rename(columns=mapping_to_apply)

    # 确定STO标签
    if 'Substrate' in original_data_raw_df.columns:
        sto_flags = (original_data_raw_df['Substrate'] == 'STO').astype(int)
    else:
        sto_flags = np.zeros(len(original_data_raw_df))

    # === 新增: 计算 original_alphas ===
    def normalize_weights_for_alpha(weights, min_alpha=0.2, max_alpha=1.0):
        """将权重归一化到alpha值范围"""
        if weights is None:
            return None
        if np.max(weights) == np.min(weights):
            return np.full_like(weights, (min_alpha + max_alpha) / 2)
        normalized = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
        return min_alpha + normalized * (max_alpha - min_alpha)

    original_alphas = normalize_weights_for_alpha(original_weights)
    # === 新增结束 ===

    # 获取基础模型(教师模型)预测 - 使用 DataFrame
    teacher_pred_original, teacher_std_original = None, None
    teacher_pred_new, teacher_std_new = None, None
    if teacher_model is not None:
        if X_original_raw_df is not None:
            teacher_pred_original, teacher_std_original = teacher_model.predict(X_original_raw_df)
        teacher_pred_new, teacher_std_new = teacher_model.predict(X_new_raw_df)
        logger.info("Using teacher model predictions for comparison plot.")
    else:
        logger.warning("Teacher model not provided, skipping teacher model plots.")

    # 获取原数据和新数据的预测 - 使用 DataFrame
    if X_original_raw_df is not None:
        pred_original, std_original = model.predict(X_original_raw_df)

    # 新数据预测
    pred_new, std_new = model.predict(X_new_raw_df)

    # 1. 所有数据的真实值 vs 预测值散点图(添加基于权重的透明度)
    ax1 = axes[0, 0]
    if X_original_raw is not None:
        sto_mask = sto_flags == 1
        non_sto_mask = sto_flags == 0

        # 先绘制底层数据:Original non-STO(使用权重控制透明度)
        if np.sum(non_sto_mask) > 0 and original_alphas is not None:
            for i in np.where(non_sto_mask)[0]:
                ax1.scatter(y_original[i], pred_original[i],
                            alpha=original_alphas[i],
                            color=colors['original_non_sto'],
                            s=40, edgecolors='white', linewidth=0.5)
            ax1.scatter([], [], alpha=0.7, color=colors['original_non_sto'],
                        label='Original non-STO', s=40, edgecolors='white', linewidth=0.5)

        # 再绘制中层数据:Original STO(使用权重控制透明度)
        if np.sum(sto_mask) > 0 and original_alphas is not None:
            for i in np.where(sto_mask)[0]:
                ax1.scatter(y_original[i], pred_original[i],
                            alpha=original_alphas[i],
                            color=colors['original_sto'],
                            s=40, edgecolors='white', linewidth=0.5)
            ax1.scatter([], [], alpha=0.7, color=colors['original_sto'],
                        label='Original STO', s=40, edgecolors='white', linewidth=0.5)

    # 最后绘制上层数据:Experiment Data(在最上层,alpha=1.0)
    ax1.scatter(y_new, pred_new, alpha=1.0, color=colors['new_data'],
                label='Experiment Data', s=40, edgecolors='white', linewidth=0.5, zorder=10)

    # 添加理想线
    min_val = min(y_new.min(), y_original.min() if X_original_raw is not None else y_new.min())
    max_val = max(y_new.max(), y_original.max() if X_original_raw is not None else y_new.max())
    ax1.plot([min_val, max_val], [min_val, max_val],
             color=colors['ideal_line'], linestyle='--', alpha=0.8, linewidth=1.5)
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title('(a) True vs Predicted Values')
    ax1.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linewidth=0.5)

    # 2. 残差图(同样添加基于权重的透明度)
    ax2 = axes[0, 1]
    if X_original_raw is not None:
        residuals_original = y_original - pred_original

        # 先绘制底层数据:Original non-STO
        if np.sum(non_sto_mask) > 0 and original_alphas is not None:
            for i in np.where(non_sto_mask)[0]:
                ax2.scatter(pred_original[i], residuals_original[i],
                            alpha=original_alphas[i],
                            color=colors['original_non_sto'],
                            s=40, edgecolors='white', linewidth=0.5)
            ax2.scatter([], [], alpha=0.7, color=colors['original_non_sto'],
                        label='Original non-STO', s=40, edgecolors='white', linewidth=0.5)

        # 再绘制中层数据:Original STO
        if np.sum(sto_mask) > 0 and original_alphas is not None:
            for i in np.where(sto_mask)[0]:
                ax2.scatter(pred_original[i], residuals_original[i],
                            alpha=original_alphas[i],
                            color=colors['original_sto'],
                            s=40, edgecolors='white', linewidth=0.5)
            ax2.scatter([], [], alpha=0.7, color=colors['original_sto'],
                        label='Original STO', s=40, edgecolors='white', linewidth=0.5)

    # 最后绘制上层数据:Experiment Data (alpha=1.0)
    residuals_new = y_new - pred_new
    ax2.scatter(pred_new, residuals_new, alpha=1.0, color=colors['new_data'],
                label='Experiment Data', s=40, edgecolors='white', linewidth=0.5, zorder=10)
    ax2.axhline(y=0, color=colors['ideal_line'], linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('(b) Residual Analysis')
    ax2.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linewidth=0.5)

    # 3. 基础模型(教师模型)预测效果对比
    ax3 = axes[1, 0]
    if teacher_model is not None:
        # 绘制教师模型在原数据上的预测 vs 真实值
        if X_original_raw is not None and teacher_pred_original is not None:
            ax3.scatter(y_original, teacher_pred_original, alpha=0.6, color=colors['teacher_model'],
                        label='Teacher Model (Original)', s=30, edgecolors='white', linewidth=0.3, zorder=5)
        # 绘制教师模型在新数据上的预测 vs 真实值
        if teacher_pred_new is not None:
            ax3.scatter(y_new, teacher_pred_new, alpha=0.8, color=colors['new_data'],
                        label='Teacher Model (Experiment Data)', s=30, edgecolors='white', linewidth=0.3, zorder=10)
        # 添加理想线
        ax3.plot([min_val, max_val], [min_val, max_val],
                 color=colors['ideal_line'], linestyle='--', alpha=0.8, linewidth=1.5)
        ax3.set_xlabel('True Values')
        ax3.set_ylabel('Teacher Predicted Values')
        ax3.set_title('(c) Teacher Model Prediction Comparison')
        ax3.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.9)
        ax3.grid(True, alpha=0.3, linewidth=0.5)
    else:
        # 如果没有教师模型,绘制原数据和新数据的不确定性分布
        logger.warning("Teacher model not provided, plotting uncertainty distribution instead.")
        if X_original_raw is not None:
            ax3.hist(std_original, bins=30, alpha=0.7, color=colors['hist_original'],
                     label='Original Data', edgecolor='white', linewidth=0.5)
        ax3.hist(std_new, bins=30, alpha=0.7, color=colors['hist_new'],
                 label='Experiment Data', edgecolor='white', linewidth=0.5)
        ax3.set_xlabel('Prediction Uncertainty')
        ax3.set_ylabel('Frequency')
        ax3.set_title('(c) Uncertainty Distribution')
        ax3.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.9)
        ax3.grid(True, alpha=0.3, linewidth=0.5)

    # 4. 性能指标对比 - 使用加权R²
    ax4 = axes[1, 1]

    # 收集所有数据组的性能指标
    metrics_data = {}
    groups = []

    # Teacher Original All
    if teacher_model is not None and X_original_raw is not None and original_weights is not None:
        teacher_pred_original, _ = teacher_model.predict(
            pd.DataFrame(X_original_raw, columns=base_features) if isinstance(X_original_raw,
                                                                              np.ndarray) else X_original_raw
        )
        r2_original_teacher = ModelFineTuner.weighted_r2_score(
            y_original, teacher_pred_original, original_weights)
        mae_original_teacher = mean_absolute_error(y_original, teacher_pred_original)
        rmse_original_teacher = np.sqrt(mean_squared_error(y_original, teacher_pred_original))
        metrics_data['Teacher Original All'] = [r2_original_teacher, mae_original_teacher, rmse_original_teacher]
        groups.append('Teacher Original All')

    # Original All
    if X_original_raw is not None and original_weights is not None:
        r2_original = ModelFineTuner.weighted_r2_score(y_original, pred_original, original_weights)
        mae_original = mean_absolute_error(y_original, pred_original)
        rmse_original = np.sqrt(mean_squared_error(y_original, pred_original))
        metrics_data['Original All'] = [r2_original, mae_original, rmse_original]
        groups.append('Original All')

        # Original STO
        if np.sum(sto_mask) > 0:
            sto_weights = original_weights[sto_mask]
            r2_sto = ModelFineTuner.weighted_r2_score(
                y_original[sto_mask], pred_original[sto_mask], sto_weights)
            mae_sto = mean_absolute_error(y_original[sto_mask], pred_original[sto_mask])
            rmse_sto = np.sqrt(mean_squared_error(y_original[sto_mask], pred_original[sto_mask]))
            metrics_data['Original STO'] = [r2_sto, mae_sto, rmse_sto]
            groups.append('Original STO')

        # Original non-STO
        if np.sum(non_sto_mask) > 0:
            non_sto_weights = original_weights[non_sto_mask]
            r2_non_sto = ModelFineTuner.weighted_r2_score(
                y_original[non_sto_mask], pred_original[non_sto_mask], non_sto_weights)
            mae_non_sto = mean_absolute_error(y_original[non_sto_mask], pred_original[non_sto_mask])
            rmse_non_sto = np.sqrt(mean_squared_error(y_original[non_sto_mask], pred_original[non_sto_mask]))
            metrics_data['Original non-STO'] = [r2_non_sto, mae_non_sto, rmse_non_sto]
            groups.append('Original non-STO')

    # Teacher Experiment Data (只保留一个)
    if teacher_model is not None and experiment_weights is not None:
        r2_new_teacher = ModelFineTuner.weighted_r2_score(
            y_new, teacher_pred_new, experiment_weights)
        mae_new_teacher = mean_absolute_error(y_new, teacher_pred_new)
        rmse_new_teacher = np.sqrt(mean_squared_error(y_new, teacher_pred_new))
        metrics_data['Teacher Experiment Data'] = [r2_new_teacher, mae_new_teacher, rmse_new_teacher]
        groups.append('Teacher Experiment Data')

    # Student Experiment Data
    pred_new, _ = model.predict(
        pd.DataFrame(X_new_raw, columns=base_features) if isinstance(X_new_raw, np.ndarray) else X_new_raw
    )
    r2_new = ModelFineTuner.weighted_r2_score(y_new, pred_new, experiment_weights)
    mae_new = mean_absolute_error(y_new, pred_new)
    rmse_new = np.sqrt(mean_squared_error(y_new, pred_new))
    metrics_data['Experiment Data'] = [r2_new, mae_new, rmse_new]
    groups.append('Experiment Data')

    # 设置横坐标为不同Metrics
    metrics = ['R²', 'MAE', 'RMSE']
    x = np.arange(len(metrics))
    width = 0.8 / len(groups)

    # 颜色列表
    group_colors = [colors['teacher_model'], colors['original_non_sto'], colors['original_sto'],
                    '#AEC7E8', colors['new_data'], '#C5B0D5']

    # 计算y轴范围
    all_values = []
    for values in metrics_data.values():
        all_values.extend(values)
    y_min = min(all_values)
    y_max = max(all_values)

    if y_min < 0:
        y_min = y_min * 1.1
    else:
        y_min = 0
    if y_max > 0:
        y_max = y_max * 1.15
    else:
        y_max = abs(y_min) * 0.1

    # 绘制柱状图
    for i, group in enumerate(groups):
        values = metrics_data[group]
        positions = x - width * (len(groups) - 1) / 2 + i * width
        bars = ax4.bar(positions, values, width, label=group, alpha=0.8,
                       color=group_colors[i % len(group_colors)],
                       edgecolor='white', linewidth=0.5)

        # 为负值柱子添加特殊样式
        for j, (bar, val) in enumerate(zip(bars, values)):
            if val < 0:
                bar.set_alpha(0.9)
                bar.set_edgecolor('darkred')
                bar.set_linewidth(1)

    ax4.set_xlabel('Evaluation Metrics')
    ax4.set_ylabel('Score')
    ax4.set_title('(d) Performance Metrics Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.9)
    ax4.grid(True, alpha=0.3, linewidth=0.5)
    ax4.set_ylim(y_min, y_max)

    # 添加零线
    if min(all_values) < 0:
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.7)

    # 添加数值标签
    for i, group in enumerate(groups):
        values = metrics_data[group]
        positions = x - width * (len(groups) - 1) / 2 + i * width
        for j, (pos, val) in enumerate(zip(positions, values)):
            if val >= 0:
                va = 'bottom'
                y_offset = y_max * 0.01
            else:
                va = 'top'
                y_offset = y_min * 0.01
            color = 'black' if abs(val) > (y_max - y_min) * 0.1 else 'darkred'
            ax4.text(pos, val + y_offset, f'{val:.3f}',
                     ha='center', va=va, fontsize=8, fontweight='bold',
                     color=color, bbox=dict(boxstyle="round,pad=0.1", facecolor='white',
                                            alpha=0.7, edgecolor='none'))

    # 调整子图间距
    plt.tight_layout()

    # 保存高质量图片
    plt.savefig(str(output_path / "prediction_comparison_combined.png"),
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

    logger.info(f"Combined prediction comparison plot saved to {output_path / 'prediction_comparison_combined.png'}")
    plt.close()

    return fig


# --- 新增: 集成预测函数 ---
def ensemble_predict(teacher_model, student_model, X_raw, base_features,
                     teacher_weight: float = 0.5):
    """
    对于小样本,采用教师-学生模型集成预测
    教师模型权重更高(0.7)以保持稳定性
    """
    # 确保输入是 DataFrame
    if isinstance(X_raw, np.ndarray):
        X_raw_df = pd.DataFrame(X_raw, columns=base_features)
    elif isinstance(X_raw, pd.DataFrame):
        X_raw_df = X_raw
    else:
        raise ValueError(f"Unsupported input type: {type(X_raw)}")

    # 教师预测
    teacher_pred, teacher_std = teacher_model.predict(X_raw_df)

    # 学生预测
    student_pred, student_std = student_model.predict(X_raw_df)

    # 加权集成
    ensemble_pred = teacher_weight * teacher_pred + (1 - teacher_weight) * student_pred

    # 不确定性也加权
    ensemble_std = np.sqrt(
        teacher_weight * teacher_std ** 2 +
        (1 - teacher_weight) * student_std ** 2
    )

    return ensemble_pred, ensemble_std


def run_fine_tuning_experiment(
        model_path: str,
        new_data_path_raw: str,
        base_features: List[str],
        target_column: str,
        weight_column: Optional[str] = None,
        config: Optional[FineTuneConfig] = None,
        output_dir: str = "./finetuned_model"
):
    """运行微调实验（单个模型）"""
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"输出目录已创建: {output_path}")

    # 加载原数据（用于对比）
    logger.info("Loading original data for comparison")
    original_data_raw_path = './data/converted_file.xlsx'
    try:
        # 加载原数据
        original_data_raw = pd.read_excel(original_data_raw_path, engine='openpyxl')
        # 列名映射（与原模型一致）
        column_mapping = {
            'oxygen_pressure': 'Oxygen pressure',
            'laser_energy_density': 'Laser energy density',
            'temperature': 'Temperature',
            'frequency': 'Frequency',
            'thickness(measure)': 'Thickness',
            'rrr': 'RRR'
        }
        # 重命名列
        existing_columns = original_data_raw.columns
        mapping_to_apply = {k: v for k, v in column_mapping.items() if k in existing_columns}
        original_data_raw = original_data_raw.rename(columns=mapping_to_apply)

        # 提取原数据特征和目标
        X_original_raw = original_data_raw[base_features].values
        X_original_processed = original_data_raw[base_features].values
        y_original = original_data_raw[target_column].values

        # 获取原数据的STO标签
        sto_flags_original = None
        if 'Substrate' in original_data_raw.columns:
            sto_flags_original = (original_data_raw['Substrate'] == 'STO').astype(int)
            logger.info(f"Original data STO samples: {np.sum(sto_flags_original)}")
        elif 'is_sto' in original_data_raw.columns:
            sto_flags_original = original_data_raw['is_sto'].values
            logger.info(f"Original data STO samples: {np.sum(sto_flags_original)}")
        else:
            logger.warning("No STO information found in original data")
            sto_flags_original = np.zeros(len(y_original))

        # 直接使用STO权重，不计算边界权重
        logger.info("Using STO weights only (no boundary weights) for original data")
        original_weights = np.ones(len(y_original))

        # 应用STO权重
        if sto_flags_original is not None and np.sum(sto_flags_original) > 0:
            sto_mask = sto_flags_original == 1
            sto_weight_multiplier = config.sto_weight_multiplier if config else 1.5
            original_weights[sto_mask] *= sto_weight_multiplier

            # 权重归一化
            original_weights = original_weights / np.mean(original_weights)
            original_weights = np.clip(original_weights, 0.05, 3.0)
            logger.info(f"Applied STO weight multiplier to original STO data: {sto_weight_multiplier}")

        logger.info(f"Original data loaded: {len(y_original)} samples")
    except Exception as e:
        logger.warning(f"Failed to load original data: {e}")
        X_original_raw, X_original_processed, y_original = None, None, None
        sto_flags_original = None
        original_weights = None

    # 加载新数据（实验数据）
    logger.info(f"Loading Experiment Data from {new_data_path_raw}")
    file_extension = os.path.splitext(new_data_path_raw)[1].lower()
    if file_extension == '.csv':
        data_raw = pd.read_csv(new_data_path_raw)
    elif file_extension in ['.xlsx', '.xls']:
        data_raw = pd.read_excel(new_data_path_raw, engine='openpyxl')
    else:
        try:
            data_raw = pd.read_csv(new_data_path_raw)
        except:
            try:
                data_raw = pd.read_excel(new_data_path_raw, engine='openpyxl')
            except Exception as e:
                raise ValueError(f"无法读取文件 {new_data_path_raw}: {e}")

    # 列名映射
    column_mapping = {
        'oxygen_pressure': 'Oxygen pressure',
        'laser_energy_density': 'Laser energy density',
        'temperature': 'Temperature',
        'frequency': 'Frequency',
        'thickness(measure)': 'Thickness',
        'rrr': 'RRR'
    }
    existing_columns = data_raw.columns
    mapping_to_apply = {k: v for k, v in column_mapping.items() if k in existing_columns}
    data_raw = data_raw.rename(columns=mapping_to_apply)

    # 特征处理器
    logger.info("Standardizing data using EnhancedFeatureProcessor")
    feature_processor = EnhancedFeatureProcessor(
        base_features=base_features,
        scaler_type='robust',
        interpolation_method='knn',
        n_neighbors=3
    )
    X_raw = data_raw[base_features]
    X_processed = feature_processor.fit_transform(data_raw[base_features])
    y = data_raw[target_column].values

    # 为实验数据计算权重
    logger.info("Experiment data: all samples are STO data")
    sto_weight_multiplier = config.sto_weight_multiplier if config else 1.5
    sample_weights = np.ones(len(y)) * sto_weight_multiplier
    sample_weights = sample_weights / np.mean(sample_weights)
    sample_weights = np.clip(sample_weights, 0.05, 3.0)

    # 创建微调配置
    config = config or SmallSampleFineTuneConfig()
    logger.info(f"Using configuration: {config.__class__.__name__}")

    # 创建微调器
    fine_tuner = ModelFineTuner(model_path, config)

    # 执行微调
    logger.info("Starting model fine-tuning...")
    finetuned_model = fine_tuner.fine_tune(X_raw, X_processed, y, sample_weights)

    # === 保存微调后的模型 ===
    logger.info("Saving fine-tuned model...")
    try:
        # 保存模型
        model_save_path = output_path / "model"
        fine_tuner.save_finetuned_model(str(model_save_path))
        logger.info(f"Fine-tuned model saved to: {model_save_path}")

        # 保存训练历史图表
        history_plot_path = output_path / "training_history.png"
        fine_tuner.plot_training_history(str(history_plot_path))
        logger.info(f"Training history plot saved to: {history_plot_path}")

        # === 生成并保存预测对比图 ===
        logger.info("Generating prediction comparison plot...")
        try:
            plot_prediction_comparison(
                model=finetuned_model,
                X_original_raw=X_original_raw,
                y_original=y_original,
                X_new_raw=X_raw.values if isinstance(X_raw, pd.DataFrame) else X_raw,
                y_new=y,
                output_path=output_path,
                base_features=base_features,
                teacher_model=fine_tuner.teacher_model,
                original_weights=original_weights,
                experiment_weights=sample_weights
            )
            logger.info(f"Prediction comparison plot saved to: {output_path / 'prediction_comparison_combined.png'}")
        except Exception as e:
            logger.error(f"Failed to generate prediction comparison plot: {e}")
            import traceback
            logger.error(traceback.format_exc())

        # 保存特征处理器
        processor_path = output_path / "feature_processor.joblib"
        joblib.dump(feature_processor, str(processor_path))
        logger.info(f"Feature processor saved to: {processor_path}")

    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise

    # 评估微调效果
    logger.info("Evaluating fine-tuned model...")

    # 初始化评估结果字典
    evaluation_results = {}

    # 1. 在新数据上评估（微调后模型）
    pred_mean_new, pred_std_new = finetuned_model.predict(data_raw[base_features])
    evaluation_results['experiment_data_r2'] = ModelFineTuner.weighted_r2_score(y, pred_mean_new, sample_weights)
    evaluation_results['experiment_data_mae'] = mean_absolute_error(y, pred_mean_new)
    evaluation_results['experiment_data_rmse'] = np.sqrt(mean_squared_error(y, pred_mean_new))
    evaluation_results['experiment_data_mean_uncertainty'] = np.mean(pred_std_new)

    # 2. 在新数据上评估（原模型/教师模型）
    teacher_pred_new, teacher_std_new = fine_tuner.teacher_model.predict(data_raw[base_features])
    evaluation_results['teacher_experiment_data_r2'] = ModelFineTuner.weighted_r2_score(y, teacher_pred_new,
                                                                                        sample_weights)
    evaluation_results['teacher_experiment_data_mae'] = mean_absolute_error(y, teacher_pred_new)
    evaluation_results['teacher_experiment_data_rmse'] = np.sqrt(mean_squared_error(y, teacher_pred_new))
    evaluation_results['teacher_experiment_data_mean_uncertainty'] = np.mean(teacher_std_new)

    # 3. 在原数据上评估（如果可用）
    if X_original_raw is not None:
        X_original_raw_df = pd.DataFrame(X_original_raw, columns=base_features)

        # 微调后模型在原数据上的评估
        pred_mean_original, pred_std_original = finetuned_model.predict(X_original_raw_df)
        evaluation_results['original_all_r2'] = ModelFineTuner.weighted_r2_score(y_original, pred_mean_original,
                                                                                 original_weights)
        evaluation_results['original_all_mae'] = mean_absolute_error(y_original, pred_mean_original)
        evaluation_results['original_all_rmse'] = np.sqrt(mean_squared_error(y_original, pred_mean_original))
        evaluation_results['original_all_mean_uncertainty'] = np.mean(pred_std_original)

        # 原模型在原数据上的评估
        teacher_pred_original, teacher_std_original = fine_tuner.teacher_model.predict(X_original_raw_df)
        evaluation_results['teacher_original_all_r2'] = ModelFineTuner.weighted_r2_score(y_original,
                                                                                         teacher_pred_original,
                                                                                         original_weights)
        evaluation_results['teacher_original_all_mae'] = mean_absolute_error(y_original, teacher_pred_original)
        evaluation_results['teacher_original_all_rmse'] = np.sqrt(mean_squared_error(y_original, teacher_pred_original))
        evaluation_results['teacher_original_all_mean_uncertainty'] = np.mean(teacher_std_original)

        # 在原数据的STO子集上评估
        if sto_flags_original is not None and np.sum(sto_flags_original) > 0:
            sto_mask = sto_flags_original == 1
            sto_weights = original_weights[sto_mask] if original_weights is not None else None

            # 微调后模型在STO子集上的评估
            evaluation_results['original_sto_r2'] = ModelFineTuner.weighted_r2_score(
                y_original[sto_mask], pred_mean_original[sto_mask], sto_weights)
            evaluation_results['original_sto_mae'] = mean_absolute_error(y_original[sto_mask],
                                                                         pred_mean_original[sto_mask])
            evaluation_results['original_sto_rmse'] = np.sqrt(
                mean_squared_error(y_original[sto_mask], pred_mean_original[sto_mask]))

            # 原模型在STO子集上的评估
            evaluation_results['teacher_original_sto_r2'] = ModelFineTuner.weighted_r2_score(
                y_original[sto_mask], teacher_pred_original[sto_mask], sto_weights)
            evaluation_results['teacher_original_sto_mae'] = mean_absolute_error(y_original[sto_mask],
                                                                                 teacher_pred_original[sto_mask])
            evaluation_results['teacher_original_sto_rmse'] = np.sqrt(
                mean_squared_error(y_original[sto_mask], teacher_pred_original[sto_mask]))

        # 在原数据的non-STO子集上评估
        if sto_flags_original is not None and np.sum(sto_flags_original == 0) > 0:
            non_sto_mask = sto_flags_original == 0
            non_sto_weights = original_weights[non_sto_mask] if original_weights is not None else None

            # 微调后模型在non-STO子集上的评估
            evaluation_results['original_non_sto_r2'] = ModelFineTuner.weighted_r2_score(
                y_original[non_sto_mask], pred_mean_original[non_sto_mask], non_sto_weights)
            evaluation_results['original_non_sto_mae'] = mean_absolute_error(y_original[non_sto_mask],
                                                                             pred_mean_original[non_sto_mask])
            evaluation_results['original_non_sto_rmse'] = np.sqrt(
                mean_squared_error(y_original[non_sto_mask], pred_mean_original[non_sto_mask]))

            # 原模型在non-STO子集上的评估
            evaluation_results['teacher_original_non_sto_r2'] = ModelFineTuner.weighted_r2_score(
                y_original[non_sto_mask], teacher_pred_original[non_sto_mask], non_sto_weights)
            evaluation_results['teacher_original_non_sto_mae'] = mean_absolute_error(y_original[non_sto_mask],
                                                                                     teacher_pred_original[
                                                                                         non_sto_mask])
            evaluation_results['teacher_original_non_sto_rmse'] = np.sqrt(
                mean_squared_error(y_original[non_sto_mask], teacher_pred_original[non_sto_mask]))

    # 记录评估结果
    logger.info("Evaluation Results:")
    logger.info(f"Experiment Data - Student R²: {evaluation_results.get('experiment_data_r2', 'N/A'):.6f}")
    logger.info(f"Experiment Data - Teacher R²: {evaluation_results.get('teacher_experiment_data_r2', 'N/A'):.6f}")

    if 'original_all_r2' in evaluation_results:
        logger.info(f"Original All - Student R²: {evaluation_results['original_all_r2']:.6f}")
        logger.info(f"Original All - Teacher R²: {evaluation_results['teacher_original_all_r2']:.6f}")

    # 保存评估结果
    results = {
        'evaluation_results': evaluation_results,
        'config': config.__dict__ if hasattr(config, '__dict__') else str(config),
        'history': fine_tuner.history,
    }

    results_path = output_path / "finetune_results.joblib"
    joblib.dump(results, str(results_path))
    logger.info(f"Fine-tuning results saved to: {results_path}")

    # 保存文本格式的结果摘要
    summary_path = output_path / "results_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Fine-tuning Results Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Experiment Data - Student R²: {evaluation_results.get('experiment_data_r2', 'N/A'):.6f}\n")
        f.write(f"Experiment Data - Teacher R²: {evaluation_results.get('teacher_experiment_data_r2', 'N/A'):.6f}\n")
        if 'original_all_r2' in evaluation_results:
            f.write(f"Original All - Student R²: {evaluation_results['original_all_r2']:.6f}\n")
            f.write(f"Original All - Teacher R²: {evaluation_results['teacher_original_all_r2']:.6f}\n")
        f.write(f"Model saved to: {model_save_path}\n")
        f.write(f"Total training epochs: {len(fine_tuner.history['loss'])}\n")

    logger.info(f"Results summary saved to: {summary_path}")

    return finetuned_model, fine_tuner, results


def generate_performance_report(all_results: dict, output_dir: Path):
    """生成批量微调的性能对比报告，包括全部评估结果"""
    # 提取性能指标
    performance_data = []

    for solution_name, result in all_results.items():
        # 处理跳过的结果
        if result.get('status') == 'skipped':
            try:
                results_data = result.get('results', {})
                evaluation_results = results_data.get('evaluation_results', {})
                history = results_data.get('history', {})

                # 提取训练历史信息
                total_epochs = len(history.get('loss', [])) if history.get('loss') else None
                best_r2 = max(history.get('r2', [0])) if history.get('r2') else None
                best_loss = min(history.get('loss', [float('inf')])) if history.get('loss') else None
                final_loss = history.get('loss', [])[-1] if history.get('loss') else None
                final_r2 = history.get('r2', [])[-1] if history.get('r2') else None

                performance_data.append({
                    'solution': solution_name,
                    'status': 'skipped',
                    'error': None,
                    # 实验数据评估结果
                    'experiment_data_r2': evaluation_results.get('experiment_data_r2'),
                    'experiment_data_mae': evaluation_results.get('experiment_data_mae'),
                    'experiment_data_rmse': evaluation_results.get('experiment_data_rmse'),
                    'teacher_experiment_data_r2': evaluation_results.get('teacher_experiment_data_r2'),
                    'teacher_experiment_data_mae': evaluation_results.get('teacher_experiment_data_mae'),
                    'teacher_experiment_data_rmse': evaluation_results.get('teacher_experiment_data_rmse'),
                    # 原始数据全部评估结果
                    'original_all_r2': evaluation_results.get('original_all_r2'),
                    'original_all_mae': evaluation_results.get('original_all_mae'),
                    'original_all_rmse': evaluation_results.get('original_all_rmse'),
                    'teacher_original_all_r2': evaluation_results.get('teacher_original_all_r2'),
                    'teacher_original_all_mae': evaluation_results.get('teacher_original_all_mae'),
                    'teacher_original_all_rmse': evaluation_results.get('teacher_original_all_rmse'),
                    # 原始数据STO子集评估结果
                    'original_sto_r2': evaluation_results.get('original_sto_r2'),
                    'original_sto_mae': evaluation_results.get('original_sto_mae'),
                    'original_sto_rmse': evaluation_results.get('original_sto_rmse'),
                    'teacher_original_sto_r2': evaluation_results.get('teacher_original_sto_r2'),
                    'teacher_original_sto_mae': evaluation_results.get('teacher_original_sto_mae'),
                    'teacher_original_sto_rmse': evaluation_results.get('teacher_original_sto_rmse'),
                    # 原始数据non-STO子集评估结果
                    'original_non_sto_r2': evaluation_results.get('original_non_sto_r2'),
                    'original_non_sto_mae': evaluation_results.get('original_non_sto_mae'),
                    'original_non_sto_rmse': evaluation_results.get('original_non_sto_rmse'),
                    'teacher_original_non_sto_r2': evaluation_results.get('teacher_original_non_sto_r2'),
                    'teacher_original_non_sto_mae': evaluation_results.get('teacher_original_non_sto_mae'),
                    'teacher_original_non_sto_rmse': evaluation_results.get('teacher_original_non_sto_rmse'),
                    # 训练信息
                    'total_epochs': total_epochs,
                    'best_r2': best_r2,
                    'best_loss': best_loss,
                    'final_loss': final_loss,
                    'final_r2': final_r2,
                    'training_time_minutes': None
                })
            except Exception as e:
                logger.warning(f"Failed to extract data from skipped result {solution_name}: {e}")
                performance_data.append({
                    'solution': solution_name,
                    'status': 'skipped_no_data',
                    'error': str(e),
                    # 所有指标设为None
                    **{key: None for key in ['experiment_data_r2', 'experiment_data_mae', 'experiment_data_rmse',
                                             'teacher_experiment_data_r2', 'teacher_experiment_data_mae',
                                             'teacher_experiment_data_rmse', 'original_all_r2', 'original_all_mae',
                                             'original_all_rmse', 'teacher_original_all_r2', 'teacher_original_all_mae',
                                             'teacher_original_all_rmse', 'original_sto_r2', 'original_sto_mae',
                                             'original_sto_rmse', 'teacher_original_sto_r2', 'teacher_original_sto_mae',
                                             'teacher_original_sto_rmse', 'original_non_sto_r2', 'original_non_sto_mae',
                                             'original_non_sto_rmse', 'teacher_original_non_sto_r2',
                                             'teacher_original_non_sto_mae', 'teacher_original_non_sto_rmse',
                                             'total_epochs', 'best_r2', 'best_loss', 'final_loss', 'final_r2',
                                             'training_time_minutes']}
                })
            continue

        if 'error' in result:
            # 记录失败的结果
            performance_data.append({
                'solution': solution_name,
                'status': 'failed',
                'error': result['error'],
                # 其余字段保持不变...
                **{key: None for key in ['experiment_data_r2', 'experiment_data_mae', 'experiment_data_rmse',
                                         'teacher_experiment_data_r2', 'teacher_experiment_data_mae',
                                         'teacher_experiment_data_rmse', 'original_all_r2', 'original_all_mae',
                                         'original_all_rmse', 'teacher_original_all_r2', 'teacher_original_all_mae',
                                         'teacher_original_all_rmse', 'original_sto_r2', 'original_sto_mae',
                                         'original_sto_rmse', 'teacher_original_sto_r2', 'teacher_original_sto_mae',
                                         'teacher_original_sto_rmse', 'original_non_sto_r2', 'original_non_sto_mae',
                                         'original_non_sto_rmse', 'teacher_original_non_sto_r2',
                                         'teacher_original_non_sto_mae', 'teacher_original_non_sto_rmse',
                                         'total_epochs', 'best_r2', 'best_loss', 'final_loss', 'final_r2',
                                         'training_time_minutes']}
            })
            continue

        # 处理成功的结果（原有代码保持不变）
        try:
            results_data = result['results']
            evaluation_results = results_data.get('evaluation_results', {})
            history = results_data.get('history', {})

            # 提取训练历史信息
            total_epochs = len(history.get('loss', []))
            best_r2 = max(history.get('r2', [0])) if history.get('r2') else 0
            best_loss = min(history.get('loss', [float('inf')])) if history.get('loss') else float('inf')
            final_loss = history.get('loss', [])[-1] if history.get('loss') else None
            final_r2 = history.get('r2', [])[-1] if history.get('r2') else None

            # 计算训练时间（如果有时间信息）
            training_time = None
            if 'training_start_time' in results_data and 'training_end_time' in results_data:
                time_diff = results_data['training_end_time'] - results_data['training_start_time']
                training_time = time_diff.total_seconds() / 60  # 转换为分钟

            # 基础性能指标
            performance_data.append({
                'solution': solution_name,
                'status': 'success',
                'error': None,
                # 实验数据评估结果
                'experiment_data_r2': evaluation_results.get('experiment_data_r2'),
                'experiment_data_mae': evaluation_results.get('experiment_data_mae'),
                'experiment_data_rmse': evaluation_results.get('experiment_data_rmse'),
                'teacher_experiment_data_r2': evaluation_results.get('teacher_experiment_data_r2'),
                'teacher_experiment_data_mae': evaluation_results.get('teacher_experiment_data_mae'),
                'teacher_experiment_data_rmse': evaluation_results.get('teacher_experiment_data_rmse'),
                # 原始数据全部评估结果
                'original_all_r2': evaluation_results.get('original_all_r2'),
                'original_all_mae': evaluation_results.get('original_all_mae'),
                'original_all_rmse': evaluation_results.get('original_all_rmse'),
                'teacher_original_all_r2': evaluation_results.get('teacher_original_all_r2'),
                'teacher_original_all_mae': evaluation_results.get('teacher_original_all_mae'),
                'teacher_original_all_rmse': evaluation_results.get('teacher_original_all_rmse'),
                # 原始数据STO子集评估结果
                'original_sto_r2': evaluation_results.get('original_sto_r2'),
                'original_sto_mae': evaluation_results.get('original_sto_mae'),
                'original_sto_rmse': evaluation_results.get('original_sto_rmse'),
                'teacher_original_sto_r2': evaluation_results.get('teacher_original_sto_r2'),
                'teacher_original_sto_mae': evaluation_results.get('teacher_original_sto_mae'),
                'teacher_original_sto_rmse': evaluation_results.get('teacher_original_sto_rmse'),
                # 原始数据non-STO子集评估结果
                'original_non_sto_r2': evaluation_results.get('original_non_sto_r2'),
                'original_non_sto_mae': evaluation_results.get('original_non_sto_mae'),
                'original_non_sto_rmse': evaluation_results.get('original_non_sto_rmse'),
                'teacher_original_non_sto_r2': evaluation_results.get('teacher_original_non_sto_r2'),
                'teacher_original_non_sto_mae': evaluation_results.get('teacher_original_non_sto_mae'),
                'teacher_original_non_sto_rmse': evaluation_results.get('teacher_original_non_sto_rmse'),
                # 训练信息
                'total_epochs': total_epochs,
                'best_r2': best_r2,
                'best_loss': best_loss,
                'final_loss': final_loss,
                'final_r2': final_r2,
                'training_time_minutes': training_time
            })

        except Exception as e:
            logger.error(f"Error processing results for {solution_name}: {e}")
            performance_data.append({
                'solution': solution_name,
                'status': 'error_processing',
                'error': str(e),
                # 所有评估指标设为None
                'experiment_data_r2': None,
                'experiment_data_mae': None,
                'experiment_data_rmse': None,
                'teacher_experiment_data_r2': None,
                'teacher_experiment_data_mae': None,
                'teacher_experiment_data_rmse': None,
                'original_all_r2': None,
                'original_all_mae': None,
                'original_all_rmse': None,
                'teacher_original_all_r2': None,
                'teacher_original_all_mae': None,
                'teacher_original_all_rmse': None,
                'original_sto_r2': None,
                'original_sto_mae': None,
                'original_sto_rmse': None,
                'teacher_original_sto_r2': None,
                'teacher_original_sto_mae': None,
                'teacher_original_sto_rmse': None,
                'original_non_sto_r2': None,
                'original_non_sto_mae': None,
                'original_non_sto_rmse': None,
                'teacher_original_non_sto_r2': None,
                'teacher_original_non_sto_mae': None,
                'teacher_original_non_sto_rmse': None,
                'total_epochs': None,
                'best_r2': None,
                'best_loss': None,
                'final_loss': None,
                'final_r2': None,
                'training_time_minutes': None
            })

    if not performance_data:
        logger.warning("No results to generate performance report")
        return

    # 创建DataFrame并排序（按实验数据R²降序）
    df_performance = pd.DataFrame(performance_data)

    # 对成功的结果进行排序，失败的结果放在最后
    success_mask = df_performance['status'] == 'success'
    if success_mask.any():
        success_df = df_performance[success_mask].sort_values('experiment_data_r2', ascending=False)
        other_df = df_performance[~success_mask]
        df_performance = pd.concat([success_df, other_df], ignore_index=True)

    # 保存基础性能对比CSV
    csv_path = output_dir / "performance_comparison.csv"
    df_performance.to_csv(csv_path, index=False, encoding='utf-8')

    # 生成统计摘要
    summary_stats = generate_summary_statistics(df_performance, output_dir)

    # 生成可视化报告
    generate_visual_report(df_performance, output_dir)

    logger.info(f"Performance comparison report saved to {csv_path}")
    logger.info(f"Summary statistics saved to {output_dir / 'summary_statistics.txt'}")

    return df_performance, summary_stats


def generate_summary_statistics(df_performance: pd.DataFrame, output_dir: Path) -> dict:
    """生成统计摘要"""
    summary = {
        'total_models': len(df_performance),
        'successful_models': len(df_performance[df_performance['status'] == 'success']),
        'failed_models': len(df_performance[df_performance['status'] == 'failed']),
        'error_models': len(df_performance[df_performance['status'] == 'error_processing'])
    }

    # 成功模型的统计
    success_df = df_performance[df_performance['status'] == 'success']
    if not success_df.empty:
        # 实验数据统计
        summary.update({
            'experiment_data_stats': {
                'student_r2_mean': success_df['experiment_data_r2'].mean(),
                'student_r2_std': success_df['experiment_data_r2'].std(),
                'student_r2_min': success_df['experiment_data_r2'].min(),
                'student_r2_max': success_df['experiment_data_r2'].max(),
                'teacher_r2_mean': success_df['teacher_experiment_data_r2'].mean(),
                'teacher_r2_std': success_df['teacher_experiment_data_r2'].std(),
                'teacher_r2_min': success_df['teacher_experiment_data_r2'].min(),
                'teacher_r2_max': success_df['teacher_experiment_data_r2'].max(),
            },
            'training_stats': {
                'mean_epochs': success_df['total_epochs'].mean(),
                'mean_training_time': success_df['training_time_minutes'].mean(),
                'best_experiment_r2_model': success_df.loc[success_df['experiment_data_r2'].idxmax(), 'solution'],
                'best_experiment_r2_score': success_df['experiment_data_r2'].max(),
                'best_teacher_experiment_r2_score': success_df['teacher_experiment_data_r2'].max()
            }
        })

    summary_path = output_dir / "summary_statistics.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Batch Fine-tuning Summary Statistics\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total models processed: {summary['total_models']}\n")
        f.write(f"Successful: {summary['successful_models']}\n")
        f.write(f"Failed: {summary['failed_models']}\n")
        f.write(f"Processing errors: {summary['error_models']}\n\n")

        if 'r2_new_stats' in summary:
            stats = summary['r2_new_stats']
            f.write("New Data R² Statistics (successful models):\n")
            f.write(f"  Mean: {stats['mean']:.6f}\n")
            f.write(f"  Std: {stats['std']:.6f}\n")
            f.write(f"  Min: {stats['min']:.6f}\n")
            f.write(f"  Max: {stats['max']:.6f}\n")
            f.write(f"  Median: {stats['median']:.6f}\n\n")

            f.write(f"Best model: {summary['training_stats']['best_r2_model']}\n")
            f.write(f"Best R² score: {summary['training_stats']['best_r2_score']:.6f}\n")
            f.write(f"Average training epochs: {summary['training_stats']['mean_epochs']:.1f}\n")
            if summary['training_stats']['mean_training_time']:
                f.write(f"Average training time: {summary['training_stats']['mean_training_time']:.1f} minutes\n")


def batch_fine_tune_trial_solutions(
        trial_dir: str,
        new_data_path_raw: str,
        base_features: List[str],
        target_column: str,
        config: Optional[FineTuneConfig] = None,
        base_output_dir: str = "./batch_finetuned_models",
        skip_existing: bool = True  # 新增参数：是否跳过已存在的文件夹
):
    """
    批量微调trial_solutions目录下的所有solution_*模型

    Args:
        trial_dir: trial目录路径
        new_data_path_raw: 新数据文件路径
        base_features: 基础特征列表
        target_column: 目标列名
        config: 微调配置
        base_output_dir: 输出基础目录
        skip_existing: 是否跳过已存在且非空的输出文件夹，默认True
    """
    solution_pattern = os.path.join(trial_dir, "trial_*")
    solution_dirs = glob.glob(solution_pattern)

    if not solution_dirs:
        logger.warning(f"No solution directories found matching pattern: {solution_pattern}")
        return {}

    logger.info(f"Found {len(solution_dirs)} solution directories")

    base_output_path = Path(base_output_dir)
    base_output_path.mkdir(parents=True, exist_ok=True)

    all_results = {}
    skipped_count = 0  # 统计跳过的数量

    for solution_dir in sorted(solution_dirs):
        solution_name = os.path.basename(solution_dir)
        logger.info(f"Processing {solution_name}")

        model_dir = os.path.join(solution_dir, "model")
        if not os.path.exists(model_dir):
            logger.warning(f"Model directory not found in {solution_name}: {model_dir}")
            continue

        output_dir = base_output_path / solution_name

        # 检查输出目录是否已存在且非空
        if skip_existing and output_dir.exists():
            # 检查关键文件是否存在
            model_save_path = output_dir / "model"
            results_file = output_dir / "finetune_results.joblib"

            if model_save_path.exists() and results_file.exists():
                logger.info(f"Skipping {solution_name}: output directory already exists and contains model files")
                skipped_count += 1

                # 尝试加载已有结果
                try:
                    existing_results = joblib.load(str(results_file))
                    all_results[solution_name] = {
                        'status': 'skipped',
                        'results': existing_results,
                        'output_dir': output_dir,
                        'model_save_path': model_save_path
                    }
                except Exception as e:
                    logger.warning(f"Failed to load existing results for {solution_name}: {e}")
                    all_results[solution_name] = {
                        'status': 'skipped',
                        'output_dir': output_dir,
                        'model_save_path': model_save_path
                    }

                continue

        try:
            # 运行微调实验（会自动保存模型）
            finetuned_model, fine_tuner, results = run_fine_tuning_experiment(
                model_path=model_dir,
                new_data_path_raw=new_data_path_raw,
                base_features=base_features,
                target_column=target_column,
                config=config,
                output_dir=str(output_dir)
            )

            all_results[solution_name] = {
                'model': finetuned_model,
                'fine_tuner': fine_tuner,
                'results': results,
                'output_dir': output_dir,
                'model_save_path': output_dir / "model",
                'status': 'success'
            }

            logger.info(f"Successfully fine-tuned and saved {solution_name}")

        except Exception as e:
            logger.error(f"Failed to fine-tune {solution_name}: {str(e)}")
            all_results[solution_name] = {
                'error': str(e),
                'output_dir': output_dir,
                'status': 'failed'
            }

    # 保存批量处理摘要
    summary_path = base_output_path / "batch_finetune_summary.joblib"
    joblib.dump(all_results, str(summary_path))

    # 生成性能对比报告
    generate_performance_report(all_results, base_output_path)

    # 打印统计信息
    logger.info("=" * 60)
    logger.info(f"Batch fine-tuning summary:")
    logger.info(f"Total solutions: {len(solution_dirs)}")
    logger.info(f"Skipped (already exists): {skipped_count}")
    logger.info(f"Processed: {len(all_results) - skipped_count}")
    logger.info("=" * 60)

    return all_results


def fine_tune_specific_model(
        model_dir: str,
        new_data_path_raw: str,
        base_features: List[str],
        target_column: str,
        config: Optional[FineTuneConfig] = None,
        output_dir: str = "./specific_finetuned_model"
):
    """
    对指定的单个模型文件夹进行微调

    Args:
        model_dir: 模型文件夹路径
        new_data_path_raw: 新数据文件路径
        base_features: 基础特征列表
        target_column: 目标列名
        config: 微调配置
        output_dir: 输出目录
    """
    # 检查模型目录是否存在
    if not os.path.exists(model_dir):
        raise ValueError(f"模型目录不存在: {model_dir}")

    # 获取模型名称（用于输出目录命名）
    model_name = os.path.basename(model_dir.rstrip('/\\'))
    if not model_name:
        model_name = "specific_model"

    # 设置输出目录
    output_dir = Path(output_dir) / f"finetuned_{model_name}"

    logger.info(f"开始微调指定模型: {model_name}")
    logger.info(f"模型路径: {model_dir}")
    logger.info(f"输出目录: {output_dir}")

    try:
        # 运行微调实验
        finetuned_model, fine_tuner, results = run_fine_tuning_experiment(
            model_path=model_dir,
            new_data_path_raw=new_data_path_raw,
            base_features=base_features,
            target_column=target_column,
            config=config,
            output_dir=str(output_dir)
        )

        logger.info(f"模型 {model_name} 微调完成!")
        logger.info(f"实验结果保存至: {output_dir}")

        return {
            'model': finetuned_model,
            'fine_tuner': fine_tuner,
            'results': results,
            'output_dir': output_dir,
            'model_name': model_name
        }

    except Exception as e:
        logger.error(f"微调模型 {model_name} 失败: {str(e)}")
        raise


def generate_visual_report(df_performance: pd.DataFrame, output_dir: Path):
    """生成可视化报告"""
    try:
        # 只处理成功的模型
        success_df = df_performance[df_performance['status'] == 'success']
        if success_df.empty:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. R²分布直方图
        axes[0, 0].hist(success_df['r2_new'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('R² Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of R² Scores')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. R²排名图
        sorted_df = success_df.sort_values('r2_new', ascending=True)
        axes[0, 1].barh(range(len(sorted_df)), sorted_df['r2_new'], color='lightgreen', alpha=0.7)
        axes[0, 1].set_yticks(range(len(sorted_df)))
        axes[0, 1].set_yticklabels(sorted_df['solution'], fontsize=8)
        axes[0, 1].set_xlabel('R² Score')
        axes[0, 1].set_title('Model Performance Ranking')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 训练轮数 vs R²
        axes[1, 0].scatter(success_df['total_epochs'], success_df['r2_new'], alpha=0.6, color='coral')
        axes[1, 0].set_xlabel('Training Epochs')
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].set_title('Training Epochs vs Performance')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. MAE vs RMSE
        axes[1, 1].scatter(success_df['mae_new'], success_df['rmse_new'], alpha=0.6, color='purple')
        axes[1, 1].set_xlabel('MAE')
        axes[1, 1].set_ylabel('RMSE')
        axes[1, 1].set_title('MAE vs RMSE Correlation')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "performance_visualization.png", dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        logger.warning(f"Failed to generate visual report: {e}")


# 使用示例
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Fine-tune hybrid model')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['tradition', 'LLM'],
                        help='Mode: tradition or LLM')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['attention', 'series', 'uncertainty_1', 'uncertainty_2'],
                        help='Model type: attention, series, uncertainty_1, or uncertainty_2')
    parser.add_argument('--distillation_alpha', type=float, default=0.6,
                        help='Distillation alpha (default: 0.6)')
    parser.add_argument('--distillation_temperature', type=float, default=1.5,
                        help='Distillation temperature (default: 1.5)')

    args = parser.parse_args()

    # 动态导入模型
    model_path = f"./{args.mode}/XGB_BNN_{args.model_type}_hybrid_model"
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), model_path))
    from model import HybridModel

    # 配置参数
    TRIAL_DIR = f"{model_path}/pretrain/hyperparameter_tuning_results/all_trials"
    NEW_DATA_PATH_RAW = './data/sobol_samples_results.csv'
    BASE_FEATURES = ['Oxygen pressure', 'Laser energy density', 'Temperature', 'Frequency', 'Thickness']
    TARGET_COLUMN = 'RRR'
    BASE_OUTPUT_DIR = f"./{args.mode}/XGB_BNN_{args.model_type}_hybrid_model/fine-tune/batch_finetuned_models"

    # 创建微调配置
    finetune_config = SmallSampleFineTuneConfig(
        sto_weight_multiplier=1.5,
        xgb_fine_tune=True,
        bnn_epochs=1000,
        batch_size=4,
        learning_rate=5e-5,
        weight_decay=1e-3,

        # 使用命令行参数
        distillation_alpha=args.distillation_alpha,
        distillation_temperature=args.distillation_temperature,

        patience=200,
        min_epochs=300,
        warmup_epochs=150,
        l2_regularization=1e-4,
        weight_change_penalty=1e-4,
        verbose=True,
        use_data_augmentation=False
    )

    print("=" * 60)
    print(f"Starting fine-tuning with mode={args.mode}, model_type={args.model_type}")
    print(f"Distillation alpha: {args.distillation_alpha}")
    print(f"Distillation temperature: {args.distillation_temperature}")
    print("=" * 60)

    try:
        all_results = batch_fine_tune_trial_solutions(
            trial_dir=TRIAL_DIR,
            new_data_path_raw=NEW_DATA_PATH_RAW,
            base_features=BASE_FEATURES,
            target_column=TARGET_COLUMN,
            config=finetune_config,
            base_output_dir=BASE_OUTPUT_DIR,
            skip_existing=True
        )

        print("=" * 60)
        print("Batch fine-tuning completed!")
        print("=" * 60)

        # 统计成功和失败的数量
        successful = sum(1 for result in all_results.values() if 'error' not in result)
        failed = len(all_results) - successful

        print(f"Total solutions processed: {len(all_results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Results saved to: {BASE_OUTPUT_DIR}")

        # 打印成功模型的保存路径
        for name, result in all_results.items():
            if 'error' not in result:
                print(f"Model {name} saved to: {result.get('model_save_path', 'Unknown')}")

        print("=" * 60)

    except Exception as e:
        logger.error(f"Batch fine-tuning failed: {str(e)}")
        raise
