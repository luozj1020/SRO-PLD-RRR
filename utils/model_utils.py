import ast

import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import random
import warnings
from sklearn.impute import SimpleImputer, KNNImputer
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging
from torch.utils.data import Dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plot parameters
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def uncertainty_based_filtering(
        X, y, model,
        uncertainty_threshold_percentile=85,
        min_samples_to_keep=200
):
    """Remove the most uncertain predictions while retaining enough samples."""
    if not model.bnn_model or not model.is_fitted:
        logger.warning(
            "Model or BNN not fitted, cannot perform uncertainty filtering. "
            "Returning original data."
        )
        return X, y

    prediction_input = (
        pd.DataFrame(X, columns=model.base_features)
        if isinstance(X, np.ndarray)
        else X
    )
    _, uncertainties = model.predict(prediction_input)
    threshold = np.percentile(uncertainties, uncertainty_threshold_percentile)

    keep_mask = uncertainties < threshold
    if np.sum(keep_mask) < min_samples_to_keep:
        indices_to_keep = np.argsort(uncertainties)[:min_samples_to_keep]
        keep_mask = np.zeros(len(uncertainties), dtype=bool)
        keep_mask[indices_to_keep] = True

    if isinstance(X, pd.DataFrame):
        filtered_X = X.iloc[keep_mask]
        filtered_y = y.iloc[keep_mask] if isinstance(y, pd.Series) else y[keep_mask]
    else:
        filtered_X = X[keep_mask]
        filtered_y = y[keep_mask]

    return filtered_X, filtered_y


class enable_dropout:
    """Temporarily enable dropout while preserving the model training state."""

    def __init__(self, model):
        self.model = model
        self.original_training = model.training

    def __enter__(self):
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.train(self.original_training)


SEQUENCE_FEATURE_COLUMNS = (
    'Temperature',
    'Laser energy density',
    'Oxygen pressure',
    'Frequency',
    'Thickness',
)


class OrdinalDataset(Dataset):
    """Load favourable and unfavourable ordinal condition sequences."""

    def __init__(self, processor, good_data_path, bad_data_path, base_features):
        self.processor = processor
        self.base_features = base_features
        try:
            self.good_data = pd.read_csv(good_data_path)
            self.bad_data = pd.read_csv(bad_data_path)
            self.good_sequences = self._preprocess_data(self.good_data)
            self.bad_sequences = self._preprocess_data(self.bad_data)
        except Exception as exc:
            logger.error("Sequence data loading failed: %s", exc)
            self.good_sequences = []
            self.bad_sequences = []

    def _preprocess_data(self, data):
        sequences = []
        for idx in range(len(data)):
            sequence = []
            for column in (f'condition_{i}' for i in range(1, 6)):
                try:
                    if column in data.columns and not pd.isna(data[column].iloc[idx]):
                        condition_values = np.array(
                            ast.literal_eval(data[column].iloc[idx]), dtype=np.float32
                        )
                        if len(condition_values) != len(SEQUENCE_FEATURE_COLUMNS):
                            raise ValueError(
                                f"Expected {len(SEQUENCE_FEATURE_COLUMNS)} sequence features, "
                                f"got {len(condition_values)}"
                            )
                        condition_df = pd.DataFrame(
                            [condition_values], columns=SEQUENCE_FEATURE_COLUMNS
                        )[self.base_features]
                        sequence.append(self.processor.transform(condition_df)[0])
                    else:
                        sequence.append(
                            np.zeros(len(self.base_features), dtype=np.float32)
                        )
                except Exception as exc:
                    logger.warning("Error processing condition: %s", exc)
                    sequence.append(np.zeros(len(self.base_features), dtype=np.float32))
            sequences.append(np.stack(sequence))
        return sequences

    def __len__(self):
        return len(self.good_sequences) + len(self.bad_sequences)

    def __getitem__(self, idx):
        if idx < len(self.good_sequences):
            return torch.FloatTensor(self.good_sequences[idx]), 0
        return torch.FloatTensor(
            self.bad_sequences[idx - len(self.good_sequences)]
        ), 1


def sequence_ranking_loss(bnn_model, residual_stats, sequences, labels, config):
    """Calculate the differentiable ordinal contrast loss."""
    batch_size, seq_len, feature_dim = sequences.shape
    flat_sequences = sequences.reshape(-1, feature_dim)

    device = next(bnn_model.parameters()).device
    flat_sequences_device = flat_sequences.to(device)
    sequence_mask = torch.ones_like(flat_sequences_device, device=device)

    bnn_residuals = bnn_model(flat_sequences_device, sequence_mask).reshape(
        batch_size, seq_len
    )
    residual_std = torch.as_tensor(
        residual_stats['std'], dtype=bnn_residuals.dtype, device=device
    )
    residual_mean = torch.as_tensor(
        residual_stats['mean'], dtype=bnn_residuals.dtype, device=device
    )
    final_preds = bnn_residuals * residual_std + residual_mean

    intra_good_loss = torch.tensor(0.0, device=sequences.device)
    intra_bad_loss = torch.tensor(0.0, device=sequences.device)
    inter_loss = torch.tensor(0.0, device=sequences.device)

    good_seqs = final_preds[labels == 0]
    bad_seqs = final_preds[labels == 1]

    if good_seqs.numel() > 0 and good_seqs.shape[1] > 1:
        good_diffs = good_seqs[:, 1:] - good_seqs[:, :-1]
        intra_good_loss = (
            torch.sum(torch.relu(good_diffs))
            * config.sequence_loss['intra_good_penalty']
        )

    if bad_seqs.numel() > 0 and bad_seqs.shape[1] > 1:
        bad_diffs = bad_seqs[:, :-1] - bad_seqs[:, 1:]
        intra_bad_loss = (
            torch.sum(torch.relu(bad_diffs))
            * config.sequence_loss['intra_bad_penalty']
        )

    if good_seqs.numel() > 0 and bad_seqs.numel() > 0:
        min_good = good_seqs.min(dim=1).values
        max_bad = bad_seqs.max(dim=1).values
        inter_diff = max_bad.unsqueeze(1) - min_good.unsqueeze(0)
        inter_loss = (
            torch.sum(torch.relu(inter_diff))
            * config.sequence_loss['inter_penalty']
        )

    total_loss = intra_good_loss + intra_bad_loss + inter_loss
    return total_loss, intra_good_loss, intra_bad_loss, inter_loss


def plot_sequence_loss(history: Dict, save_path: Optional[str] = None):
    """可视化序列对比损失变化"""
    if not history.get('sequence_loss_enabled', False):
        logger.info("Sequence loss not enabled, skipping visualization")
        return None

    # 检查序列损失数据是否存在
    if not history.get('sequence_loss') or len(history['sequence_loss']) == 0:
        logger.warning("Sequence loss data not available")
        return None

    plt.figure(figsize=(12, 8))

    # 总序列损失
    plt.subplot(2, 1, 1)
    plt.plot(history['sequence_loss'], label='Total Sequence Loss', color='blue')
    plt.title('Sequence Ranking Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 各分量损失
    plt.subplot(2, 1, 2)
    if history.get('intra_good_loss'):
        plt.plot(history['intra_good_loss'], label='Intra-Good Loss', color='green')
    if history.get('intra_bad_loss'):
        plt.plot(history['intra_bad_loss'], label='Intra-Bad Loss', color='red')
    if history.get('inter_loss'):
        plt.plot(history['inter_loss'], label='Inter-Sequence Loss', color='purple')

    plt.title('Sequence Loss Components')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return plt.gcf()


class ModelVisualizer:
    """Visualization tools for model evaluation"""

    @staticmethod
    def plot_training_history(history: Dict, save_path: Optional[str] = None,
                              feature_names: Optional[List[str]] = None):
        """Plot training history metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Training History', fontsize=16)

        # Loss curve
        axes[0, 0].plot(history['loss'], label='Training Loss', color='blue')
        if 'val_loss' in history and history['val_loss']:
            axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].annotate(f'Final Loss: {history["loss"][-1]:.4f}',
                            xy=(len(history['loss']), history['loss'][-1]),
                            xytext=(-40, 20), textcoords='offset points',
                            arrowprops=dict(arrowstyle='->'))
        axes[0, 0].set_title('Loss Curve')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # R² curve
        axes[0, 1].plot(history['r2'], label='Training $R^2$', color='green')
        if 'val_r2' in history and history['val_r2']:
            axes[0, 1].plot(history['val_r2'], label='Validation $R^2$', color='orange')
        axes[0, 1].annotate(f'Final R²: {history["r2"][-1]:.4f}',
                            xy=(len(history['r2']), history['r2'][-1]),
                            xytext=(-40, 20), textcoords='offset points',
                            arrowprops=dict(arrowstyle='->'))
        axes[0, 1].set_title('$R^2$ Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('$R^2$')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Learning rate schedule
        axes[1, 0].plot(history['learning_rates'], color='purple')
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)

        # Feature importance
        if history.get('feature_importance'):
            latest_importance = history['feature_importance'][-1]['xgb']

            # 修改这里：使用特征名称而不是索引
            if feature_names is not None and len(feature_names) == len(latest_importance):
                # 使用特征名称作为x轴标签
                x_positions = range(len(feature_names))
                axes[1, 1].bar(x_positions, latest_importance)
                axes[1, 1].set_xticks(x_positions)
                axes[1, 1].set_xticklabels(feature_names, rotation=45, ha='right')
                axes[1, 1].set_title('Feature Importance (XGBoost)')
                axes[1, 1].set_xlabel('Feature Name')
            else:
                # 回退到原来的实现
                axes[1, 1].bar(range(len(latest_importance)), latest_importance)
                axes[1, 1].set_title('Feature Importance (XGBoost)')
                axes[1, 1].set_xlabel('Feature Index')

            axes[1, 1].set_ylabel('Importance')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].set_title('Feature Importance Not Available')
            axes[1, 1].set_axis_off()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

    @staticmethod
    def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                         uncertainty: Optional[np.ndarray] = None,
                         full_r2: Optional[float] = None,
                         sto_r2: Optional[float] = None,
                         save_path: Optional[str] = None,
                         sto_flags: Optional[np.ndarray] = None,
                         sample_weights: Optional[np.ndarray] = None,
                         augmentation_flags: Optional[np.ndarray] = None):  # 新增参数
        """Plot predicted vs true values with residuals, showing both full and STO R²"""

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # 处理STO标识
        has_sto = sto_flags is not None and np.any(sto_flags == 1)
        has_augmentation = augmentation_flags is not None  # 新增

        # 定义颜色
        non_sto_color = 'blue'
        sto_color = 'orange'
        augmented_color = 'lightgreen'  # 新增:增强数据颜色

        # 修复alpha计算，确保在[0,1]范围内
        if sample_weights is not None:
            # 归一化权重到[0,1]范围
            if np.max(sample_weights) > 0:
                normalized_weights = sample_weights / np.max(sample_weights)
            else:
                normalized_weights = sample_weights

            # 计算alpha并确保在[0.1, 1.0]范围内
            alphas = 0.1 + 0.9 * (normalized_weights ** 0.5)
            alphas = np.clip(alphas, 0.1, 1.0)  # 确保alpha在有效范围内
        else:
            alphas = np.ones(len(y_true)) * 0.6

        # Prediction vs actual scatter plot
        if has_augmentation:
            # 分离原始数据和增强数据
            original_mask = augmentation_flags == 0
            augmented_mask = augmentation_flags == 1

            if has_sto:
                # 四种类型:原始非STO、原始STO、增强非STO、增强STO
                original_non_sto = original_mask & (sto_flags == 0)
                original_sto = original_mask & (sto_flags == 1)
                augmented_non_sto = augmented_mask & (sto_flags == 0)
                augmented_sto = augmented_mask & (sto_flags == 1)

                # 绘制原始非STO点
                if np.sum(original_non_sto) > 0:
                    axes[0].scatter(y_true[original_non_sto], y_pred[original_non_sto],
                                    alpha=alphas[original_non_sto], color=non_sto_color,
                                    label='Original Non-STO', s=40, marker='o')

                # 绘制原始STO点
                if np.sum(original_sto) > 0:
                    axes[0].scatter(y_true[original_sto], y_pred[original_sto],
                                    alpha=alphas[original_sto], color=sto_color,
                                    marker='s', s=60, label='Original STO')

                # 绘制增强非STO点
                if np.sum(augmented_non_sto) > 0:
                    axes[0].scatter(y_true[augmented_non_sto], y_pred[augmented_non_sto],
                                    alpha=alphas[augmented_non_sto] * 0.7, color=augmented_color,
                                    label='Augmented Non-STO', s=30, marker='^')

                # 绘制增强STO点
                if np.sum(augmented_sto) > 0:
                    axes[0].scatter(y_true[augmented_sto], y_pred[augmented_sto],
                                    alpha=alphas[augmented_sto] * 0.7, color=augmented_color,
                                    marker='D', s=50, label='Augmented STO',
                                    edgecolors=sto_color, linewidths=1.5)
            else:
                # 仅区分原始和增强
                if np.sum(original_mask) > 0:
                    axes[0].scatter(y_true[original_mask], y_pred[original_mask],
                                    alpha=alphas[original_mask], color=non_sto_color,
                                    label='Original Data', s=40, marker='o')

                if np.sum(augmented_mask) > 0:
                    axes[0].scatter(y_true[augmented_mask], y_pred[augmented_mask],
                                    alpha=alphas[augmented_mask] * 0.7, color=augmented_color,
                                    label='Augmented Data', s=30, marker='^')

        elif has_sto:
            # 原有逻辑:仅区分STO
            non_sto_mask = sto_flags == 0
            sto_mask = sto_flags == 1

            axes[0].scatter(y_true[non_sto_mask], y_pred[non_sto_mask],
                            alpha=alphas[non_sto_mask], color=non_sto_color,
                            label='Non-STO', s=40)
            axes[0].scatter(y_true[sto_mask], y_pred[sto_mask],
                            alpha=alphas[sto_mask], color=sto_color,
                            marker='s', s=60, label='STO')
        else:
            # 原有逻辑:无分类
            axes[0].scatter(y_true, y_pred, alpha=alphas, color=non_sto_color, s=40)

        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        axes[0].set_xlabel('True Values')
        axes[0].set_ylabel('Predicted Values')
        title = f'Predictions vs True Values'
        if full_r2 is not None:
            title += f'\nFull $R^2$ = {full_r2:.4f}'
        if sto_r2 is not None:
            title += f' | STO $R^2$ = {sto_r2:.4f}'
        if has_augmentation:
            orig_count = np.sum(augmentation_flags == 0)
            aug_count = np.sum(augmentation_flags == 1)
            title += f'\n(Original: {orig_count}, Augmented: {aug_count})'
        axes[0].set_title(title)
        axes[0].grid(True)
        axes[0].legend()

        # Residual plot - 类似处理
        residuals = y_true - y_pred
        if has_augmentation:
            original_mask = augmentation_flags == 0
            augmented_mask = augmentation_flags == 1

            if has_sto:
                original_non_sto = original_mask & (sto_flags == 0)
                original_sto = original_mask & (sto_flags == 1)
                augmented_non_sto = augmented_mask & (sto_flags == 0)
                augmented_sto = augmented_mask & (sto_flags == 1)

                if np.sum(original_non_sto) > 0:
                    axes[1].scatter(y_pred[original_non_sto], residuals[original_non_sto],
                                    alpha=alphas[original_non_sto], color=non_sto_color,
                                    label='Original Non-STO', s=40)
                if np.sum(original_sto) > 0:
                    axes[1].scatter(y_pred[original_sto], residuals[original_sto],
                                    alpha=alphas[original_sto], color=sto_color,
                                    marker='s', s=60, label='Original STO')
                if np.sum(augmented_non_sto) > 0:
                    axes[1].scatter(y_pred[augmented_non_sto], residuals[augmented_non_sto],
                                    alpha=alphas[augmented_non_sto] * 0.7, color=augmented_color,
                                    label='Augmented Non-STO', s=30, marker='^')
                if np.sum(augmented_sto) > 0:
                    axes[1].scatter(y_pred[augmented_sto], residuals[augmented_sto],
                                    alpha=alphas[augmented_sto] * 0.7, color=augmented_color,
                                    marker='D', s=50, label='Augmented STO',
                                    edgecolors=sto_color, linewidths=1.5)
            else:
                if np.sum(original_mask) > 0:
                    axes[1].scatter(y_pred[original_mask], residuals[original_mask],
                                    alpha=alphas[original_mask], color=non_sto_color,
                                    label='Original Data', s=40)
                if np.sum(augmented_mask) > 0:
                    axes[1].scatter(y_pred[augmented_mask], residuals[augmented_mask],
                                    alpha=alphas[augmented_mask] * 0.7, color=augmented_color,
                                    label='Augmented Data', s=30, marker='^')

        elif has_sto:
            axes[1].scatter(y_pred[non_sto_mask], residuals[non_sto_mask],
                            alpha=alphas[non_sto_mask], color=non_sto_color,
                            label='Non-STO', s=40)
            axes[1].scatter(y_pred[sto_mask], residuals[sto_mask],
                            alpha=alphas[sto_mask], color=sto_color,
                            marker='s', s=60, label='STO')
        else:
            axes[1].scatter(y_pred, residuals, alpha=alphas, color='green', s=40)

        axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.8)
        axes[1].set_xlabel('Predicted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot')
        axes[1].grid(True)
        axes[1].legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

    @staticmethod
    def plot_feature_distribution_2d(model, processor, data: pd.DataFrame,
                                     x_feature: str, y_feature: str,
                                     fixed_values: Optional[Dict] = None,
                                     resolution: int = 50,
                                     save_path: Optional[str] = None):
        """Generate 2D feature distribution contour plots using original scale"""

        # 定义原始尺度边界（从BOUND_1）
        ORIGINAL_BOUNDS = {
            'Oxygen pressure': (1e-4, 1),  # 注意：这是原始值，不是对数尺度
            'Laser energy density': (0.5, 5),
            'Temperature': (500, 900),
            'Frequency': (0, 11),
            'Thickness': (0, 200)
        }

        def get_feature_range(feature, values):
            """使用原始尺度边界而不是标准化后的范围"""
            if feature in ORIGINAL_BOUNDS:
                return ORIGINAL_BOUNDS[feature]
            return values.min(), values.max()

        # 确定轴范围（使用原始尺度）
        x_min, x_max = get_feature_range(x_feature, data[x_feature])
        y_min, y_max = get_feature_range(y_feature, data[y_feature])

        # 生成网格点（使用原始尺度）
        if x_feature == 'Oxygen pressure':
            # 对于氧压，使用对数空间
            x_range = np.logspace(np.log10(x_min), np.log10(x_max), resolution)
        else:
            x_range = np.linspace(x_min, x_max, resolution)

        if y_feature == 'Oxygen pressure':
            # 对于氧压，使用对数空间
            y_range = np.logspace(np.log10(y_min), np.log10(y_max), resolution)
        else:
            y_range = np.linspace(y_min, y_max, resolution)

        # 创建网格
        X_grid, Y_grid = np.meshgrid(x_range, y_range)

        # 准备预测数据
        grid_data = []
        for i in range(resolution):
            for j in range(resolution):
                point = {x_feature: X_grid[i, j], y_feature: Y_grid[i, j]}
                if fixed_values:
                    point.update(fixed_values)
                else:
                    # 使用原始值的中位数而不是标准化值
                    for col in processor.base_features:
                        if col not in [x_feature, y_feature]:
                            # 使用原始数据的中位数
                            point[col] = data[col].median()
                grid_data.append(point)

        grid_df = pd.DataFrame(grid_data)

        # 确保所有基础特征都存在
        for feature in processor.base_features:
            if feature not in grid_df.columns:
                grid_df[feature] = data[feature].median()

        # 使用处理器转换数据（标准化）
        try:
            X_grid_processed = processor.transform(grid_df[processor.base_features])
        except Exception as e:
            logger.error(f"Error processing grid data: {e}")
            raise

        # 获取原始数据用于预测
        X_raw_grid = grid_df[processor.base_features].values

        # 预测 - 保留您想要的代码部分，但进行适配
        try:
            # 检查模型类型，适配不同的预测接口
            if hasattr(model, 'xgb_model'):  # XGBoost模型
                # XGBoost只需要一个输入参数
                predictions = model.predict(X_raw_grid)
                uncertainties = np.zeros_like(predictions)  # XGBoost没有不确定性
            elif hasattr(model, 'bnn_model'):  # BNN模型
                predictions, uncertainties = model.predict(X_raw_grid, X_grid_processed, n_samples=30)
            else:  # 混合模型
                # 混合模型需要两个输入参数
                predictions, uncertainties = model.predict(X_raw_grid, X_grid_processed)

            # 保留您想要的代码部分
            pred_grid = predictions.reshape(resolution, resolution)
            uncertainty_grid = uncertainties.reshape(resolution, resolution)
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            pred_grid = np.zeros((resolution, resolution))
            uncertainty_grid = np.zeros((resolution, resolution))

        # 创建图表
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # 预测等高线图
        im1 = axes[0].contourf(X_grid, Y_grid, pred_grid, levels=20, cmap='viridis')
        if x_feature == 'Oxygen pressure':
            axes[0].set_xscale('log')
        if y_feature == 'Oxygen pressure':
            axes[0].set_yscale('log')
        axes[0].set_xlabel(x_feature)
        axes[0].set_ylabel(y_feature)
        axes[0].set_title(f'Predicted Values: {x_feature} vs {y_feature}')
        plt.colorbar(im1, ax=axes[0])

        # 不确定性等高线图
        im2 = axes[1].contourf(X_grid, Y_grid, uncertainty_grid, levels=20, cmap='Reds')
        if x_feature == 'Oxygen pressure':
            axes[1].set_xscale('log')
        if y_feature == 'Oxygen pressure':
            axes[1].set_yscale('log')
        axes[1].set_xlabel(x_feature)
        axes[1].set_ylabel(y_feature)
        axes[1].set_title(f'Uncertainty: {x_feature} vs {y_feature}')
        plt.colorbar(im2, ax=axes[1])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig


def weighted_r2_score_numpy(y_true: np.ndarray, y_pred: np.ndarray,
                            sample_weights: Optional[np.ndarray] = None) -> float:
    """
    Calculate weighted R2 score using NumPy.
    If sample_weights is None, it's equivalent to the standard R2 score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights)
        if len(sample_weights) != len(y_true):
            raise ValueError("Length of sample_weights must match length of y_true.")

    # Calculate weighted mean of y_true
    if sample_weights is not None:
        # Avoid division by zero if all weights are zero (shouldn't happen in practice)
        sum_weights = np.sum(sample_weights)
        if sum_weights == 0:
            # Fallback to unweighted mean if weights sum to zero
            y_wmean = np.average(y_true)
        else:
            y_wmean = np.average(y_true, weights=sample_weights)
    else:
        y_wmean = np.average(y_true)

    # Calculate weighted sums of squares
    if sample_weights is not None:
        ss_res = np.sum(sample_weights * (y_true - y_pred) ** 2)
        ss_tot = np.sum(sample_weights * (y_true - y_wmean) ** 2)
        # Avoid division by zero
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
    else:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_wmean) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0

    return 1.0 - (ss_res / ss_tot)


def analyze_model_performance(model, X: np.ndarray, y: np.ndarray,
                              feature_names: Optional[List[str]] = None,
                              sto_flags: Optional[np.ndarray] = None,
                              sample_weights: Optional[np.ndarray] = None,
                              X_processed: Optional[np.ndarray] = None) -> Dict:
    """Calculate model performance metrics, optionally using sample weights"""
    # 确保sto_flags不为None且长度匹配
    if sto_flags is None:
        sto_flags = np.zeros(len(y), dtype=int)
    elif len(sto_flags) != len(y):
        sto_flags = np.zeros(len(y), dtype=int)

    # 处理样本权重
    effective_sample_weights = None
    if sample_weights is not None:
        if len(sample_weights) != len(y):
            logger.warning("Length of sample_weights does not match y. Ignoring weights for performance analysis.")
        else:
            effective_sample_weights = sample_weights

    # 修改预测调用 - 检查模型是否需要两个输入
    if hasattr(model, 'requires_two_inputs') and model.requires_two_inputs:
        if X_processed is None:
            raise ValueError("This model requires processed features, but X_processed was not provided.")
        pred_mean, pred_std = model.predict(X, X_processed)
    else:
        pred_mean, pred_std = model.predict(X)

    # Core metrics - 使用加权R²计算
    full_r2 = weighted_r2_score_numpy(y, pred_mean, effective_sample_weights)
    full_mae = mean_absolute_error(y, pred_mean)  # MAE 通常不加权，如果需要可自定义
    full_rmse = np.sqrt(mean_squared_error(y, pred_mean))  # RMSE 通常不加权，如果需要可自定义
    # 如果提供了权重，计算加权的 MAE 和 RMSE
    if effective_sample_weights is not None:
        # 加权 MAE
        full_mae = np.average(np.abs(y - pred_mean), weights=effective_sample_weights)
        # 加权 RMSE
        full_rmse = np.sqrt(np.average((y - pred_mean) ** 2, weights=effective_sample_weights))

    full_performance = {
        'r2_score': full_r2,
        'mae': full_mae,
        'rmse': full_rmse
    }

    # STO样本性能 (如果存在STO样本)
    sto_performance = None
    if sto_flags is not None and np.any(sto_flags == 1):
        sto_indices = np.where(sto_flags == 1)[0]
        sto_y = y[sto_indices]
        sto_pred_mean = pred_mean[sto_indices]
        sto_sample_weights = effective_sample_weights[sto_indices] if effective_sample_weights is not None else None

        # STO R2 (使用加权或非加权)
        sto_r2 = weighted_r2_score_numpy(sto_y, sto_pred_mean, sto_sample_weights)
        # STO MAE 和 RMSE
        sto_mae = mean_absolute_error(sto_y, sto_pred_mean)
        sto_rmse = np.sqrt(mean_squared_error(sto_y, sto_pred_mean))
        if sto_sample_weights is not None:
            sto_mae = np.average(np.abs(sto_y - sto_pred_mean), weights=sto_sample_weights)
            sto_rmse = np.sqrt(np.average((sto_y - sto_pred_mean) ** 2, weights=sto_sample_weights))

        sto_performance = {
            'r2_score': sto_r2,
            'mae': sto_mae,
            'rmse': sto_rmse,
            'num_samples': len(sto_indices)
        }

    # Uncertainty metrics
    mean_uncertainty = np.mean(pred_std)
    uncertainty_std = np.std(pred_std)

    # Residual analysis
    residuals = y - pred_mean
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)

    # Feature importance (这部分通常不受样本权重直接影响，除非模型训练时权重改变了特征重要性)
    try:
        xgb_importance = model.xgb_model.feature_importances_
    except AttributeError:
        xgb_importance = np.zeros(X.shape[1])  # Fallback if not fitted or no XGBoost

    return {
        'full': full_performance,
        'sto': sto_performance,
        'mean_uncertainty': mean_uncertainty,
        'uncertainty_std': uncertainty_std,
        'residual_mean': residual_mean,
        'residual_std': residual_std,
        'feature_importance': xgb_importance.tolist(),
        'feature_names': feature_names or [f'feature_{i}' for i in range(len(xgb_importance))],
        'sto_flags': sto_flags,
        'sample_weights_used': effective_sample_weights is not None  # 新增信息
    }


def create_experiment_report(results: Dict, output_path: Path):
    """Generate experiment report in Markdown format"""

    def format_value(value, default='N/A'):
        """安全格式化数值，如果是字符串则直接返回"""
        if value is None or value == 'N/A':
            return default
        try:
            return f"{value:.4f}"
        except (TypeError, ValueError):
            return str(value)

    report_content = f"""
# Hybrid Model Experiment Report

## Model Configuration
- XGBoost Parameters: {results['config'].xgb_params}
"""
    if results['config'].bnn_params:
        report_content += f"""
- BNN Parameters: {results['config'].bnn_params}
"""
    report_content += f"""
- Training Parameters: {results['config'].training_params}

## Performance Metrics

### Training Set Performance
"""

    # 训练集完整性能
    train_full = results['train_performance']['full']
    report_content += f"""
### Training Set Performance (Full)
- R² Score: {format_value(train_full['r2_score'])}
- MAE: {format_value(train_full['mae'])}
- RMSE: {format_value(train_full['rmse'])}
- Mean Uncertainty: {format_value(train_full.get('mean_uncertainty', 'N/A'))}
- Std Uncertainty: {format_value(train_full.get('std_uncertainty', 'N/A'))}
- 95% Uncertainty Percentile: {format_value(train_full.get('uncertainty_95_percentile', 'N/A'))}
"""

    # 添加:全数据集(含增强)性能
    if 'full_with_augmentation' in results['train_performance']:
        train_aug = results['train_performance']['full_with_augmentation']
        report_content += f"""
### Training Set Performance (With Data Augmentation)
- R² Score: {format_value(train_aug['r2_score'])}
- MAE: {format_value(train_aug['mae'])}
- RMSE: {format_value(train_aug['rmse'])}
- Total Samples: {train_aug.get('total_samples', 'N/A')}
- Original Samples: {train_aug.get('original_samples', 'N/A')}
- Augmented Samples: {train_aug.get('augmented_samples', 'N/A')}
- Mean Uncertainty: {format_value(train_aug.get('mean_uncertainty', 'N/A'))}
- Std Uncertainty: {format_value(train_aug.get('std_uncertainty', 'N/A'))}
- 95% Uncertainty Percentile: {format_value(train_aug.get('uncertainty_95_percentile', 'N/A'))}
"""

    # 训练集STO性能
    if 'sto' in results['train_performance'] and results['train_performance']['sto']:
        train_sto = results['train_performance']['sto']
        report_content += f"""
### STO Training Set Performance
- R² Score: {format_value(train_sto['r2_score'])}
- MAE: {format_value(train_sto['mae'])}
- RMSE: {format_value(train_sto['rmse'])}
- Samples: {train_sto.get('num_samples', 'N/A')}
- Mean Uncertainty: {format_value(train_sto.get('mean_uncertainty', 'N/A'))}
- Std Uncertainty: {format_value(train_sto.get('std_uncertainty', 'N/A'))}
- 95% Uncertainty Percentile: {format_value(train_sto.get('uncertainty_95_percentile', 'N/A'))}
"""

    # 验证集完整性能
    if results.get('val_performance'):
        val_full = results['val_performance']['full']
        report_content += f"""
### Validation Set Performance (Full)
- R² Score: {format_value(val_full['r2_score'])}
- MAE: {format_value(val_full['mae'])}
- RMSE: {format_value(val_full['rmse'])}
- Mean Uncertainty: {format_value(val_full.get('mean_uncertainty', 'N/A'))}
- Std Uncertainty: {format_value(val_full.get('std_uncertainty', 'N/A'))}
- 95% Uncertainty Percentile: {format_value(val_full.get('uncertainty_95_percentile', 'N/A'))}
"""

    # 验证集STO性能
    if (results.get('val_performance') and
            'sto' in results['val_performance'] and
            results['val_performance']['sto']):
        val_sto = results['val_performance']['sto']
        report_content += f"""
### STO Validation Set Performance
- R² Score: {format_value(val_sto['r2_score'])}
- MAE: {format_value(val_sto['mae'])}
- RMSE: {format_value(val_sto['rmse'])}
- Samples: {val_sto.get('num_samples', 'N/A')}
- Mean Uncertainty: {format_value(val_sto.get('mean_uncertainty', 'N/A'))}
- Std Uncertainty: {format_value(val_sto.get('std_uncertainty', 'N/A'))}
- 95% Uncertainty Percentile: {format_value(val_sto.get('uncertainty_95_percentile', 'N/A'))}
"""

    # 修复：安全地处理 full_dataset 变量
    full_dataset = results.get('full_dataset_performance', {})

    # 新增：全数据集性能
    if full_dataset:  # 只有当 full_dataset 不为空时才处理
        report_content += """
## Full Dataset Performance (Training + Validation)
"""
        full_full = full_dataset['full']
        report_content += f"""
### Full Dataset Performance (All Samples)
- R² Score: {format_value(full_full['r2_score'])}
- MAE: {format_value(full_full['mae'])}
- RMSE: {format_value(full_full['rmse'])}
- Mean Uncertainty: {format_value(full_full.get('mean_uncertainty', 'N/A'))}
- Std Uncertainty: {format_value(full_full.get('std_uncertainty', 'N/A'))}
- 95% Uncertainty Percentile: {format_value(full_full.get('uncertainty_95_percentile', 'N/A'))}
- Total Samples: {len(full_dataset.get('y_true', [])) if 'y_true' in full_dataset else 'N/A'}
"""

        # 全数据集STO性能（现在在条件内部）
        if 'sto' in full_dataset and full_dataset['sto']:
            full_sto = full_dataset['sto']
            report_content += f"""
### STO Full Dataset Performance
- R² Score: {format_value(full_sto['r2_score'])}
- MAE: {format_value(full_sto['mae'])}
- RMSE: {format_value(full_sto['rmse'])}
- Samples: {full_sto.get('num_samples', 'N/A')}
- Mean Uncertainty: {format_value(full_sto.get('mean_uncertainty', 'N/A'))}
- Std Uncertainty: {format_value(full_sto.get('std_uncertainty', 'N/A'))}
- 95% Uncertainty Percentile: {format_value(full_sto.get('uncertainty_95_percentile', 'N/A'))}
"""

    report_content += """
## Performance Summary Comparison
"""

    # 添加性能对比表格
    report_content += """
| Dataset | R² Score | MAE | RMSE | Samples |
|---------|----------|-----|------|---------|
"""

    # 训练集性能
    train_full = results['train_performance']['full']
    report_content += f"| Training Set | {format_value(train_full['r2_score'])} | {format_value(train_full['mae'])} | {format_value(train_full['rmse'])} | {len(results.get('X_train', [])) if 'X_train' in results else 'N/A'} |\n"

    # 验证集性能
    if results.get('val_performance'):
        val_full = results['val_performance']['full']
        report_content += f"| Validation Set | {format_value(val_full['r2_score'])} | {format_value(val_full['mae'])} | {format_value(val_full['rmse'])} | {len(results.get('X_val', [])) if 'X_val' in results else 'N/A'} |\n"

    # 全数据集性能（只有在存在时才显示）
    if full_dataset:
        full_full = full_dataset['full']
        total_samples = len(results.get('X_train', [])) + len(
            results.get('X_val', [])) if 'X_train' in results and 'X_val' in results else 'N/A'
        report_content += f"| **Full Dataset** | **{format_value(full_full['r2_score'])}** | **{format_value(full_full['mae'])}** | **{format_value(full_full['rmse'])}** | **{total_samples}** |\n"

    # 在适当位置添加合并数据集的性能报告
    combined_perf = results.get('combined_dataset_performance', {})
    if combined_perf:
        report_content += """
## Combined Dataset Performance (Original + Augmented)
"""
        combined_full = combined_perf['full']
        report_content += f"""
### Combined Dataset Performance (All Samples)
- R² Score: {format_value(combined_full['r2_score'])}
- MAE: {format_value(combined_full['mae'])}
- RMSE: {format_value(combined_full['rmse'])}
- Mean Uncertainty: {format_value(combined_full.get('mean_uncertainty', 'N/A'))}
- Std Uncertainty: {format_value(combined_full.get('std_uncertainty', 'N/A'))}
- 95% Uncertainty Percentile: {format_value(combined_full.get('uncertainty_95_percentile', 'N/A'))}
- Total Samples: {combined_full.get('total_samples', 'N/A')}
- Original Samples: {combined_full.get('original_samples', 'N/A')}
- Augmented Samples: {combined_full.get('augmented_samples', 'N/A')}
"""

        # 合并数据集的STO性能（在条件内部）
        if 'sto' in combined_perf and combined_perf['sto']:
            combined_sto = combined_perf['sto']
            report_content += f"""
### STO Combined Dataset Performance
- R² Score: {format_value(combined_sto['r2_score'])}
- MAE: {format_value(combined_sto['mae'])}
- RMSE: {format_value(combined_sto['rmse'])}
- Samples: {combined_sto.get('num_samples', 'N/A')}
- Mean Uncertainty: {format_value(combined_sto.get('mean_uncertainty', 'N/A'))}
- Std Uncertainty: {format_value(combined_sto.get('std_uncertainty', 'N/A'))}
- 95% Uncertainty Percentile: {format_value(combined_sto.get('uncertainty_95_percentile', 'N/A'))}
"""

    # 添加序列损失信息
    if results.get('final_sequence_loss'):
        report_content += """
### Training-Loss Results
"""
        report_content += f"- Final Sequence Loss: {format_value(results['final_sequence_loss'])}\n"

    if results.get('cv_results'):
        report_content += """
### Cross-Validation Results
"""
        for metric, stats in results['cv_results'].items():
            report_content += f"- {metric.upper()}: {format_value(stats['mean'])} ± {format_value(stats['std'])}\n"

    report_content += """
## Feature Importance
"""
    # 修复：使用 get 方法安全访问 feature_importance
    feature_importance = results.get('feature_importance', [])
    feature_names = results.get('feature_names', [])

    if feature_importance:
        # 如果 feature_importance 是列表，取最后一个
        if isinstance(feature_importance, list) and feature_importance:
            latest_importance = feature_importance[-1].get('xgb', [])
        else:
            latest_importance = feature_importance.get('xgb', []) if isinstance(feature_importance, dict) else []

        if len(latest_importance) == len(feature_names):
            for i, (name, importance) in enumerate(zip(feature_names, latest_importance)):
                report_content += f"- {name}: {format_value(importance)}\n"
        else:
            report_content += "- Feature importance data format mismatch\n"
    else:
        report_content += "- No feature importance data available\n"

    # 序列损失分量分析
    if results.get('sequence_loss_components'):
        report_content += """
## Sequence Loss Analysis
"""
        components = results['sequence_loss_components']
        report_content += f"- Intra-Good Loss: {format_value(components['intra_good'])}\n"
        report_content += f"- Intra-Bad Loss: {format_value(components['intra_bad'])}\n"
        report_content += f"- Inter-Sequence Loss: {format_value(components['inter'])}\n"

    # Save report
    with open(output_path / "experiment_report.md", 'w', encoding='utf-8') as f:
        f.write(report_content)


def create_bnn_experiment_report(results: Dict, output_path: Path):
    """Generate Pure BNN experiment report in Markdown format"""
    config = results['config']

    report_content = f"""
# Pure BNN Model Experiment Report

## Model Configuration
- BNN Parameters: {config.bnn_params}
- Training Parameters: {config.training_params}

## Performance Metrics

### Training Set Performance
"""

    # 训练集完整性能
    train_full = results['train_performance']['full']
    report_content += f"""
### Training Set Performance (Full)
- R² Score: {train_full['r2_score']:.4f}
- MAE: {train_full['mae']:.4f}
- RMSE: {train_full['rmse']:.4f}
- Mean Uncertainty: {results['train_performance']['mean_uncertainty']:.4f}
"""

    # 训练集STO性能
    if results['train_performance']['sto']:
        train_sto = results['train_performance']['sto']
        report_content += f"""
### STO Training Set Performance
- R² Score: {train_sto['r2_score']:.4f}
- MAE: {train_sto['mae']:.4f}
- RMSE: {train_sto['rmse']:.4f}
- Samples: {train_sto['num_samples']}
"""

    # 验证集完整性能
    if results.get('val_performance'):
        val_full = results['val_performance']['full']
        report_content += f"""
### Validation Set Performance (Full)
- R² Score: {val_full['r2_score']:.4f}
- MAE: {val_full['mae']:.4f}
- RMSE: {val_full['rmse']:.4f}
"""

    # 验证集STO性能
    if results.get('val_performance') and results['val_performance'].get('sto'):
        val_sto = results['val_performance']['sto']
        report_content += f"""
### STO Validation Set Performance
- R² Score: {val_sto['r2_score']:.4f}
- MAE: {val_sto['mae']:.4f}
- RMSE: {val_sto['rmse']:.4f}
- Samples: {val_sto['num_samples']}
"""

    if results.get('cv_results'):
        report_content += """
### Cross-Validation Results
"""
        for metric, stats in results['cv_results'].items():
            report_content += f"- {metric.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f}\n"

    report_content += """
## Feature Importance
"""
    feature_importance = results['train_performance']['feature_importance']
    feature_names = results['feature_names']
    for i, (name, importance) in enumerate(zip(feature_names, feature_importance)):
        report_content += f"- {name}: {importance:.4f}\n"

    # Save report
    with open(output_path / "experiment_report.md", 'w', encoding='utf-8') as f:
        f.write(report_content)


def create_xgboost_experiment_report(results: Dict, output_path: Path):
    """Generate XGBoost-specific experiment report in Markdown format"""

    def format_value(value, default='N/A'):
        """安全格式化数值"""
        if value is None or value == 'N/A':
            return default
        try:
            return f"{value:.4f}"
        except (TypeError, ValueError):
            return str(value)

    # 安全地获取训练集性能数据
    train_performance = results.get('train_performance', {})
    val_performance = results.get('val_performance', {})

    report_content = f"""
# XGBoost Model Experiment Report

## Model Configuration
- XGBoost Parameters: {results['config'].xgb_params}

## Performance Metrics

### Training Set Performance
"""
    # 训练集完整性能 - 使用安全的格式化函数
    train_full = train_performance.get('full', {})
    report_content += f"""
### Training Set Performance (Full)
- R² Score: {format_value(train_full.get('r2_score'))}
- MAE: {format_value(train_full.get('mae'))}
- RMSE: {format_value(train_full.get('rmse'))}
"""

    # 训练集STO性能
    train_sto = train_performance.get('sto')
    if train_sto:  # 只有当sto存在且不为空时才显示
        report_content += f"""
### STO Training Set Performance
- R² Score: {format_value(train_sto.get('r2_score'))}
- MAE: {format_value(train_sto.get('mae'))}
- RMSE: {format_value(train_sto.get('rmse'))}
- Samples: {train_sto.get('num_samples', 'N/A')}
"""

    # 验证集完整性能
    if val_performance:
        val_full = val_performance.get('full', {})
        report_content += f"""
### Validation Set Performance (Full)
- R² Score: {format_value(val_full.get('r2_score'))}
- MAE: {format_value(val_full.get('mae'))}
- RMSE: {format_value(val_full.get('rmse'))}
"""

    # 验证集STO性能
    val_sto = val_performance.get('sto') if val_performance else None
    if val_sto:  # 只有当sto存在且不为空时才显示
        report_content += f"""
### STO Validation Set Performance
- R² Score: {format_value(val_sto.get('r2_score'))}
- MAE: {format_value(val_sto.get('mae'))}
- RMSE: {format_value(val_sto.get('rmse'))}
- Samples: {val_sto.get('num_samples', 'N/A')}
"""

    # 添加全数据集性能（如果存在）
    combined_performance = results.get('combined_dataset_performance', {})
    if combined_performance:
        combined_full = combined_performance.get('full', {})
        report_content += f"""
### Combined Dataset Performance (Original + Augmented)
- R² Score: {format_value(combined_full.get('r2_score'))}
- MAE: {format_value(combined_full.get('mae'))}
- RMSE: {format_value(combined_full.get('rmse'))}
- Total Samples: {combined_full.get('total_samples', 'N/A')}
- Original Samples: {combined_full.get('original_samples', 'N/A')}
- Augmented Samples: {combined_full.get('augmented_samples', 'N/A')}
"""

    # 交叉验证结果（如果存在）
    if results.get('cv_results'):
        report_content += """
### Cross-Validation Results
"""
        for metric, stats in results['cv_results'].items():
            mean_val = stats.get('mean', 'N/A')
            std_val = stats.get('std', 'N/A')
            report_content += f"- {metric.upper()}: {format_value(mean_val)} ± {format_value(std_val)}\n"

    # 特征重要性（如果存在）
    report_content += """
## Feature Importance
"""
    feature_importance = results.get('feature_importance', [])
    feature_names = results.get('feature_names', [])

    if feature_importance and feature_names:
        # 安全地获取最新的特征重要性
        latest_importance = []
        if isinstance(feature_importance, list) and feature_importance:
            latest_item = feature_importance[-1]
            if isinstance(latest_item, dict):
                latest_importance = latest_item.get('xgb', [])
            else:
                latest_importance = latest_item
        elif isinstance(feature_importance, dict):
            latest_importance = feature_importance.get('xgb', [])

        if len(latest_importance) == len(feature_names):
            for i, (name, importance) in enumerate(zip(feature_names, latest_importance)):
                report_content += f"- {name}: {format_value(importance)}\n"
        else:
            report_content += "- Feature importance data format mismatch\n"
    else:
        report_content += "- No feature importance data available\n"

    # 保存报告
    report_path = output_path / "experiment_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    logger.info(f"Experiment report saved to {report_path}")
