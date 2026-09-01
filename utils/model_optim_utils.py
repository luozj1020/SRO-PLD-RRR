import numpy as np
import pandas as pd
import optuna
import joblib
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import logging
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import gc
import torch

from model_utils import ModelVisualizer, plot_sequence_loss
from utils.data_processer import calculate_boundary_penalty_weights, compute_fixed_covariance_matrix, BOUND_1, BOUND_2
from utils.tuning_config import TuningConfig

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class StabilityAwareObjective:
    """计算训练稳定性的目标函数"""

    def __init__(self, tuner_instance):
        self.tuner = tuner_instance
        self.stability_weight = 0.3

    def calculate_training_stability(self, model) -> float:
        """计算模型训练的稳定性分数"""
        if not hasattr(model, 'history') or model.history is None:
            return 0.0
        r2_history = model.history.get('r2', [])
        loss_history = model.history.get('loss', [])

        if not r2_history or not loss_history:
            return 0.0

        min_len = min(len(r2_history), len(loss_history))
        r2_history = r2_history[:min_len]
        loss_history = loss_history[:min_len]

        if len(r2_history) < 10:
            return 0.0

        # R²稳定性
        recent_r2 = r2_history[-20:] if len(r2_history) >= 20 else r2_history
        r2_cv = np.std(recent_r2) / (np.abs(np.mean(recent_r2)) + 1e-8)
        r2_stability = max(0, 1 - r2_cv * 10)

        # 损失收敛性
        recent_loss = loss_history[-20:] if len(loss_history) >= 20 else loss_history
        if len(recent_loss) < 2:
            loss_trend = 0.0
        else:
            loss_trend = np.polyfit(range(len(recent_loss)), recent_loss, 1)[0]
        convergence_score = max(0, min(1, -loss_trend * 1000))

        # 过拟合检测
        if len(r2_history) >= 30:
            early_r2 = np.mean(r2_history[10:20])
            late_r2 = np.mean(r2_history[-10:])
            overfitting_penalty = max(0, early_r2 - late_r2) * 2
            stability_score = max(0, 1 - overfitting_penalty)
        else:
            stability_score = 1.0

        # 振荡检测
        if len(r2_history) >= 10:
            r2_changes = np.abs(np.diff(r2_history[-20:]))
            oscillation_score = max(0, 1 - np.mean(r2_changes) * 20)
        else:
            oscillation_score = 1.0

        total_stability = (r2_stability * 0.3 + convergence_score * 0.3 +
                           stability_score * 0.2 + oscillation_score * 0.2)
        return np.clip(total_stability, 0, 1)


class HyperparameterTuner:
    """超参数调优器 - 适配嵌入式处理器的模型"""

    def __init__(
            self,
            data_path: str,  # 只需要一个数据路径
            base_features: List[str],
            target_column: str,
            sto_column: str = 'Substrate',
            tuning_config: TuningConfig = None,
            weight_config: Dict = None,
            output_dir: str = "./tuning_results",
            visualization_output_dir: str = None,
            model_class: Any = None,
            config_class: Any = None,
            use_boundary_weights: bool = True,
            boundary_config: Dict = None,
            sequence_good_path: str = "",
            sequence_bad_path: str = ""
    ):
        self.data_path = data_path
        self.base_features = base_features
        self.target_column = target_column
        self.sto_column = sto_column
        self.tuning_config = tuning_config or TuningConfig()
        self.weight_config = weight_config or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_class = model_class
        self.config_class = config_class
        self.visualization_output_dir = Path(visualization_output_dir) if visualization_output_dir else self.output_dir
        self.visualization_output_dir.mkdir(parents=True, exist_ok=True)
        self.use_boundary_weights = use_boundary_weights
        self.boundary_config = boundary_config or {
            'gaussian_sigma': 0.8,
            'missing_penalty_rate': 0.95,
            'penalty_sharpness': 1.5,
            'use_mahalanobis': True
        }
        self.sequence_good_path = sequence_good_path
        self.sequence_bad_path = sequence_bad_path

        # 初始化属性
        self._cached_weights = {}
        self._trial_times = []
        self._memory_usage = []
        self.best_params = {}
        self.tuning_history = []
        self._last_trained_model = None
        self.fixed_covariance_inv = None

        # 加载并预处理数据
        self._load_and_preprocess_data()

        # 初始化稳定性目标
        self.stability_objective = StabilityAwareObjective(self)
        logger.info(f"Tuner initialized with {len(self.train_data)} training samples")

    def _load_and_preprocess_data(self):
        """加载并预处理数据 - 适配原模型的数据格式"""
        logger.info(f"Loading data from {self.data_path}")
        data = pd.read_excel(self.data_path)

        # 创建STO标记 - 只有'STO'才标记为1
        if self.sto_column in data.columns:
            data['is_sto'] = (data[self.sto_column] == 'STO').astype(int)
            logger.info(f"STO samples: {data['is_sto'].sum()}")
            logger.info(f"Non-STO samples: {len(data) - data['is_sto'].sum()}")
        else:
            data['is_sto'] = 0
            logger.warning(f"'{self.sto_column}' column not found. Setting all samples as non-STO")

        # 计算固定协方差矩阵(用于边界权重计算)
        logger.info("Computing fixed covariance matrix...")
        X_raw_values = data[self.base_features].values
        self.fixed_covariance_inv = compute_fixed_covariance_matrix(
            X_raw_values,
            self.base_features,
            BOUND_1,
            BOUND_2
        )

        # 计算边界权重
        if self.use_boundary_weights:
            logger.info("Calculating boundary weights...")
            boundary_weights = calculate_boundary_penalty_weights(
                X_raw_values,
                self.base_features,
                gaussian_sigma=self.boundary_config['gaussian_sigma'],
                missing_penalty_rate=self.boundary_config['missing_penalty_rate'],
                penalty_sharpness=self.boundary_config['penalty_sharpness'],
                use_mahalanobis=self.boundary_config['use_mahalanobis'],
                fixed_covariance_inv=self.fixed_covariance_inv
            )
            data['boundary_weight'] = boundary_weights
        else:
            data['boundary_weight'] = 1.0

        # 数据分割 - 使用权重分层
        val_size = 0.5
        random_state = 42

        # 创建二分类标签用于分层
        data['STO_binary'] = data['is_sto']

        try:
            # 尝试分层分割
            train_data, val_data = train_test_split(
                data,
                test_size=val_size,
                random_state=random_state,
                stratify=data['STO_binary']
            )
            logger.info("Stratified split successful")
        except ValueError as e:
            logger.warning(f"Stratified split failed: {e}, using random split")
            train_data, val_data = train_test_split(
                data,
                test_size=val_size,
                random_state=random_state
            )

        # 删除临时列
        train_data = train_data.drop('STO_binary', axis=1)
        val_data = val_data.drop('STO_binary', axis=1)

        self.train_data = train_data
        self.val_data = val_data

        logger.info(f"Data split: {len(train_data)} train, {len(val_data)} validation")
        logger.info(f"STO samples - Train: {train_data['is_sto'].sum()}, Val: {val_data['is_sto'].sum()}")

    def _get_sample_weights(self, weight_params: Dict = None) -> Tuple[np.ndarray, np.ndarray]:
        # 统一使用边界权重方法（与原模型的默认行为一致）
        weight_threshold = self.weight_config.get('weight_threshold', 0.05)

        # 训练集权重
        X_train_raw = self.train_data[self.base_features].values
        train_weights = calculate_boundary_penalty_weights(
            X_train_raw,
            self.base_features,
            gaussian_sigma=self.boundary_config['gaussian_sigma'],
            missing_penalty_rate=self.boundary_config['missing_penalty_rate'],
            penalty_sharpness=self.boundary_config['penalty_sharpness'],
            use_mahalanobis=self.boundary_config['use_mahalanobis'],
            fixed_covariance_inv=self.fixed_covariance_inv,
            weight_threshold=weight_threshold
        )

        # 验证集权重
        X_val_raw = self.val_data[self.base_features].values
        val_weights = calculate_boundary_penalty_weights(
            X_val_raw,
            self.base_features,
            gaussian_sigma=self.boundary_config['gaussian_sigma'],
            missing_penalty_rate=self.boundary_config['missing_penalty_rate'],
            penalty_sharpness=self.boundary_config['penalty_sharpness'],
            use_mahalanobis=self.boundary_config['use_mahalanobis'],
            fixed_covariance_inv=self.fixed_covariance_inv,
            weight_threshold=0.0  # 验证集不应用阈值
        )

        return train_weights, val_weights

    def _suggest_params_batch(self, trial: optuna.Trial) -> Tuple[Dict, Dict, Dict, Dict]:
        """批量建议超参数"""

        def suggest_param(param_name, config, prefix=""):
            full_name = f"{prefix}{param_name}"
            if config['type'] == 'int':
                return trial.suggest_int(full_name, config['low'], config['high'])
            elif config['type'] == 'float':
                return trial.suggest_float(full_name, config['low'], config['high'], log=config.get('log', False))
            elif config['type'] == 'categorical':
                return trial.suggest_categorical(full_name, config['choices'])
            return None

        # XGBoost参数
        xgb_params = {param: suggest_param(param, config, "xgb_")
                      for param, config in self.tuning_config.xgb_search_space.items()}
        xgb_params['random_state'] = 42

        # BNN参数
        bnn_params = {}
        training_params = {}
        for param_name, config in self.tuning_config.bnn_search_space.items():
            if param_name == 'dropout_rates':
                bnn_params[param_name] = [
                    trial.suggest_float(f"bnn_dropout_rate_{i}", config['low'], config['high'])
                    for i in range(config['size'])
                ]
            elif param_name in ['learning_rate', 'weight_decay']:
                training_params[param_name] = suggest_param(param_name, config, "bnn_")
            else:
                bnn_params[param_name] = suggest_param(param_name, config, "bnn_")

        # 权重参数
        weight_params = {param: suggest_param(param, config, "weight_")
                         for param, config in self.tuning_config.weight_search_space.items()}

        return xgb_params, bnn_params, training_params, weight_params

    def _create_model_config(self, xgb_params: Dict, bnn_params: Dict, training_params: Dict) -> Any:
        """创建模型配置"""
        config = self.config_class()
        config.xgb_params.update(xgb_params)

        for key, value in bnn_params.items():
            if key == 'first_hidden_dims_pow':
                config.first_hidden_dims_pow = value
            else:
                config.bnn_params[key] = value

        config.training_params.update({
            **training_params,
            'verbose_training': False,
            'show_progress': False
        })

        if self.sequence_good_path and self.sequence_bad_path:
            good_path = Path(self.sequence_good_path)
            bad_path = Path(self.sequence_bad_path)
            if good_path.exists() and bad_path.exists():
                config.sequence_loss['enabled'] = True
                config.sequence_loss['good_data_path'] = str(good_path)
                config.sequence_loss['bad_data_path'] = str(bad_path)
            else:
                config.sequence_loss['enabled'] = False

        # 静默模式
        config.output_control = {
            'mode': 'silent',
            'show_progress_bar': False,
            'show_epoch_details': False,
            'show_parameter_info': False,
            'show_early_stopping_info': False,
            'show_final_summary': False,
            'log_level': 'WARNING'
        }

        return config

    def _calculate_sto_r2(self, model, val_data: pd.DataFrame, val_weights: np.ndarray) -> float:
        """计算STO样本的R²分数"""
        try:
            sto_mask = val_data['is_sto'] == 1
            if sto_mask.sum() == 0:
                logger.warning("No STO samples in validation set")
                return -float('inf')

            # 提取STO数据
            X_val_sto = val_data[val_data['is_sto'] == 1][self.base_features + [self.sto_column]].copy()
            y_val_sto = val_data[val_data['is_sto'] == 1][self.target_column].values
            weights_sto = val_weights[sto_mask]

            # 预测
            pred_sto, _ = model.predict(X_val_sto)

            # 计算加权R²
            return r2_score(y_val_sto, pred_sto, sample_weight=weights_sto)

        except Exception as e:
            logger.error(f"STO R² calculation failed: {str(e)}")
            return -float('inf')

    def _objective(self, trial: optuna.Trial) -> Tuple[float, float, float]:
        """优化目标函数"""
        try:
            # 建议参数
            xgb_params, bnn_params, training_params, weight_params = self._suggest_params_batch(trial)
            config = self._create_model_config(xgb_params, bnn_params, training_params)

            # 获取权重
            train_weights, val_weights = self._get_sample_weights(weight_params)

            # 准备训练数据
            X_train = self.train_data[self.base_features + [self.sto_column]].copy()
            y_train = self.train_data[self.target_column]

            X_val = self.val_data[self.base_features + [self.sto_column]].copy()
            y_val = self.val_data[self.target_column]

            # 训练模型
            model = self.model_class(base_features=self.base_features, config=config)
            model.fit(X_train, y_train, val_X=X_val, val_y=y_val, sto_column=self.sto_column)

            # 计算稳定性分数
            stability_score = self.stability_objective.calculate_training_stability(model)
            self._last_trained_model = model  # 保存模型引用
            trial.set_user_attr("stability_score", stability_score)

            # 计算验证集STO R²
            sto_r2 = self._calculate_sto_r2(model, self.val_data, val_weights)
            if sto_r2 == -float('inf'):
                logger.warning(f"Invalid parameters in trial {trial.number}: STO R² calculation failed")
                return -float('inf'), -float('inf'), 0.0

            # 计算第二目标
            if self.tuning_config.multi_objective_type == 'loss':
                # 使用最终损失
                final_loss = float('inf')
                if hasattr(model, 'history') and model.history is not None:
                    loss_history = model.history.get('loss', [])
                    if loss_history and len(loss_history) > 0:
                        final_loss = loss_history[-1]

                if np.isnan(final_loss) or np.isinf(final_loss):
                    logger.warning(f"Invalid final loss in trial {trial.number}: {final_loss}")
                    return -float('inf'), -float('inf'), 0.0
                obj_value2 = final_loss
            else:
                # 使用交叉验证R²
                try:
                    cv_scores = []
                    kfold = KFold(n_splits=self.tuning_config.cv_folds, shuffle=True, random_state=42)

                    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
                        X_train_fold = X_train.iloc[train_idx].copy()
                        y_train_fold = y_train.iloc[train_idx]
                        X_val_fold = X_train.iloc[val_idx].copy()
                        y_val_fold = y_train.iloc[val_idx]
                        weights_val_fold = train_weights[val_idx]

                        fold_model = self.model_class(base_features=self.base_features, config=config)
                        fold_model.fit(X_train_fold, y_train_fold, sto_column=self.sto_column)

                        pred_fold, _ = fold_model.predict(X_val_fold)
                        fold_score = r2_score(y_val_fold, pred_fold, sample_weight=weights_val_fold)
                        cv_scores.append(fold_score)

                        del fold_model
                        gc.collect()

                    obj_value2 = np.mean(cv_scores)
                except Exception as cv_error:
                    logger.error(f"Cross-validation failed in trial {trial.number}: {str(cv_error)}")
                    return -float('inf'), -float('inf'), 0.0

            # ⭐ 新增：立即保存当前模型
            self._save_trial_model(trial, model, sto_r2, obj_value2, stability_score)

            # 记录试验结果
            self.tuning_history.append({
                'trial_number': trial.number,
                'sto_val_r2': sto_r2,
                'secondary_score': obj_value2,
                'stability_score': stability_score,
                'params': {**xgb_params, **bnn_params, **training_params, **weight_params}
            })

            return sto_r2, obj_value2, stability_score

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {type(e).__name__}: {str(e)}")
            return -float('inf'), -float('inf'), 0.0

    def _save_trial_model(self, trial: optuna.Trial, model, sto_r2: float,
                          obj_value2: float, stability_score: float):
        """保存试验模型及其性能数据"""
        try:
            # 创建试验目录
            trial_dir = self.output_dir / "all_trials" / f"trial_{trial.number}"
            trial_dir.mkdir(parents=True, exist_ok=True)

            # 1. 保存模型
            model.save_model(str(trial_dir / "model"))

            # 2. 计算并保存性能数据
            X_train = self.train_data[self.base_features + [self.sto_column]].copy()
            y_train = self.train_data[self.target_column]
            X_val = self.val_data[self.base_features + [self.sto_column]].copy()
            y_val = self.val_data[self.target_column]

            # 获取权重
            train_weights, val_weights = self._get_sample_weights()

            # 预测
            train_pred, train_std = model.predict(X_train)
            val_pred, val_std = model.predict(X_val)

            # 计算性能指标
            performance_data = {
                'trial_number': trial.number,
                'objectives': {
                    'sto_r2': sto_r2,
                    'secondary_objective': obj_value2,
                    'stability_score': stability_score
                },
                'validation_performance': {
                    'r2_score': r2_score(y_val, val_pred, sample_weight=val_weights),
                    'mae': mean_absolute_error(y_val, val_pred, sample_weight=val_weights),
                    'rmse': np.sqrt(mean_squared_error(y_val, val_pred, sample_weight=val_weights)),
                    'sto_r2': sto_r2,
                    'num_sto_samples': self.val_data['is_sto'].sum()
                },
                'training_performance': {
                    'r2_score': r2_score(y_train, train_pred, sample_weight=train_weights),
                    'mae': mean_absolute_error(y_train, train_pred, sample_weight=train_weights),
                    'rmse': np.sqrt(mean_squared_error(y_train, train_pred, sample_weight=train_weights)),
                    'sto_r2': self._calculate_sto_r2(model, self.train_data, train_weights),
                    'num_sto_samples': self.train_data['is_sto'].sum()
                },
                'params': trial.params
            }

            # 保存性能数据
            joblib.dump(performance_data, trial_dir / "performance_data.joblib")

            # 保存预测结果
            results_df = pd.DataFrame({
                **{feature: self.val_data[feature].values for feature in self.base_features},
                'true_' + self.target_column: y_val,
                'predicted_' + self.target_column: val_pred,
                'residual': y_val - val_pred,
                'uncertainty': val_std,
                'is_sto': self.val_data['is_sto'].values,
                'weight': val_weights
            })
            results_df.to_csv(trial_dir / "validation_predictions.csv", index=False, encoding='utf-8')

            logger.info(f"Trial {trial.number} model saved to {trial_dir}")

            # 清理内存
            del train_pred, train_std, val_pred, val_std
            gc.collect()

        except Exception as e:
            logger.error(f"Failed to save trial {trial.number} model: {str(e)}")


    def _plot_optimization_history(self):
        """绘制优化历史"""
        if not self.tuning_history:
            logger.warning("No optimization data to plot")
            return

        trials = [h['trial_number'] for h in self.tuning_history]
        sto_r2_scores = [h['sto_val_r2'] for h in self.tuning_history]
        secondary_scores = [h['secondary_score'] for h in self.tuning_history]
        stability_scores = [h['stability_score'] for h in self.tuning_history]

        plt.figure(figsize=(15, 15))

        # STO R²历史
        plt.subplot(3, 1, 1)
        plt.scatter(trials, sto_r2_scores, alpha=0.6, color='blue', label='STO Validation R² (per trial)')
        best_history = np.maximum.accumulate(sto_r2_scores)
        plt.plot(trials, best_history, 'r--', linewidth=2.5, label='Best STO Validation R²')
        best_idx = np.argmax(sto_r2_scores)
        plt.scatter([trials[best_idx]], [sto_r2_scores[best_idx]], s=150, c='gold',
                    marker='*', edgecolor='black', label='Best STO Point')
        plt.title('STO Validation R² Optimization History')
        plt.xlabel('Trial Number')
        plt.ylabel('STO Validation R²')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # 第二目标历史
        plt.subplot(3, 1, 2)
        plt.scatter(trials, secondary_scores, alpha=0.6,
                    color='green' if self.tuning_config.multi_objective_type != 'loss' else 'purple')
        if self.tuning_config.multi_objective_type == 'loss':
            best_history = np.minimum.accumulate(secondary_scores)
            plt.ylabel('Loss')
            plt.title('Training Loss Optimization History')
        else:
            best_history = np.maximum.accumulate(secondary_scores)
            plt.ylabel('CV R²')
            plt.title('Cross-Validation R² Optimization History')
        plt.plot(trials, best_history, 'r--', linewidth=2.5, label='Best')
        plt.xlabel('Trial Number')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # 稳定性历史
        plt.subplot(3, 1, 3)
        plt.scatter(trials, stability_scores, alpha=0.6, color='orange', label='Training Stability')
        best_stability = np.maximum.accumulate(stability_scores)
        plt.plot(trials, best_stability, 'r--', linewidth=2.5, label='Best Stability')
        plt.title('Training Stability Optimization History')
        plt.xlabel('Trial Number')
        plt.ylabel('Stability Score')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        save_path = self.visualization_output_dir / "optimization_history.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Optimization history saved to {save_path}")

    def _visualize_tuning_results(self, study: optuna.Study):
        """可视化调优结果"""
        if not study or not study.trials:
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        plt.suptitle("Hyperparameter Tuning Results", fontsize=16)

        # 帕累托前沿图
        ax_tune = axes[0]
        full_scores = [t.values[0] for t in study.trials if t.values and len(t.values) >= 3]
        obj_scores = [t.values[1] for t in study.trials if t.values and len(t.values) >= 3]
        stability_scores = [t.values[2] for t in study.trials if t.values and len(t.values) >= 3]

        scatter = ax_tune.scatter(full_scores, obj_scores, c=stability_scores,
                                    cmap='viridis', alpha=0.6, label='All Trials', s=30)
        cbar = plt.colorbar(scatter, ax=ax_tune)
        cbar.set_label('Training Stability Score', rotation=270, labelpad=15)

        ax_tune.set_xlabel('STO Validation R²')
        ylabel = 'Training Loss' if self.tuning_config.multi_objective_type == 'loss' else 'CV R²'
        ax_tune.set_ylabel(ylabel)
        ax_tune.legend()
        ax_tune.grid(True, alpha=0.3)

        # 参数重要性
        ax_importances = axes[1]
        if full_scores:
            try:
                importances = optuna.importance.get_param_importances(study, target=lambda t: t.values[0])
                df_importances = pd.DataFrame({
                    'Parameter': list(importances.keys()),
                    'Importance': list(importances.values())
                }).sort_values('Importance', ascending=True)
                ax_importances.barh(df_importances['Parameter'], df_importances['Importance'], color='skyblue')
                ax_importances.set_title('Parameter Importance (STO R²)')
                ax_importances.set_xlabel('Importance')
                ax_importances.grid(True, linestyle='--', alpha=0.6)
            except Exception as e:
                ax_importances.text(0.5, 0.5, f'Parameter importance failed: {str(e)}',
                                    ha='center', va='center')

        plt.tight_layout()
        save_path = self.output_dir / "tuning_visualization.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Tuning visualization saved to {save_path}")

    def _save_tuning_results(self, results: Dict):
        """保存调优结果"""
        # 保存主结果
        joblib.dump(results, self.output_dir / "tuning_results.joblib")

        # 保存优化历史图
        self._plot_optimization_history()

        # 保存所有试验到CSV
        all_trials_data = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE and trial.values:
                trial_data = {
                    'trial_number': trial.number,
                    'sto_r2_trial': trial.values[0],
                    'secondary_score': trial.values[1],
                    'stability_score': trial.values[2],
                }
                for param_name, param_value in trial.params.items():
                    trial_data[param_name] = param_value
                all_trials_data.append(trial_data)

        if all_trials_data:
            df_trials = pd.DataFrame(all_trials_data)
            df_trials = df_trials.sort_values('trial_number')
            df_trials.to_csv(self.output_dir / "all_trials_results.csv", index=False, encoding='utf-8')
            logger.info(f"Saved {len(all_trials_data)} trial results to CSV")

        logger.info(f"Tuning results saved to {self.output_dir}")

    def _create_visualizations_from_saved_data(self, solution_dir: Path, solution_num: int):
        """从已保存的数据创建可视化"""
        try:
            viz_dir = solution_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)

            # 读取性能数据
            perf_data = joblib.load(solution_dir / "performance_data.joblib")

            # 读取预测结果
            pred_df = pd.read_csv(solution_dir / "validation_predictions.csv")

            # 创建验证集预测图
            visualizer = ModelVisualizer()
            fig = visualizer.plot_predictions(
                pred_df['true_' + self.target_column].values,
                pred_df['predicted_' + self.target_column].values,
                pred_df['uncertainty'].values,
                full_r2=perf_data['validation_performance']['r2_score'],
                sto_r2=perf_data['validation_performance']['sto_r2'],
                save_path=str(viz_dir / "val_predictions.png"),
                sto_flags=pred_df['is_sto'].values,
                sample_weights=pred_df['weight'].values
            )
            if fig:
                plt.close(fig)

            logger.info(f"Visualizations created for solution #{solution_num}")

        except Exception as e:
            logger.warning(f"Could not create visualizations for solution #{solution_num}: {str(e)}")

    def _create_best_config(self, best_params: Dict) -> Any:
        """从最佳参数创建配置"""
        config = self.config_class()

        # XGBoost参数
        xgb_params = {k[4:]: v for k, v in best_params.items() if k.startswith('xgb_')}
        config.xgb_params.update(xgb_params)

        # BNN参数
        bnn_params = {}
        for k, v in best_params.items():
            if k.startswith('bnn_'):
                param_name = k[4:]
                if param_name.startswith('dropout_rate_'):
                    if 'dropout_rates' not in bnn_params:
                        bnn_params['dropout_rates'] = [0.0] * 3
                    idx = int(param_name.split('_')[-1])
                    if idx < len(bnn_params['dropout_rates']):
                        bnn_params['dropout_rates'][idx] = v
                else:
                    bnn_params[param_name] = v

        # 更新BNN配置
        if 'first_hidden_dims_pow' in bnn_params:
            config.first_hidden_dims_pow = bnn_params.pop('first_hidden_dims_pow')
        config.bnn_params.update(bnn_params)

        return config

    def tune_hyperparameters(self) -> Dict:
        """执行超参数调优"""
        logger.info("Starting multi-objective hyperparameter tuning with stability...")
        start_time = time.time()

        try:
            # 优化方向
            directions = [
                "maximize",  # STO R²
                "minimize" if self.tuning_config.multi_objective_type == 'loss' else "maximize",  # 损失或CV R²
                "maximize"  # 稳定性
            ]

            # 创建研究
            sampler = optuna.samplers.TPESampler(
                n_startup_trials=max(20, int(0.3 * self.tuning_config.n_trials)),
                seed=42,
                multivariate=True,  # 启用多变量采样
                constant_liar=True  # 允许并行优化
            )
            # ⭐ 启用剪枝器提前终止无希望的trial
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=5
            )
            study = optuna.create_study(
                directions=directions,
                sampler=sampler,
                pruner=pruner,  # 添加剪枝器
                study_name="tuning_with_stability"
            )
            self.study = study

            # 执行优化
            study.optimize(
                self._objective,
                n_trials=self.tuning_config.n_trials,
                timeout=self.tuning_config.timeout,
                n_jobs=1,  # Windows + CUDA 必须使用单进程
                show_progress_bar=self.tuning_config.show_progress
            )

            tuning_time = time.time() - start_time
            logger.info(f"Tuning completed in {tuning_time:.2f}s")

            # 构建结果
            tuning_results = {
                'best_params': self.best_params,
                'n_trials': len(study.trials),
                'tuning_time': tuning_time,
                'study': study,
                'history': self.tuning_history
            }

            # 保存结果
            self._save_tuning_results(tuning_results)

            # 可视化
            if study:
                self._visualize_tuning_results(study)

            logger.info("Tuning completed")
            return tuning_results

        except Exception as e:
            logger.error(f"Hyperparameter tuning failed: {str(e)}")
            import traceback
            traceback.print_exc()

            tuning_time = time.time() - start_time
            return {
                'best_params': {},
                'n_trials': 0,
                'tuning_time': tuning_time,
                'study': None,
                'history': self.tuning_history
            }


def create_tuning_pipeline():
    """创建调优管道"""

    def run_tuning_experiment(
            data_path: str,
            base_features: List[str],
            target_column: str,
            sto_column: str = 'Substrate',
            weight_config: Dict = None,
            tuning_config: TuningConfig = None,
            output_dir: str = "./tuning_results",
            visualization_output_dir: str = None,
            verbose: bool = True,
            show_progress: bool = True,
            model_class: Any = None,
            config_class: Any = None,
            use_boundary_weights: bool = True,
            boundary_config: Dict = None,
            sequence_good_path: str = "",
            sequence_bad_path: str = ""
    ):
        """运行调优实验"""
        if tuning_config is None:
            tuning_config = TuningConfig()
            tuning_config.n_trials = 50
            tuning_config.cv_folds = 3

        tuning_config.verbose_training = False
        tuning_config.verbose = verbose
        tuning_config.show_progress = show_progress

        tuner = None
        try:
            tuner = HyperparameterTuner(
                data_path=data_path,
                base_features=base_features,
                target_column=target_column,
                sto_column=sto_column,
                tuning_config=tuning_config,
                weight_config=weight_config,
                output_dir=output_dir,
                visualization_output_dir=visualization_output_dir,
                model_class=model_class,
                config_class=config_class,
                use_boundary_weights=use_boundary_weights,
                boundary_config=boundary_config,
                sequence_good_path=sequence_good_path,
                sequence_bad_path=sequence_bad_path
            )

            tuning_results = tuner.tune_hyperparameters()

            results = {
                'tuner': tuner,
                'tuning_results': tuning_results
            }

            return results

        except Exception as e:
            logger.error(f"Tuning experiment failed: {str(e)}")
            raise
        finally:
            # 资源清理
            if tuner is not None:
                try:
                    logger.info("Starting resource cleanup...")
                    attrs_to_del = [
                        'train_data', 'val_data',
                        '_cached_weights', '_last_trained_model',
                        'fixed_covariance_inv'
                    ]
                    for attr in attrs_to_del:
                        if hasattr(tuner, attr):
                            try:
                                delattr(tuner, attr)
                            except:
                                pass

                    if hasattr(tuner, 'tuning_history'):
                        tuner.tuning_history.clear()
                    if hasattr(tuner, 'pareto_front'):
                        tuner.pareto_front = []

                    logger.info("Tuner attributes cleaned")
                except Exception as cleanup_error:
                    logger.warning(f"Non-critical cleanup error: {cleanup_error}")

                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        logger.info("GPU memory cleared")
                except:
                    pass

                for _ in range(3):
                    gc.collect()

                logger.info("Memory resources released")

    return run_tuning_experiment
