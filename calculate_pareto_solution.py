import os
import pandas as pd
import numpy as np
import shutil
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import warnings
import sys
from pathlib import Path
import argparse
import gc
from collections import OrderedDict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
from scipy import stats
from typing import Tuple, List, Optional
from matplotlib.lines import Line2D
from tqdm import tqdm

from utils.optim_prediction_utils import (
    CachedObjectiveFunction,
    BayesianOptimizer,
    create_and_save_all_visualizations,
    ParameterProcessor
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

# --- START OF combine_metrics.py CONTENT ---
def process_trial_data(mode, model_type):
    """
    Process trial data from specified folder, extract 5 key metrics, and save results.
    Parameters:
    folder_name (str): Name of the folder containing the data.
    """
    # Construct file paths
    base_path = f"./{mode}/XGB_BNN_{model_type}_hybrid_model"
    pretrain_path = f"{base_path}/pretrain/hyperparameter_tuning_results/all_trials_results.csv"
    finetune_path = f"{base_path}/fine-tune/batch_finetuned_models/performance_comparison.csv"
    output_dir = f"{base_path}/pareto_solution"  # 修改输出目录
    output_path = f"{output_dir}/combined_metrics.csv"

    # Check if input files exist
    if not os.path.exists(pretrain_path):
        print(f"Error: Pretraining results file does not exist: {pretrain_path}")
        return False  # Indicate failure
    if not os.path.exists(finetune_path):
        print(f"Error: Finetuning results file does not exist: {finetune_path}")
        return False  # Indicate failure

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Read pretraining results file
        pretrain_df = pd.read_csv(pretrain_path)
        print(f"Successfully read pretraining results file, {len(pretrain_df)} rows.")

        # Read finetuning results file
        finetune_df = pd.read_csv(finetune_path)
        print(f"Successfully read finetuning results file, {len(finetune_df)} rows.")

        # Check if necessary columns exist
        required_pretrain_cols = ['trial_number', 'sto_r2_trial', 'secondary_score', 'stability_score']
        required_finetune_cols = ['solution', 'experiment_data_r2', 'original_sto_r2']

        missing_pretrain = [col for col in required_pretrain_cols if col not in pretrain_df.columns]
        missing_finetune = [col for col in required_finetune_cols if col not in finetune_df.columns]

        if missing_pretrain:
            print(f"Error: Missing columns in pretraining results file: {missing_pretrain}")
            return False  # Indicate failure
        if missing_finetune:
            print(f"Error: Missing columns in finetuning results file: {missing_finetune}")
            return False  # Indicate failure

        # Extract trial_number from the 'solution' column in finetuning results
        finetune_df['trial_number'] = finetune_df['solution'].str.extract(r'trial_(\d+)').astype(int)

        # Merge the two DataFrames
        merged_df = pd.merge(
            pretrain_df[required_pretrain_cols],
            finetune_df[['trial_number', 'experiment_data_r2', 'original_sto_r2']],
            on='trial_number',
            how='inner'  # Only keep trials present in both files
        )

        print(f"Successfully merged data, {len(merged_df)} trials.")

        # Save the combined results
        merged_df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
        print(f"Extracted metrics: {list(merged_df.columns)}")

        return True  # Indicate success

    except Exception as e:
        print(f"An error occurred during processing: {str(e)}")
        return False  # Indicate failure

# --- END OF combine_metrics.py CONTENT ---

# --- START OF calculate_pareto_solution.py CONTENT ---
# Optimized graphical style settings
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.unicode_minus': False,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14
})

# Color scheme
COLORS = {
    'non_pareto': '#E0E0E0',
    'pareto_all': '#3498DB',
    'pareto_selected': '#E74C3C',
    'primary': '#2C3E50',
    'secondary': '#95A5A6'
}

class ParetoAnalyzer:
    """Pareto Frontier Analyzer"""
    def __init__(self, mode: str, model_type: str, secondary_score_direction: str = 'max'):
        self.mode = mode
        self.model_type = model_type
        self.secondary_score_direction = secondary_score_direction.lower()
        base_path = f"./{mode}/XGB_BNN_{model_type}_hybrid_model"
        self.folder_name = base_path
        self.input_path = f"{base_path}/pareto_solution/combined_metrics.csv"
        self.output_dir = f"{base_path}/pareto_solution"
        self.viz_dir = f"{self.output_dir}/visualizations"

        # Objective columns and optimization directions
        self.objective_columns = [
            'sto_r2_trial', 'secondary_score', 'stability_score',
            'experiment_data_r2', 'original_sto_r2'
        ]
        self._set_maximize_mask()

    def _set_maximize_mask(self):
        """Set optimization direction mask."""
        if self.secondary_score_direction == 'max':
            self.maximize_mask = np.array([True, True, True, True, True])
        elif self.secondary_score_direction == 'min':
            self.maximize_mask = np.array([True, False, True, True, True])
        else:
            raise ValueError(f"Invalid direction: {self.secondary_score_direction}")

    def load_data(self) -> Optional[pd.DataFrame]:
        """Load and validate data."""
        if not os.path.exists(self.input_path):
            print(f"❌ Error: Input file does not exist: {self.input_path}")
            return None

        try:
            df = pd.read_csv(self.input_path)
            print(f"✓ Successfully loaded data, {len(df)} rows.")

            # Validate required columns
            required_cols = ['trial_number'] + self.objective_columns
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                print(f"❌ Missing columns: {missing_cols}")
                return None

            # Handle missing values
            valid_mask = ~df[self.objective_columns].isnull().any(axis=1)
            if not valid_mask.all():
                n_invalid = (~valid_mask).sum()
                print(f"⚠ Removed {n_invalid} rows containing null values.")
                df = df[valid_mask].reset_index(drop=True)

            return df

        except Exception as e:
            print(f"❌ Failed to read data: {str(e)}")
            return None

    @staticmethod
    def dominates(row1: np.ndarray, row2: np.ndarray, maximize_mask: np.ndarray) -> bool:
        """Check if solution 1 dominates solution 2."""
        row1_adj = np.where(maximize_mask, row1, -row1)
        row2_adj = np.where(maximize_mask, row2, -row2)

        all_better_or_equal = np.all(row1_adj >= row2_adj)
        any_strictly_better = np.any(row1_adj > row2_adj)

        return all_better_or_equal and any_strictly_better

    def find_pareto_frontier(self, data: np.ndarray) -> np.ndarray:
        """Find the Pareto frontier."""
        n = data.shape[0]
        is_pareto = np.ones(n, dtype=bool)

        for i in range(n):
            if is_pareto[i]:
                for j in range(i + 1, n):
                    if is_pareto[j]:
                        if self.dominates(data[j], data[i], self.maximize_mask):
                            is_pareto[i] = False
                            break
                        elif self.dominates(data[i], data[j], self.maximize_mask):
                            is_pareto[j] = False

        return is_pareto

    def load_sobol_optimal(self, sobol_csv_path: str = './data/sobol_samples_results.csv') -> Optional[dict]:
        """加载Sobol采样中RRR最高的实验条件"""
        try:
            sobol_df = pd.read_csv(sobol_csv_path)
            if 'RRR' not in sobol_df.columns:
                print(f"❌ Error: 'RRR' column not found in {sobol_csv_path}")
                return None

            # 找到RRR最高的行
            max_rrr_idx = sobol_df['RRR'].idxmax()
            optimal_row = sobol_df.loc[max_rrr_idx]

            # 提取参数
            sobol_optimal = {
                'oxygen_pressure': optimal_row['oxygen_pressure'],
                'laser_energy_density': optimal_row['laser_energy_density'],
                'temperature': optimal_row['temperature'],
                'frequency': optimal_row['frequency'],
                'thickness': optimal_row['thickness'],
                'RRR': optimal_row['RRR']
            }

            print(f"✓ Loaded Sobol optimal point (RRR={sobol_optimal['RRR']:.4f}):")
            print(f"  O2={sobol_optimal['oxygen_pressure']:.4f}, Laser={sobol_optimal['laser_energy_density']:.2f}, "
                  f"Temp={sobol_optimal['temperature']:.1f}, Freq={sobol_optimal['frequency']:.0f}, "
                  f"Thick={sobol_optimal['thickness']:.2f}")

            return sobol_optimal

        except Exception as e:
            print(f"❌ Failed to load Sobol optimal point: {str(e)}")
            return None

    def filter_pareto_by_grid_search(
            self,
            pareto_df: pd.DataFrame,
            sobol_optimal: dict,
            eta: float = 0.1,
            grid_points_per_dim: int = 5,
            distance_threshold: float = 0.3,
            use_gpu: bool = True,
            exceed_ratio_threshold: float = 0.3
    ) -> pd.DataFrame:
        """
        通过网格搜索筛选帕累托解

        对每个帕累托解f_i，在参数空间中进行网格搜索。
        如果存在远离Sobol最优点S的格点s，使得f_i(s) > f_i(S) + η*f_i(S)，则排除该帕累托解。

        Args:
            pareto_df: 帕累托解DataFrame
            sobol_optimal: Sobol最优点字典
            eta: 阈值，用于判断是否排除帕累托解
            grid_points_per_dim: 每个维度的网格点数
            distance_threshold: 距离阈值（归一化空间中的欧氏距离），超过此距离的格点被认为是"远离S"
            use_gpu: 是否使用GPU加速距离计算

        Returns:
            筛选后的帕累托解DataFrame
        """
        print(f"\n🔍 开始网格搜索筛选")
        print(f"  参数: η={eta}, 每维网格点数={grid_points_per_dim}, 归一化距离阈值={distance_threshold}, 超过比例阈值={exceed_ratio_threshold}")
        print(f"  GPU加速: {'启用' if use_gpu and torch.cuda.is_available() else '禁用'}")

        # 参数边界（从model_evaluation.py的BOUND导入）
        param_bounds = {
            'oxygen_pressure': (0.0001, 0.5),
            'laser_energy_density': (1.5, 3.0),
            'temperature': (500, 750),
            'frequency': (4, 10),
            'thickness': (30, 200)
        }

        # 生成网格点（使用numpy向量化加速）
        print(f"\n📊 生成网格点...")
        total_points = grid_points_per_dim ** 5
        print(f"  预计生成 {total_points:,} 个网格点")

        if total_points > 100_000_000:
            print(f"  ⚠️  警告: 网格点数量过大 ({total_points:,})，可能导致内存不足")
            print(f"  建议: 减少 grid_points_per_dim (当前={grid_points_per_dim})")
            response = input("  是否继续? (y/n): ")
            if response.lower() != 'y':
                print("  用户取消操作")
                return pareto_df

        grid_points = self._generate_grid_points_fast(param_bounds, grid_points_per_dim)
        print(f"  ✓ 生成了 {len(grid_points):,} 个网格点")

        # 计算Sobol最优点的归一化坐标（用于计算距离）
        sobol_normalized = self._normalize_point(sobol_optimal, param_bounds)
        sobol_normalized_array = np.array(list(sobol_normalized.values()))

        # 向量化筛选远离Sobol最优点的格点（使用GPU加速）
        print(f"\n🔍 筛选远离Sobol最优点的格点...")
        distant_grid_points, distances = self._filter_distant_points_fast(
            grid_points, param_bounds, sobol_normalized_array, distance_threshold, use_gpu=use_gpu
        )

        print(f"  ✓ 筛选出 {len(distant_grid_points):,} 个远离Sobol最优点的格点")
        if len(distances) > 0:
            print(f"  距离统计: 最小={distances.min():.4f}, 最大={distances.max():.4f}, 平均={distances.mean():.4f}")
            print(f"  归一化空间: 所有距离都在[0, √5]范围内 (5维空间对角线长度)")

        if len(distant_grid_points) == 0:
            print("  ⚠ 没有远离Sobol最优点的格点，跳过筛选")
            return pareto_df

        # 对每个帕累托解进行验证，并保存网格搜索采样结果
        print(f"\n🔬 验证 {len(pareto_df)} 个帕累托解...")
        valid_trials = []
        excluded_trials = []
        grid_search_samples = {}  # 存储每个trial的网格搜索采样结果

        for idx, row in tqdm(pareto_df.iterrows(), total=len(pareto_df), desc="验证帕累托解"):
            trial_number = int(row['trial_number'])

            # 加载该trial的模型并评估
            is_valid, trial_samples = self._validate_pareto_solution(
                trial_number, sobol_optimal, distant_grid_points, eta, exceed_ratio_threshold
            )

            if is_valid:
                valid_trials.append(trial_number)
                grid_search_samples[trial_number] = trial_samples
            else:
                excluded_trials.append(trial_number)

        print(f"\n📊 筛选结果:")
        print(f"  ✓ 保留 {len(valid_trials)} 个帕累托解")
        print(f"  ✗ 排除 {len(excluded_trials)} 个帕累托解")

        if excluded_trials:
            print(f"  排除的trial: {excluded_trials}")

        # 保存网格搜索采样结果到文件
        if grid_search_samples:
            self._save_grid_search_samples(grid_search_samples)

        # 返回筛选后的DataFrame
        filtered_df = pareto_df[pareto_df['trial_number'].isin(valid_trials)].copy()
        return filtered_df

    def _generate_grid_points(self, param_bounds: dict, points_per_dim: int) -> list:
        """生成参数空间的网格点（保留用于兼容性）"""
        return self._generate_grid_points_fast(param_bounds, points_per_dim)

    def _generate_grid_points_fast(self, param_bounds: dict, points_per_dim: int) -> list:
        """使用numpy向量化快速生成参数空间的网格点"""
        param_names = list(param_bounds.keys())
        grid_values = []

        for param in param_names:
            min_val, max_val = param_bounds[param]
            grid_values.append(np.linspace(min_val, max_val, points_per_dim))

        # 使用numpy meshgrid加速生成所有组合
        from itertools import product
        grid_points = []
        for values in product(*grid_values):
            point = {param: val for param, val in zip(param_names, values)}
            grid_points.append(point)

        return grid_points

    def _filter_distant_points_fast(self, grid_points: list, param_bounds: dict,
                                     sobol_normalized_array: np.ndarray,
                                     distance_threshold: float,
                                     use_gpu: bool = True,
                                     batch_size: int = 1_000_000) -> tuple:
        """
        向量化筛选远离Sobol最优点的格点，支持GPU加速和批处理

        Args:
            grid_points: 网格点列表
            param_bounds: 参数边界
            sobol_normalized_array: Sobol最优点的归一化坐标数组
            distance_threshold: 归一化空间中的距离阈值
            use_gpu: 是否使用GPU加速
            batch_size: 批处理大小，避免GPU内存溢出

        Returns:
            (distant_points, all_distances): 远离的点列表和所有距离数组
        """
        param_names = list(param_bounds.keys())
        n_points = len(grid_points)
        n_dims = len(param_names)

        # 检查是否使用GPU
        use_gpu = use_gpu and torch.cuda.is_available()
        device = torch.device('cuda' if use_gpu else 'cpu')

        if use_gpu:
            print(f"  使用GPU加速 (批处理大小: {batch_size:,})")
        else:
            print(f"  使用CPU计算")

        # 预分配数组
        normalized_points = np.zeros((n_points, n_dims), dtype=np.float32)

        # 归一化所有点
        print(f"  归一化 {n_points:,} 个网格点...")
        for i, point in enumerate(grid_points):
            for j, param in enumerate(param_names):
                min_val, max_val = param_bounds[param]
                normalized_points[i, j] = (point[param] - min_val) / (max_val - min_val)

        if use_gpu:
            # GPU批处理计算距离
            all_distances = []
            distant_indices = []

            # 转换Sobol点到GPU
            sobol_tensor = torch.from_numpy(sobol_normalized_array.astype(np.float32)).to(device)

            n_batches = (n_points + batch_size - 1) // batch_size
            print(f"  GPU批处理计算距离 ({n_batches} 批次)...")

            for batch_idx in tqdm(range(n_batches), desc="GPU距离计算"):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_points)

                # 转换批次到GPU
                batch_points = torch.from_numpy(normalized_points[start_idx:end_idx]).to(device)

                # 计算距离
                batch_distances = torch.norm(batch_points - sobol_tensor, dim=1)

                # 筛选远离点
                distant_mask = batch_distances > distance_threshold
                batch_distant_indices = torch.where(distant_mask)[0].cpu().numpy() + start_idx

                # 保存结果
                all_distances.append(batch_distances.cpu().numpy())
                distant_indices.extend(batch_distant_indices.tolist())

                # 清理GPU内存
                del batch_points, batch_distances, distant_mask
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()

            # 合并所有距离
            distances = np.concatenate(all_distances)

            # 清理GPU内存
            torch.cuda.empty_cache()

        else:
            # CPU向量化计算所有点到Sobol最优点的欧氏距离
            print(f"  CPU向量化计算距离...")
            distances = np.linalg.norm(normalized_points - sobol_normalized_array, axis=1)

            # 筛选距离大于阈值的点
            distant_mask = distances > distance_threshold
            distant_indices = np.where(distant_mask)[0]

        # 提取远离的点
        distant_points = [grid_points[i] for i in distant_indices]
        distant_distances = distances[distant_indices] if len(distant_indices) > 0 else np.array([])

        return distant_points, distant_distances

    def _normalize_point(self, point: dict, param_bounds: dict) -> dict:
        """将参数点归一化到[0, 1]，只处理param_bounds中定义的参数"""
        normalized = {}
        for param in param_bounds.keys():
            if param in point:
                min_val, max_val = param_bounds[param]
                normalized[param] = (point[param] - min_val) / (max_val - min_val)
        return normalized

    def _validate_pareto_solution(
            self,
            trial_number: int,
            sobol_optimal: dict,
            distant_grid_points: list,
            eta: float,
            exceed_ratio_threshold: float = 0.3
    ) -> tuple:
        """
        验证单个帕累托解是否有效

        加载模型，评估在Sobol最优点和远离格点的预测值。
        如果存在格点s使得f_i(s) > f_i(S) + η，返回False（排除）。

        Returns:
            (is_valid, samples): is_valid表示是否保留该解，samples为网格搜索采样结果列表
        """
        samples = []  # 存储采样结果 [point_dict, mean_val, std_val]

        try:
            # 加载模型
            # 路径结构: {output_dir}/trial_{trial_number}/model
            # copy_model_folders 现在直接复制到 { output_dir}/trial_{trial_number}
            model_path = f"{self.output_dir}/trial_{trial_number}/model"
            if not os.path.exists(model_path):
                print(f"    ⚠ Trial {trial_number}: 模型路径不存在 ({model_path})，保留")
                return True, samples

            # 确保 pickle 能找到 data_processer 模块
            import utils.data_processer as data_processer
            sys.modules['data_processer'] = data_processer

            # 动态导入模型类并设置模块别名
            if self.model_type == 'attention':
                from tradition.XGB_BNN_attention_hybrid_model import model as model_module
                from tradition.XGB_BNN_attention_hybrid_model.model import HybridModel
            elif self.model_type == 'series':
                from tradition.XGB_BNN_series_hybrid_model import model as model_module
                from tradition.XGB_BNN_series_hybrid_model.model import HybridModel
            elif self.model_type == 'uncertainty_1':
                from tradition.XGB_BNN_uncertainty_1_hybrid_model import model as model_module
                from tradition.XGB_BNN_uncertainty_1_hybrid_model.model import HybridModel
            elif self.model_type == 'uncertainty_2':
                from tradition.XGB_BNN_uncertainty_2_hybrid_model import model as model_module
                from tradition.XGB_BNN_uncertainty_2_hybrid_model.model import HybridModel
            else:
                print(f"    ⚠ Trial {trial_number}: 未知的模型类型 {self.model_type}，保留")
                return True, samples

            # 设置模块别名，让 pickle 能找到
            sys.modules['model'] = model_module

            # 加载模型
            model = HybridModel.load_model(model_path)

            # 列名映射：从小写转换为模型期望的格式
            column_mapping = {
                'oxygen_pressure': 'Oxygen pressure',
                'laser_energy_density': 'Laser energy density',
                'temperature': 'Temperature',
                'frequency': 'Frequency',
                'thickness': 'Thickness'
            }

            # --- 修正: 定义 sobol_df ---
            sobol_mapped = {column_mapping[k]: v for k, v in sobol_optimal.items() if k in column_mapping}
            sobol_df = pd.DataFrame([sobol_mapped])
            # --- 修正结束 ---

            # 在Sobol最优点评估
            f_S_mean, f_S_std = model.predict(sobol_df, n_samples=30)
            f_S_mean = float(f_S_mean[0])
            f_S_std = float(f_S_std[0])
            samples.append((sobol_optimal.copy(), f_S_mean, f_S_std))

            # 在远离格点评估，统计超过阈值的格点比例
            n_exceed = 0
            n_total_checked = 0  # 记录已检查的格点总数
            total_distant_points = len(distant_grid_points)

            # 计算需要达到多少个点才能超过阈值并导致排除
            min_points_to_exclude = int(np.ceil(exceed_ratio_threshold * total_distant_points))

            for point in distant_grid_points:
                point_mapped = {column_mapping[k]: v for k, v in point.items() if k in column_mapping}
                point_df = pd.DataFrame([point_mapped])
                f_s_mean, f_s_std = model.predict(point_df, n_samples=30)
                f_s_mean = float(f_s_mean[0])
                f_s_std = float(f_s_std[0])
                samples.append((point.copy(), f_s_mean, f_s_std))

                n_total_checked += 1
                if f_s_mean > f_S_mean + eta * f_S_mean:
                    n_exceed += 1
                    # 如果超过阈值的点数已经达到了可以排除的数量，则立即停止
                    if n_exceed >= min_points_to_exclude:
                        print(
                            f"    ✗ Trial {trial_number}: Early stop at {n_total_checked}/{total_distant_points} points. "
                            f"{n_exceed}/{total_distant_points} ({n_exceed / total_distant_points:.1%}) "
                            f"points exceeded threshold. Excluding solution.")
                        return False, samples

            # 循环结束仍未达到排除阈值，说明该解有效
            exceed_ratio = n_exceed / total_distant_points if total_distant_points > 0 else 0.0
            print(f"    ✓ Trial {trial_number}: Checked all {total_distant_points} points. "
                  f"{n_exceed}/{total_distant_points} ({exceed_ratio:.1%}) exceeded threshold. Keeping solution.")
            return True, samples

        except Exception as e:
            print(f"    ⚠ Trial {trial_number}: 验证失败 ({str(e)})，保留")
            import traceback
            traceback.print_exc()
            return True, samples

    def _save_grid_search_samples(self, grid_search_samples: dict):
        """
        保存网格搜索采样结果到文件

        Args:
            grid_search_samples: {trial_number: [(point_dict, mean, std), ...]}
        """
        if not grid_search_samples:
            print("\n⚠ 没有网格搜索采样结果需要保存")
            return

        save_dir = Path(self.output_dir) / "grid_search_samples"
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n💾 保存网格搜索采样结果到 {save_dir}")

        total_samples = 0
        for trial_number, samples in grid_search_samples.items():
            if len(samples) == 0:
                continue

            # 转换为DataFrame格式
            sample_data = []
            for point_dict, mean_val, std_val in samples:
                sample_data.append({
                    'oxygen_pressure': point_dict['oxygen_pressure'],
                    'laser_energy_density': point_dict['laser_energy_density'],
                    'temperature': point_dict['temperature'],
                    'frequency': point_dict['frequency'],
                    'thickness': point_dict['thickness'],
                    'mean': mean_val,
                    'std': std_val
                })

            if len(sample_data) > 0:
                df = pd.DataFrame(sample_data)
                save_path = save_dir / f"trial_{trial_number}_grid_samples.csv"
                df.to_csv(save_path, index=False)
                total_samples += len(sample_data)
                print(f"  ✓ trial_{trial_number}: {len(sample_data)} 个采样点")

        print(f"\n  总计保存 {total_samples} 个网格搜索采样点")

    def select_representative_solutions(
            self,
            pareto_df: pd.DataFrame,
            target_max: int = 20,
            enable_grid_filter: bool = True,
            sobol_csv_path: str = './data/sobol_samples_results.csv',
            eta: float = 0.1,
            grid_points_per_dim: int = 5,
            distance_threshold: float = 0.3,
            use_gpu: bool = True,
            exceed_ratio_threshold: float = 0.3
    ) -> pd.DataFrame:
        """Select representative Pareto solutions with optional grid search filtering."""
        # 网格搜索筛选
        if enable_grid_filter:
            sobol_optimal = self.load_sobol_optimal(sobol_csv_path)
            if sobol_optimal is not None:
                pareto_df = self.filter_pareto_by_grid_search(
                    pareto_df, sobol_optimal, eta=eta, grid_points_per_dim=grid_points_per_dim,
                    distance_threshold=distance_threshold, use_gpu=use_gpu,
                    exceed_ratio_threshold=exceed_ratio_threshold
                )
                print(f"\n✓ 网格搜索筛选后剩余 {len(pareto_df)} 个帕累托解")
            else:
                print("⚠ 无法加载Sobol最优点，跳过网格搜索筛选")

        n_pareto = len(pareto_df)

        if n_pareto <= target_max:
            print(f"✓ Number of Pareto solutions {n_pareto} is within limit (max={target_max}).")
            return pareto_df

        print(f"⚠ Number of Pareto solutions {n_pareto} exceeds maximum, using clustering to select {target_max} representatives.")
        return self._cluster_selection(pareto_df, target_max)

    def _cluster_selection(self, pareto_df: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
        """Use clustering to select representative solutions."""
        objective_data = pareto_df[self.objective_columns].values
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(objective_data)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)

        selected_indices = []
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            if cluster_mask.any():
                cluster_points = scaled_data[cluster_mask]
                cluster_center = kmeans.cluster_centers_[cluster_id]
                distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
                closest_idx = np.argmin(distances)
                original_idx = pareto_df.index[cluster_mask][closest_idx]
                selected_indices.append(original_idx)

        return pareto_df.loc[selected_indices].sort_values('trial_number')

    def get_direction_labels(self) -> List[str]:
        """Get labels with optimization direction."""
        labels = []
        for i, col in enumerate(self.objective_columns):
            direction = "max" if self.maximize_mask[i] else "min"
            labels.append(f"{col}\n({direction})")
        return labels


class ParetoVisualizer:
    """Pareto Solution Visualizer"""
    def __init__(self, analyzer: ParetoAnalyzer):
        self.analyzer = analyzer
        self.viz_dir = analyzer.viz_dir
        os.makedirs(self.viz_dir, exist_ok=True)

    def create_all_visualizations(
            self,
            df: pd.DataFrame,
            pareto_df: pd.DataFrame,
            selected_df: pd.DataFrame
    ):
        """Create all visualizations."""
        print("\n📊 Creating visualization charts...")

        viz_methods = [
            (self._create_pairplot, "Scatter Plot Matrix"),
            (self._create_parallel_coordinates, "Parallel Coordinates"),
            (self._create_radar_chart, "Radar Chart"),
            (self._create_distributions, "Distribution Histograms"),
            (self._create_statistics, "Statistics Chart"),
            (self._create_heatmap, "Correlation Heatmap")
        ]

        for method, name in viz_methods:
            try:
                method(df, pareto_df, selected_df)
                print(f"  ✓ {name}")
            except Exception as e:
                print(f"  ✗ {name}: {str(e)}")

        print(f"\n✓ All charts saved to: {self.viz_dir}")

    def _prepare_plot_data(self, df: pd.DataFrame, pareto_df: pd.DataFrame,
                           selected_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare plotting data."""
        plot_data = df[self.analyzer.objective_columns].copy()
        plot_data['Type'] = 'Non-Pareto'
        plot_data.loc[pareto_df.index, 'Type'] = 'Pareto (All)'
        plot_data.loc[selected_df.index, 'Type'] = 'Pareto (Selected)'
        plot_data['Size'] = 30
        plot_data.loc[pareto_df.index, 'Size'] = 60
        plot_data.loc[selected_df.index, 'Size'] = 100
        return plot_data

    def _create_pairplot(self, df: pd.DataFrame, pareto_df: pd.DataFrame,
                         selected_df: pd.DataFrame):
        """Create an improved scatter plot matrix."""
        plot_data = self._prepare_plot_data(df, pareto_df, selected_df)

        g = sns.pairplot(
            plot_data,
            hue='Type',
            hue_order=['Non-Pareto', 'Pareto (All)', 'Pareto (Selected)'],
            palette={
                'Non-Pareto': COLORS['non_pareto'],
                'Pareto (All)': COLORS['pareto_all'],
                'Pareto (Selected)': COLORS['pareto_selected']
            },
            diag_kind='kde',
            plot_kws={'alpha': 0.6, 's': 50},
            diag_kws={'alpha': 0.7, 'linewidth': 2}
        )

        g.fig.suptitle(
            f'{self.analyzer.folder_name} - Pareto Solution Distribution',
            y=1.02, fontsize=16, fontweight='bold'
        )
        plt.savefig(f"{self.viz_dir}/01_pareto_pairplot.png")
        plt.close()

    def _create_parallel_coordinates(self, df: pd.DataFrame, pareto_df: pd.DataFrame,
                                     selected_df: pd.DataFrame):
        """Create an improved parallel coordinates plot."""
        fig, ax = plt.subplots(figsize=(14, 7))

        # Prepare normalized data
        scaler = MinMaxScaler()
        scaled_data = df[self.analyzer.objective_columns].copy()
        scaled_data[self.analyzer.objective_columns] = scaler.fit_transform(
            scaled_data[self.analyzer.objective_columns]
        )

        # Plot non-Pareto solutions (transparent gray)
        for idx in df.index:
            if idx not in pareto_df.index:
                ax.plot(scaled_data.loc[idx],
                        color=COLORS['non_pareto'], alpha=0.2, linewidth=0.5)

        # Plot all Pareto solutions (blue)
        for idx in pareto_df.index:
            if idx not in selected_df.index:
                ax.plot(scaled_data.loc[idx],
                        color=COLORS['pareto_all'], alpha=0.5, linewidth=1.5)

        # Plot selected Pareto solutions (red, thicker)
        for idx in selected_df.index:
            ax.plot(scaled_data.loc[idx],
                    color=COLORS['pareto_selected'], alpha=0.8, linewidth=2.5)

        # Set axes
        direction_labels = self.analyzer.get_direction_labels()
        ax.set_xticks(range(len(direction_labels)))
        ax.set_xticklabels(direction_labels, rotation=15, ha='right')
        ax.set_ylabel('Normalized Value', fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        # Add legend

        legend_elements = [
            Line2D([0], [0], color=COLORS['non_pareto'], lw=2, alpha=0.5, label='Non-Pareto'),
            Line2D([0], [0], color=COLORS['pareto_all'], lw=2, alpha=0.7, label='Pareto (All)'),
            Line2D([0], [0], color=COLORS['pareto_selected'], lw=3, label='Pareto (Selected)')
        ]
        ax.legend(handles=legend_elements, loc='best')

        plt.title(f'{self.analyzer.folder_name} - Parallel Coordinates',
                  fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f"{self.viz_dir}/02_parallel_coordinates.png")
        plt.close()

    def _create_radar_chart(self, df: pd.DataFrame, pareto_df: pd.DataFrame,
                            selected_df: pd.DataFrame):
        """Create an improved radar chart."""
        max_display = min(10, len(selected_df))
        display_df = selected_df.head(max_display)

        # Normalize data
        scaler = MinMaxScaler()
        radar_data = scaler.fit_transform(display_df[self.analyzer.objective_columns])

        # Set angles
        n_vars = len(self.analyzer.objective_columns)
        angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
        angles += angles[:1]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))

        # Use gradient colors
        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, max_display))

        for i in range(max_display):
            values = radar_data[i].tolist() + [radar_data[i][0]]
            trial_num = display_df.iloc[i]['trial_number']

            ax.plot(angles, values, 'o-', linewidth=2.5,
                    label=f'Trial {trial_num}', color=colors[i], markersize=6)
            ax.fill(angles, values, alpha=0.15, color=colors[i])

        # Set labels
        direction_labels = self.analyzer.get_direction_labels()
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(direction_labels, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_rlabel_position(30)
        ax.grid(True, alpha=0.3)

        plt.title(f'Pareto Solutions Radar Chart (Top {max_display})',
                  fontsize=14, fontweight='bold', pad=25)
        plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.05), fontsize=9)
        plt.tight_layout()
        plt.savefig(f"{self.viz_dir}/03_radar_chart.png")
        plt.close()

    def _create_distributions(self, df: pd.DataFrame, pareto_df: pd.DataFrame,
                              selected_df: pd.DataFrame):
        """Create improved distribution histograms."""
        n_objectives = len(self.analyzer.objective_columns)
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()

        direction_labels = self.analyzer.get_direction_labels()

        for i, col in enumerate(self.analyzer.objective_columns):
            if i < len(axes):
                # Use KDE instead of plain histogram
                axes[i].hist(df[col], bins=30, alpha=0.4,
                             label='All Solutions', color=COLORS['non_pareto'], density=True)

                if len(pareto_df) > 0:
                    axes[i].hist(pareto_df[col], bins=20, alpha=0.5,
                                 label='Pareto Solutions', color=COLORS['pareto_all'], density=True)

                if len(selected_df) > 0:
                    axes[i].hist(selected_df[col], bins=15, alpha=0.7,
                                 label='Selected Solutions', color=COLORS['pareto_selected'], density=True)

                # Add KDE curves
                try:
                    kde = stats.gaussian_kde(df[col].dropna())
                    x_range = np.linspace(df[col].min(), df[col].max(), 100)
                    axes[i].plot(x_range, kde(x_range), 'k-', linewidth=2, alpha=0.5)
                except:
                    pass

                axes[i].set_title(direction_labels[i], fontsize=11, fontweight='bold')
                axes[i].set_xlabel('Value', fontsize=10)
                axes[i].set_ylabel('Density', fontsize=10)
                axes[i].legend(fontsize=8)
                axes[i].grid(True, alpha=0.3)

        # Hide extra subplots
        for i in range(n_objectives, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(f'{self.analyzer.folder_name} - Objective Distributions',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.viz_dir}/04_distributions.png")
        plt.close()

    def _create_statistics(self, df: pd.DataFrame, pareto_df: pd.DataFrame,
                           selected_df: pd.DataFrame):
        """Create improved statistical bar chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Solution count statistics
        categories = ['Total\nSolutions', 'Pareto\nSolutions', 'Selected\nSolutions']
        values = [len(df), len(pareto_df), len(selected_df)]
        colors = [COLORS['non_pareto'], COLORS['pareto_all'], COLORS['pareto_selected']]

        bars = ax1.bar(categories, values, color=colors, edgecolor='black',
                       linewidth=1.5, alpha=0.8)

        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, height + max(values) * 0.02,
                     f'{value}', ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Solution Count Statistics', fontsize=13, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        # Pareto ratio pie chart
        sizes = [len(df) - len(pareto_df), len(pareto_df)]
        labels = ['Non-Pareto', 'Pareto']
        colors_pie = [COLORS['non_pareto'], COLORS['pareto_all']]
        explode = (0, 0.1)

        ax2.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
        ax2.set_title('Pareto vs Non-Pareto', fontsize=13, fontweight='bold')

        plt.suptitle(f'{self.analyzer.folder_name} - Solution Statistics',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.viz_dir}/05_statistics.png")
        plt.close()

    def _create_heatmap(self, df: pd.DataFrame, pareto_df: pd.DataFrame,
                        selected_df: pd.DataFrame):
        """Create objective correlation heatmap."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Correlation for all solutions
        corr_all = df[self.analyzer.objective_columns].corr()
        sns.heatmap(corr_all, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, vmin=-1, vmax=1, square=True, ax=ax1,
                    cbar_kws={'label': 'Correlation'})
        ax1.set_title('All Solutions', fontsize=12, fontweight='bold')

        # Correlation for Pareto solutions
        if len(pareto_df) > 1:
            corr_pareto = pareto_df[self.analyzer.objective_columns].corr()
            sns.heatmap(corr_pareto, annot=True, fmt='.2f', cmap='coolwarm',
                        center=0, vmin=-1, vmax=1, square=True, ax=ax2,
                        cbar_kws={'label': 'Correlation'})
            ax2.set_title('Pareto Solutions', fontsize=12, fontweight='bold')

        plt.suptitle(f'{self.analyzer.folder_name} - Objective Correlations',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.viz_dir}/06_correlation_heatmap.png")
        plt.close()


def copy_model_folders(selected_df: pd.DataFrame, mode: str, model_type: str, output_dir: str) -> int:
    """Copy model folders."""
    # output_dir already contains /pareto_solution, so use it directly
    pareto_models_dir = output_dir
    os.makedirs(pareto_models_dir, exist_ok=True)
    print("\n📁 Copying Pareto solution model folders...")
    copied_count = 0
    base_path = f"./{mode}/XGB_BNN_{model_type}_hybrid_model"

    for trial_num in selected_df['trial_number']:
        source_dir = f"{base_path}/fine-tune/batch_finetuned_models/trial_{trial_num}"
        target_dir = f"{pareto_models_dir}/trial_{trial_num}"

        if os.path.exists(source_dir):
            try:
                if os.path.exists(target_dir):
                    shutil.rmtree(target_dir)
                shutil.copytree(source_dir, target_dir)
                copied_count += 1
                print(f"  ✓ trial_{trial_num}")
            except Exception as e:
                print(f"  ✗ trial_{trial_num}: {str(e)}")
        else:
            print(f"  ⚠ Source folder does not exist: trial_{trial_num}")

    print(f"\n✓ Copying finished: {copied_count}/{len(selected_df)} model folders")
    return copied_count


def print_summary(selected_df: pd.DataFrame, analyzer: ParetoAnalyzer):
    """Print Pareto solution summary."""
    print("\n" + "=" * 70)
    print("📊 PARETO OPTIMAL SOLUTION SUMMARY")
    print("=" * 70)
    print(f"\nTrial IDs: {sorted(selected_df['trial_number'].values)}")
    print(f"\nTotal of {len(selected_df)} Pareto optimal solutions\n")

    print("-" * 70)
    print(f"{'Objective Function': <25} {'Direction': <10} {'Mean': <15} {'Range'}")
    print("-" * 70)

    if len(selected_df) == 0:
        print("  (No solutions selected)")
        print("=" * 70 + "\n")
        return

    for i, col in enumerate(analyzer.objective_columns):
        direction = "max" if analyzer.maximize_mask[i] else "min"
        values = selected_df[col].values
        mean_val = np.mean(values)
        min_val = np.min(values)
        max_val = np.max(values)

        print(f"{col: <25} {direction: <10} {mean_val: >10.4f}      "
              f"[{min_val:.4f}, {max_val:.4f}]")

    print("=" * 70 + "\n")


def analyze_pareto_and_copy_models(mode: str, model_type: str, secondary_direction: str = 'max',
                                   max_sol: int = 20, create_viz: bool = True,
                                   enable_grid_filter: bool = True, eta: float = 0.1,
                                   grid_points_per_dim: int = 5, distance_threshold: float = 0.3,
                                   use_gpu: bool = True, exceed_ratio_threshold: float = 0.3):
    """Main function to analyze Pareto frontier and copy models.

    Args:
        mode: 'tradition' or 'LLM'
        model_type: Model type identifier
        secondary_direction: 'max' or 'min' for secondary score
        max_sol: Maximum number of solutions
        create_viz: Whether to create visualizations
        enable_grid_filter: Whether to enable grid search filtering
        eta: Threshold for grid search filtering
        grid_points_per_dim: Number of grid points per dimension
        distance_threshold: Distance threshold for "far from S"
        use_gpu: Whether to use GPU for grid search acceleration
        exceed_ratio_threshold: Fraction of distant grid points that must exceed eta threshold to exclude a solution
    """
    print("\n" + "=" * 70)
    print("🎯 Pareto Frontier Analysis & Model Copying Tool")
    print("=" * 70 + "\n")

    print(f"\n{'=' * 70}")
    print(f"Mode: {mode}")
    print(f"Model Type: {model_type}")
    print(f"Secondary Score Direction: {secondary_direction}")
    print(f"Pareto Solution Max: {max_sol}")
    print(f"Create Visualizations: {'Yes' if create_viz else 'No'}")
    print(f"Grid Search Filter: {'Enabled' if enable_grid_filter else 'Disabled'}")
    if enable_grid_filter:
        print(f"  η={eta}, grid_points={grid_points_per_dim}, distance_threshold={distance_threshold}, exceed_ratio_threshold={exceed_ratio_threshold}")
    print(f"{'=' * 70}\n")

    # Execute analysis
    try:
        analyzer = ParetoAnalyzer(mode, model_type, secondary_direction)

        # Load data
        df = analyzer.load_data()
        if df is None:
            return None # Indicate failure

        # Calculate Pareto frontier
        print("\n🔍 Calculating Pareto Frontier...")
        objective_data = df[analyzer.objective_columns].values
        pareto_mask = analyzer.find_pareto_frontier(objective_data)
        pareto_df = df[pareto_mask].copy()

        print(f"✓ Found {len(pareto_df)} Pareto optimal solutions.")

        # If grid filtering is enabled, copy all Pareto models first
        # so that _validate_pareto_solution can load them
        if enable_grid_filter:
            print("\n📁 Copying Pareto solution models for grid search validation...")
            copy_model_folders(pareto_df, mode, model_type, analyzer.output_dir)

        # Select representative solutions (with optional grid filtering)
        selected_df = analyzer.select_representative_solutions(
            pareto_df, target_max=max_sol,
            enable_grid_filter=enable_grid_filter,
            eta=eta,
            grid_points_per_dim=grid_points_per_dim,
            distance_threshold=distance_threshold,
            use_gpu=use_gpu,
            exceed_ratio_threshold=exceed_ratio_threshold
        )

        if len(selected_df) == 0:
            print("\n❌ All Pareto solutions were excluded by grid search filtering. "
                  "Try relaxing eta or distance_threshold.")
            # Clean up all copied trial folders since none passed
            if enable_grid_filter:
                print("\n🗑️  Cleaning up all copied trial models...")
                for trial_num in pareto_df['trial_number']:
                    target_dir = os.path.join(analyzer.output_dir, f"trial_{trial_num}")
                    if os.path.exists(target_dir):
                        shutil.rmtree(target_dir)
                        print(f"  ✓ Removed trial_{trial_num}")
            return None
        output_csv = f"{analyzer.output_dir}/pareto_frontier.csv"
        selected_df.to_csv(output_csv, index=False)
        print(f"\n✓ Pareto frontier results saved: {output_csv}")

        # Create visualizations
        if create_viz:
            visualizer = ParetoVisualizer(analyzer)
            visualizer.create_all_visualizations(df, pareto_df, selected_df)

        # Copy model folders (if grid filtering was disabled, or update the copied models)
        if not enable_grid_filter:
            copy_model_folders(selected_df, mode, model_type, analyzer.output_dir)
        else:
            # Remove models that were excluded by grid filtering
            print("\n🗑️  Cleaning up excluded models...")
            excluded_trials = set(pareto_df['trial_number']) - set(selected_df['trial_number'])
            for trial_num in excluded_trials:
                target_dir = os.path.join(analyzer.output_dir, f"trial_{trial_num}")
                if os.path.exists(target_dir):
                    shutil.rmtree(target_dir)
                    print(f"  ✓ Removed trial_{trial_num}")
            print(f"✓ Cleaned up {len(excluded_trials)} excluded models")

        # Print summary
        print_summary(selected_df, analyzer)

        print("✅ All tasks completed!\n")
        return selected_df # Return the selected Pareto solutions

    except Exception as e:
        print(f"\n❌ An error occurred during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None # Indicate failure

# --- END OF calculate_pareto_solution.py CONTENT ---

# Main execution for standalone use
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate Pareto frontier solutions with optional grid search filtering.')
    parser.add_argument('--mode', type=str, required=True, choices=['tradition', 'LLM'])
    parser.add_argument('--model_type', type=str, required=True, choices=['attention', 'series', 'uncertainty_1', 'uncertainty_2'])
    parser.add_argument('--secondary_direction', type=str, default='max', choices=['max', 'min'])
    parser.add_argument('--max_sol', type=int, default=20)
    parser.add_argument('--no_viz', action='store_true')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--enable_grid_filter', action='store_true', default=True)
    parser.add_argument('--disable_grid_filter', action='store_true')
    parser.add_argument('--eta', type=float, default=0.1)
    parser.add_argument('--grid_points_per_dim', type=int, default=5)
    parser.add_argument('--distance_threshold', type=float, default=0.5)
    parser.add_argument('--exceed_ratio_threshold', type=float, default=0.05,
                        help='Fraction of distant grid points exceeding eta threshold to exclude a solution (default: 0.3)')
    
    args = parser.parse_args()
    
    print(f"\n--- Step 1: Combining Metrics for {args.mode}/{args.model_type} ---")
    if not process_trial_data(args.mode, args.model_type):
        sys.exit(1)
    
    print(f"\n--- Step 2: Calculating Pareto Frontier ---")
    enable_grid_filter = args.enable_grid_filter and not args.disable_grid_filter

    selected_pareto_df = analyze_pareto_and_copy_models(
        mode=args.mode, model_type=args.model_type,
        secondary_direction=args.secondary_direction,
        max_sol=args.max_sol,
        create_viz=not args.no_viz, enable_grid_filter=enable_grid_filter,
        eta=args.eta,
        grid_points_per_dim=args.grid_points_per_dim,
        distance_threshold=args.distance_threshold,
        exceed_ratio_threshold=args.exceed_ratio_threshold
    )
    
    if selected_pareto_df is None:
        sys.exit(1)
    
    print("\n✅ Pareto frontier calculation completed!")
