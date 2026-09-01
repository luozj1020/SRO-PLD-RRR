# optim_prediction_utils.py - 精简整合版
import torch
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import itertools
import os
import warnings
from scipy.spatial import cKDTree
from typing import Tuple, Dict, List, Union, Optional, Any
from pathlib import Path

# BoTorch imports
from botorch.models import SingleTaskGP
from botorch.acquisition import LogExpectedImprovement, UpperConfidenceBound, ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

# Configuration
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_seed(seed: int) -> None:
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


setup_seed(42)


class ParameterProcessor:
    """统一处理参数转换和DataFrame创建"""

    PARAM_NAMES = ['log_oxygen_pressure', 'laser_energy_density', 'temperature', 'frequency', 'thickness']
    STANDARD_COLUMNS = ['Oxygen pressure', 'Laser energy density', 'Temperature', 'Frequency', 'Thickness']

    @staticmethod
    def extract_parameters(params: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
        """统一参数提取和转换"""
        # 转换为numpy数组
        if isinstance(params, torch.Tensor):
            params = params.detach().cpu().numpy()

        # 展平并验证维度
        if params.ndim > 1:
            params = params.flatten()
        if len(params) != 5:
            raise ValueError(f"Expected 5 parameters, got {len(params)}")

        # 参数处理和舍入
        log_oxygen = float(params[0])
        laser_energy = round(float(params[1]) * 100) / 100.0
        temperature = round(float(params[2]) * 10) / 10.0
        frequency = round(float(params[3]))
        thickness = float(params[4])

        return {
            'log_oxygen_pressure': log_oxygen,
            'linear_oxygen_pressure': float(10 ** log_oxygen),
            'laser_energy_density': laser_energy,
            'temperature': temperature,
            'frequency': frequency,
            'thickness': thickness
        }

    @staticmethod
    def create_dataframe(params: Dict[str, float]) -> pd.DataFrame:
        """创建标准化的DataFrame"""
        return pd.DataFrame({
            'Oxygen pressure': [params['linear_oxygen_pressure']],
            'Laser energy density': [params['laser_energy_density']],
            'Temperature': [params['temperature']],
            'Frequency': [params['frequency']],
            'Thickness': [params['thickness']]
        })

    @staticmethod
    def params_to_evaluation_array(param_dict: Dict[str, float],
                                   mean_val: float, std_val: float) -> List[float]:
        """将参数和结果转换为评估数组"""
        return [
            param_dict['log_oxygen_pressure'],
            param_dict['linear_oxygen_pressure'],
            param_dict['laser_energy_density'],
            param_dict['temperature'],
            param_dict['frequency'],
            param_dict['thickness'],
            mean_val,
            std_val
        ]


class CachedObjectiveFunction:
    """带缓存的目标函数基类"""

    def __init__(self, hybrid_model, processor, max_cache_size: int = 1000):
        self.hybrid_model = hybrid_model
        self.processor = processor
        self.param_processor = ParameterProcessor()
        self.cache = {}
        self.max_cache_size = max_cache_size
        self.evaluation_count = 0

    def _get_cache_key(self, params: Union[torch.Tensor, np.ndarray]) -> tuple:
        """生成缓存键"""
        if isinstance(params, torch.Tensor):
            params = params.detach().cpu().numpy()
        if params.ndim > 1:
            params = params.flatten()
        return tuple(np.round(params, 6))

    def _manage_cache(self):
        """管理缓存大小"""
        if len(self.cache) > self.max_cache_size:
            # 移除最早的一半缓存项
            keys = list(self.cache.keys())[:self.max_cache_size // 2]
            for key in keys:
                del self.cache[key]

    def __call__(self, params: Union[torch.Tensor, np.ndarray],
                 return_std: bool = True) -> Union[Tuple[float, float], float]:
        """目标函数调用"""
        cache_key = self._get_cache_key(params)

        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            # 参数提取和转换
            param_dict = self.param_processor.extract_parameters(params)
            test_data = self.param_processor.create_dataframe(param_dict)

            # 模型预测
            pred_mean, pred_std = self.hybrid_model.predict(test_data, n_samples=50)

            # 结果提取 - 确保张量在CPU上
            if hasattr(pred_mean, 'cpu'):
                pred_mean = pred_mean.cpu()
            if hasattr(pred_std, 'cpu'):
                pred_std = pred_std.cpu()
            
            mean_val = float(pred_mean.item() if hasattr(pred_mean, 'item') else pred_mean[0])
            std_val = float(pred_std.item() if hasattr(pred_std, 'item') else pred_std[0])

            result = (mean_val, std_val) if return_std else mean_val
            self.cache[cache_key] = result
            self._manage_cache()
            self.evaluation_count += 1

            return result

        except Exception as e:
            print(f"Error in objective function: {e}")
            import traceback
            traceback.print_exc()
            return (0.0, 1.0) if return_std else 0.0


class BayesianOptimizerCore:
    """贝叶斯优化核心功能"""

    def __init__(self, param_bounds: Dict[str, List[float]],
                 n_initial: int, n_sobol: int, seed: int = 42):
        self.param_bounds = param_bounds
        self.bounds = torch.tensor(
            [list(param_bounds[k]) for k in param_bounds],
            dtype=torch.float64
        ).T.to(device)
        self.n_initial = n_initial
        self.n_sobol = n_sobol
        self.history = {
            'best_values': [],
            'all_values': [],
            'std_values': [],
            'parameters': []
        }
        setup_seed(seed)

    def _generate_initial_points(self) -> torch.Tensor:
        """生成多样化的初始点（加2%边界缓冲，防止最优解卡在硬边界上）"""
        # 留出2%边界缓冲，与optimize_acquisition和generate_diverse_candidates保持一致
        margin = 0.02
        dim = len(self.param_bounds)
        buffer_bounds = torch.stack([
            torch.full((dim,), margin, dtype=torch.float64, device=device),
            torch.full((dim,), 1.0 - margin, dtype=torch.float64, device=device)
        ])
        
        # Sobol采样，低差异序列已能均匀覆盖参数空间
        sobol_points = draw_sobol_samples(
            bounds=buffer_bounds, n=1, q=self.n_sobol, seed=42
        ).squeeze(0)

        # 补充随机点至 n_initial
        n_random = max(0, self.n_initial - len(sobol_points))
        if n_random > 0:
            # 在归一化空间中生成 [margin, 1-margin] 范围内的随机点
            random_points = torch.rand(n_random, dim, dtype=torch.float64, device=device)
            random_points = random_points * (1.0 - 2 * margin) + margin
            all_points = torch.cat([sobol_points, random_points], dim=0)
        else:
            all_points = sobol_points

        return torch.unique(all_points, dim=0)

    def optimize_acquisition(self, gp, best_f: torch.Tensor,
                             selector, verbose: bool = False) -> torch.Tensor:
        """优化采集函数（加2%边界缓冲，防止最优解卡在硬边界上）"""
        acq_func = selector.get_acquisition_function(gp, best_f)

        # 归一化空间中留出2%缓冲，避免采集函数梯度将候选点推到硬边界
        margin = 0.02
        inner_bounds = torch.stack([
            torch.full((len(self.param_bounds),), margin, dtype=torch.float64, device=device),
            torch.full((len(self.param_bounds),), 1.0 - margin, dtype=torch.float64, device=device)
        ])

        candidates, _ = optimize_acqf(
            acq_func,
            bounds=inner_bounds,
            q=1,
            num_restarts=15,
            raw_samples=1024,
            options={"batch_limit": 5, "maxiter": 200}
        )

        # 检查重复点
        if selector.is_duplicate(candidates):
            if verbose:
                print("Duplicate detected, generating alternatives...")
            alt_candidates = selector.generate_diverse_candidates(gp)
            if len(alt_candidates) > 0:
                with torch.no_grad():
                    acq_values = acq_func(alt_candidates.unsqueeze(1))
                best_alt_idx = torch.argmax(acq_values)
                new_x = alt_candidates[best_alt_idx].unsqueeze(0)
            else:
                new_x = candidates
        else:
            new_x = candidates

        return new_x


class AdvancedAcquisitionSelector:
    """高级采集函数选择器"""

    def __init__(self, bounds: torch.Tensor, initial_points: torch.Tensor,
                 initial_y: torch.Tensor, tolerance: float = 1e-4):
        self.bounds = bounds
        self.tolerance = tolerance
        self.visited_points = initial_points.detach().cpu().numpy().astype(np.float64)
        self.train_y = initial_y.detach().cpu().numpy().flatten().astype(np.float64)
        self.iteration = 0
        self._build_kdtree()

    def _build_kdtree(self) -> None:
        """构建KDTree用于最近邻搜索"""
        if len(self.visited_points) > 0:
            self.kdtree = cKDTree(self.visited_points)
        else:
            self.kdtree = None

    def update(self, new_points: torch.Tensor, new_y: torch.Tensor) -> None:
        """更新访问点"""
        new_points_np = new_points.detach().cpu().numpy().astype(np.float64)
        new_y_np = new_y.detach().cpu().numpy().flatten().astype(np.float64)

        self.visited_points = np.vstack([self.visited_points, new_points_np])
        self.train_y = np.concatenate([self.train_y, new_y_np])
        self.iteration += 1
        self._build_kdtree()

    def is_duplicate(self, candidate: torch.Tensor,
                     tolerance: Optional[float] = None) -> bool:
        """检查候选点是否与现有点过于接近"""
        if self.kdtree is None or len(self.visited_points) == 0:
            return False

        tol = tolerance or self.tolerance
        candidate_np = candidate.detach().cpu().numpy().astype(np.float64).reshape(1, -1)
        dist, _ = self.kdtree.query(candidate_np, k=1)
        return dist < tol

    def get_acquisition_function(self, gp, best_f: torch.Tensor):
        """基于优化阶段选择采集函数"""
        if self.iteration < 50:
            return ExpectedImprovement(gp, best_f=best_f)
        elif 50 <= self.iteration < 100:
            return UpperConfidenceBound(gp, beta=2.0)
        else:
            return LogExpectedImprovement(gp, best_f=best_f)

    def generate_diverse_candidates(self, gp, n_candidates: int = 10) -> torch.Tensor:
        """生成多样化的候选点"""
        # 加入2%边界缓冲，与optimize_acquisition保持一致
        margin = 0.02
        dim = len(self.bounds[0])
        inner_lower = self.bounds[0].cpu().numpy() + margin * (
            self.bounds[1].cpu().numpy() - self.bounds[0].cpu().numpy()
        )
        inner_upper = self.bounds[1].cpu().numpy() - margin * (
            self.bounds[1].cpu().numpy() - self.bounds[0].cpu().numpy()
        )
        
        candidates = []

        # 策略1：在最佳点周围开发
        if len(self.train_y) > 0:
            top_k = min(3, len(self.train_y))
            top_indices = np.argsort(self.train_y)[-top_k:]

            for idx in top_indices:
                best_point = self.visited_points[idx]
                for _ in range(2):
                    perturbation = np.random.normal(0, 0.1, size=best_point.shape)
                    candidate = np.clip(
                        best_point + perturbation,
                        inner_lower,
                        inner_upper
                    )
                    candidates.append(candidate)

        # 策略2：在稀疏区域探索
        if self.kdtree is not None:
            sparse_candidates = np.random.uniform(
                inner_lower,
                inner_upper,
                size=(n_candidates * 5, dim)
            )

            distances, _ = self.kdtree.query(sparse_candidates, k=1)
            sparse_indices = np.argsort(distances)[-n_candidates // 2:]
            candidates.extend(sparse_candidates[sparse_indices])

        # 策略3：拉丁超立方采样
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=dim, seed=self.iteration)
            lhs_samples = sampler.random(n_candidates // 2)

            lhs_scaled = inner_lower + (inner_upper - inner_lower) * lhs_samples
            candidates.extend(lhs_scaled)
        except ImportError:
            # 备用方案：如果scipy.stats.qmc不可用
            random_samples = np.random.uniform(
                inner_lower,
                inner_upper,
                size=(n_candidates // 2, dim)
            )
            candidates.extend(random_samples)

        # 确保至少有一个候选点
        if len(candidates) == 0:
            candidates = [np.random.uniform(
                inner_lower,
                inner_upper
            )]

        # 去重
        candidates = np.array(candidates, dtype=np.float64)
        unique_candidates = []

        for candidate in candidates:
            if not any(np.linalg.norm(candidate - uc) < self.tolerance
                       for uc in unique_candidates):
                if not self.is_duplicate(torch.tensor(candidate, dtype=torch.float64)):
                    unique_candidates.append(candidate)

        if len(unique_candidates) == 0:
            unique_candidates = [np.random.uniform(
                inner_lower,
                inner_upper
            )]

        return torch.tensor(unique_candidates[:n_candidates],
                            dtype=torch.float64, device=device)


class BayesianOptimizer(BayesianOptimizerCore):
    """整合的贝叶斯优化器"""

    def __init__(self, objective_fn: CachedObjectiveFunction,
                 param_bounds: Dict[str, List[float]],
                 n_initial: int, n_sobol: int, seed: int = 42):
        super().__init__(param_bounds, n_initial, n_sobol, seed)
        self.objective_fn = objective_fn
        self.param_processor = ParameterProcessor()

        self.selector = None
        self.train_x = None
        self.train_y = None
        self.train_std = None
        self.all_evaluations = []

    def initialize(self, grid_samples=None) -> None:
        """初始化优化器

        Args:
            grid_samples: 可选的网格搜索采样结果列表
                         格式: [{'params': np.array, 'mean': float, 'std': float}, ...]
        """
        print("Initializing with diverse sampling...")

        # _generate_initial_points 已返回归一化空间 [0.02, 0.98] 的点，直接使用
        normalized_points = self._generate_initial_points()

        train_mean, train_std = [], []
        for point in tqdm(normalized_points, desc="Initial sampling"):
            # 将归一化空间的点转换为原始空间
            original_point = unnormalize(point.unsqueeze(0), self.bounds).squeeze(0)
            mean_val, std_val = self.objective_fn(original_point)
            train_mean.append(float(mean_val))
            train_std.append(float(std_val))
            self._store_evaluation(original_point, mean_val, std_val)

        self.train_x = normalized_points
        self.train_y = torch.tensor(train_mean, dtype=torch.float64, device=device).unsqueeze(-1)
        self.train_std = torch.tensor(train_std, dtype=torch.float64, device=device)

        # 添加网格搜索采样结果
        if grid_samples is not None and len(grid_samples) > 0:
            print(f"\n📊 添加 {len(grid_samples)} 个网格搜索采样点到初始化数据...")

            grid_points = []
            grid_means = []
            grid_stds = []

            for sample in grid_samples:
                params = sample['params']
                # 转换为tensor并归一化
                params_tensor = torch.tensor(params, dtype=torch.float64, device=device)
                normalized_params = normalize(params_tensor.unsqueeze(0), self.bounds).squeeze(0)

                grid_points.append(normalized_params)
                grid_means.append(float(sample['mean']))
                grid_stds.append(float(sample['std']))

                # 存储评估结果
                self._store_evaluation(params_tensor, sample['mean'], sample['std'])

            # 合并到训练数据
            grid_points_tensor = torch.stack(grid_points)
            grid_means_tensor = torch.tensor(grid_means, dtype=torch.float64, device=device).unsqueeze(-1)
            grid_stds_tensor = torch.tensor(grid_stds, dtype=torch.float64, device=device)

            self.train_x = torch.cat([self.train_x, grid_points_tensor], dim=0)
            self.train_y = torch.cat([self.train_y, grid_means_tensor], dim=0)
            self.train_std = torch.cat([self.train_std, grid_stds_tensor], dim=0)

            print(f"  ✓ 总共 {len(self.train_x)} 个初始采样点（包含网格搜索采样）")

        self.selector = AdvancedAcquisitionSelector(
            bounds=torch.stack([
                torch.zeros(len(self.param_bounds), dtype=torch.float64, device=device),
                torch.ones(len(self.param_bounds), dtype=torch.float64, device=device)
            ]),
            initial_points=self.train_x,
            initial_y=self.train_y,
            tolerance=1e-3
        )

        print(f"Initialized with {len(self.train_x)} points")
        print(f"Initial best value: {self.train_y.max().item():.4f}")

    def _store_evaluation(self, point: torch.Tensor, mean_val: float, std_val: float) -> None:
        """存储评估结果"""
        param_dict = self.param_processor.extract_parameters(point)
        eval_array = self.param_processor.params_to_evaluation_array(param_dict, mean_val, std_val)
        self.all_evaluations.append(eval_array)

    def optimize(self, n_iterations: int = 100, verbose: bool = True, grid_samples=None) -> Dict:
        """运行贝叶斯优化

        Args:
            n_iterations: 优化迭代次数
            verbose: 是否显示详细信息
            grid_samples: 可选的网格搜索采样结果
        """
        if self.train_x is None:
            self.initialize(grid_samples=grid_samples)

        print(f"Starting optimization for {n_iterations} iterations...")

        for i in tqdm(range(n_iterations), desc="Optimization"):
            try:
                # 拟合高斯过程
                gp = SingleTaskGP(self.train_x, self.train_y)
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_mll(mll)

                best_f = self.train_y.max()
                new_x = self.optimize_acquisition(gp, best_f, self.selector, verbose)

                # 评估新点
                new_params = unnormalize(new_x, self.bounds)
                new_y, new_std = self.objective_fn(new_params)

                # 存储结果
                self._store_evaluation(new_params.squeeze(), float(new_y), float(new_std))

                # 更新训练数据
                self.train_x = torch.cat([self.train_x, new_x])
                self.train_y = torch.cat([
                    self.train_y,
                    torch.tensor([[new_y]], dtype=torch.float64, device=device)
                ])
                self.train_std = torch.cat([
                    self.train_std,
                    torch.tensor([new_std], dtype=torch.float64, device=device)
                ])

                self.selector.update(new_x, torch.tensor([new_y], dtype=torch.float64, device=device))

                # 更新历史记录
                self.history['best_values'].append(float(self.train_y.max().item()))
                self.history['all_values'].append(new_y)
                self.history['std_values'].append(new_std)
                self.history['parameters'].append(new_params.detach().cpu().numpy())

                # 进度日志
                if verbose and i % 10 == 0:
                    print(f"Iteration {i}: New value = {new_y:.4f} ± {new_std:.4f}")
                    print(f"Best so far: {self.train_y.max().item():.4f}")

            except Exception as e:
                print(f"Error in iteration {i}: {e}")
                continue

        print("Optimization completed!")
        return self.get_best_result()

    def get_all_evaluations(self) -> np.ndarray:
        """获取所有评估数据"""
        return np.array(self.all_evaluations, dtype=np.float64)

    def get_best_result(self) -> Dict:
        """获取最佳结果"""
        best_idx = self.train_y.argmax()
        best_params = unnormalize(self.train_x[best_idx], self.bounds)
        best_value = float(self.train_y[best_idx].item())
        best_std = float(self.train_std[best_idx].item())

        return {
            'best_value': best_value,
            'best_std': best_std,
            'best_params': best_params.detach().cpu().numpy(),
            'best_idx': int(best_idx.item())
        }

    def get_top_results(self, k: int = 10) -> List[Dict]:
        """获取前k个结果"""
        topk_values, topk_indices = torch.topk(self.train_y.flatten(), k=min(k, len(self.train_y)))

        results = []
        for i, (idx, val) in enumerate(zip(topk_indices, topk_values)):
            params = unnormalize(self.train_x[idx], self.bounds).cpu().numpy()
            std_val = self.train_std[idx].item()

            results.append({
                'rank': i + 1,
                'value': val.item(),
                'std': std_val,
                'parameters': params
            })

        return results


class VisualizationUtils:
    """可视化工具类"""

    @staticmethod
    def create_parameter_plots(df: pd.DataFrame, save_dir: Union[str, Path]) -> None:
        """创建参数分布和关系图"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 参数效应图
        fig1, axes1 = plt.subplots(2, 2, figsize=(15, 10))

        param_plots = [
            ('oxygen_pressure', 'log', 'Oxygen Pressure (Pa)'),
            ('laser_energy_density', 'linear', 'Laser Energy Density (J/cm$^2$)'),
            ('temperature', 'linear', 'Temperature (°C)'),
            ('frequency', 'linear', 'Frequency (Hz)')
        ]

        for idx, (param, scale, xlabel) in enumerate(param_plots):
            ax = axes1.flatten()[idx]
            sns.scatterplot(x=param, y='mean', data=df, alpha=0.6, ax=ax)
            if scale == 'log':
                ax.set_xscale('log')
            ax.set_title(f'{param.replace("_", " ").title()} vs Mean')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Mean Value')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / 'parameter_effects.png', dpi=300)
        plt.close()

        # 参数分布图
        fig2, axes2 = plt.subplots(3, 2, figsize=(15, 12))
        params = ['oxygen_pressure', 'laser_energy_density', 'temperature', 'frequency', 'thickness']

        for i, param in enumerate(params):
            ax = axes2.flatten()[i]
            if param == 'oxygen_pressure':
                ax.hist(np.log10(df[param]), bins=20, alpha=0.7)
                ax.set_xlabel('log10(Oxygen Pressure)')
            else:
                ax.hist(df[param], bins=20, alpha=0.7)
                ax.set_xlabel(param)
            ax.set_title(f'{param} Distribution')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / 'parameter_distributions.png', dpi=300)
        plt.close()

        # 配对图
        plt.figure(figsize=(15, 10))
        sns.pairplot(df[['oxygen_pressure', 'laser_energy_density', 'temperature', 'frequency', 'mean']],
                     diag_kind='kde', corner=True)
        plt.savefig(save_dir / 'parameter_pairplot.png', dpi=300)
        plt.close()

    @staticmethod
    def plot_optimization_progress(optimizer: BayesianOptimizer,
                                   save_dir: Union[str, Path],
                                   figsize: Tuple[int, int] = (15, 10)) -> None:
        """绘制优化进度图"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 主进度图
        fig_main, (ax_main, ax_var) = plt.subplots(
            nrows=2,
            figsize=(12, 8),
            gridspec_kw={'height_ratios': [3, 1]},
            sharex=True
        )

        y_values = optimizer.train_y.detach().cpu().numpy().flatten()
        std_values = optimizer.train_std.detach().cpu().numpy().flatten()
        x_indices = np.arange(len(y_values))
        cumulative_max = np.maximum.accumulate(y_values)

        ax_main.plot(x_indices, cumulative_max,
                     label='Best Value Curve',
                     color='darkorange',
                     linewidth=2.5,
                     linestyle='--')

        ax_main.scatter(x_indices, y_values,
                        c='red',
                        s=40,
                        alpha=0.6,
                        label='All Samples',
                        zorder=3)

        optimal_idx = np.argmax(y_values)
        ax_main.scatter(optimal_idx, y_values[optimal_idx],
                        s=200, marker='*',
                        c='gold',
                        edgecolor='black',
                        label=f'Global Optimal ({y_values[optimal_idx]:.2f})',
                        zorder=4)

        ax_main.set_title('Optimization Progress with Variance Tracking', fontsize=14, pad=12)
        ax_main.set_ylabel('Target Value', fontsize=12)
        ax_main.grid(True, alpha=0.3)
        ax_main.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

        # 方差图
        ax_var.fill_between(x_indices, std_values,
                            color='royalblue',
                            alpha=0.2,
                            label='Variance')
        ax_var.plot(x_indices, std_values,
                    color='navy',
                    linewidth=1.2,
                    alpha=0.8)

        ax_var.set_xlabel('Sample Index', fontsize=12)
        ax_var.set_ylabel('Variance', fontsize=12)
        ax_var.grid(True, alpha=0.3)
        ax_var.set_ylim(bottom=0)

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.05)
        plt.savefig(save_dir / 'progress_plot.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 参数空间图
        if len(optimizer.history['parameters']) > 0:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            params = np.array(optimizer.history['parameters']).squeeze()
            history_values = np.array(optimizer.history['all_values'])

            if params.ndim == 2 and params.shape[1] >= 3:
                history_optimal_idx = np.argmax(history_values)

                # 绘制参数关系
                param_pairs = [
                    (0, 1, 'Oxygen Pressure (log scale)', 'Laser Energy Density'),
                    (0, 2, 'Oxygen Pressure (log scale)', 'Temperature (°C)'),
                    (1, 2, 'Laser Energy Density', 'Temperature (°C)')
                ]

                for idx, (x_idx, y_idx, xlabel, ylabel) in enumerate(param_pairs):
                    sc = axes[idx].scatter(params[:, x_idx], params[:, y_idx],
                                           c=history_values,
                                           cmap='viridis', alpha=0.7)
                    plt.colorbar(sc, ax=axes[idx], label='Objective Value')
                    axes[idx].set_xlabel(xlabel)
                    axes[idx].set_ylabel(ylabel)
                    axes[idx].set_title(f'{xlabel.split(" ")[0]} vs {ylabel.split(" ")[0]}')
                    axes[idx].grid(True, alpha=0.3)

                    # 标记最佳点
                    axes[idx].scatter(params[history_optimal_idx, x_idx],
                                      params[history_optimal_idx, y_idx],
                                      s=150, marker='*', c='red', edgecolor='black',
                                      label='Best in History')
                    axes[idx].legend()

                plt.tight_layout(rect=[0, 0, 1, 0.93])
                fig.suptitle('Parameter Space Exploration', fontsize=16, y=0.98)
                plt.savefig(save_dir / 'param_space.png', dpi=300, bbox_inches='tight')
                plt.close()


def create_and_save_all_visualizations(optimizer: BayesianOptimizer,
                                       all_evaluations: List,
                                       results_dir: Union[str, Path],
                                       optim_mode: str,
                                       trial_id: str,
                                       fixed_frequency: Optional[float] = None,
                                       thickness_mean: Optional[float] = None,
                                       thickness_std: Optional[float] = None) -> None:
    """
    创建并保存所有可视化图表

    参数:
        optimizer: BayesianOptimizer实例
        all_evaluations: 所有评估记录
        results_dir: 结果保存目录
        optim_mode: 优化模式 ('optim_all', 'fix_freq', 'fix_thick')
        trial_id: 试验ID
        fixed_frequency: 固定频率值（仅当optim_mode='fix_freq'时）
        thickness_mean: 厚度均值（仅当optim_mode='fix_thick'时）
        thickness_std: 厚度标准差（仅当optim_mode='fix_thick'时）
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not all_evaluations:
        print(f"警告: Trial {trial_id} 无有效评估数据，跳过图表生成。")
        return

    # 创建DataFrame
    columns = ['log_oxygen_pressure', 'oxygen_pressure', 'laser_energy_density',
               'temperature', 'frequency', 'thickness', 'mean', 'std']
    df = pd.DataFrame(all_evaluations, columns=columns)

    # 数据清洗
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_clean = df.dropna()
    df_clean = df_clean[~np.isinf(df_clean[numeric_cols]).any(axis=1)]

    if len(df_clean) == 0:
        print(f"警告: Trial {trial_id} 清洗后无有效数据，跳过图表生成。")
        return

    df = df_clean

    # 使用VisualizationUtils创建图表
    viz = VisualizationUtils()
    viz.create_parameter_plots(df, results_dir)
    viz.plot_optimization_progress(optimizer, results_dir)

    # 厚度不确定性信息（仅限fix_thick模式）
    if optim_mode == 'fix_thick' and thickness_mean is not None and thickness_std is not None:
        info_path = results_dir / "thickness_uncertainty_info.txt"
        with open(info_path, 'w') as f:
            f.write(f"Thickness Uncertainty Information for Trial {trial_id}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Optimization Mode: 'fix_thick' (Thickness marginalized)\n")
            f.write(f"Assumed Thickness Distribution: N(mean={thickness_mean}, std={thickness_std})\n\n")
            f.write("Statistics of Predicted Mean across Thickness Samples:\n")
            f.write("-" * 50 + "\n")
            if 'mean' in df.columns:
                mean_vals = df['mean'].dropna()
                f.write(f"Number of Evaluations: {len(mean_vals)}\n")
                f.write(f"Mean of Predicted Means: {mean_vals.mean():.6f}\n")
                f.write(f"Std of Predicted Means: {mean_vals.std():.6f}\n")
                f.write(f"Min Predicted Mean: {mean_vals.min():.6f}\n")
                f.write(f"Max Predicted Mean: {mean_vals.max():.6f}\n")
                f.write(f"Median Predicted Mean: {np.median(mean_vals.values):.6f}\n\n")

        print(f"Created thickness uncertainty info: {info_path}")

    print(f"✅ Trial {trial_id} 的可视化文件生成完成。")