#!/usr/bin/env python3
"""
Bayesian Optimization Module for Pareto Solutions

This module handles:
1. Loading Pareto solution models
2. Running Bayesian optimization on each model
3. Supporting multiple optimization modes (optim_all, fix_freq, fix_thick)
4. GPU acceleration support
5. Visualization of optimization results
"""

import os
import pandas as pd
import numpy as np
import torch
import warnings
import sys
from pathlib import Path
import argparse
import gc
from collections import OrderedDict
import matplotlib.pyplot as plt
import random

from utils.optim_prediction_utils import (
    CachedObjectiveFunction,
    BayesianOptimizer,
    create_and_save_all_visualizations,
    ParameterProcessor
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

warnings.filterwarnings('ignore')

# --- START OF optim_prediction.py CONTENT ---
# Configuration
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def setup_seed(seed):
    """Set random seeds for reproducibility across libraries"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


setup_seed(42)


class LRUCache:
    """Finite size LRU cache to prevent unbounded memory growth"""

    def __init__(self, maxsize=1000):
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)


class UncertainThicknessObjectiveFunction(CachedObjectiveFunction):
    """
    Optimized objective function, addressing GPU memory leaks and hanging issues.
    """

    def __init__(self, hybrid_model, processor, thickness_mean, thickness_std,
                 n_samples, thickness_bounds=(0.5, 200), cache_size=1000):
        super().__init__(hybrid_model, processor, cache_size)
        self.thickness_mean = thickness_mean
        self.thickness_std = thickness_std
        self.n_samples = n_samples
        self.thickness_bounds = thickness_bounds
        self._thickness_samples = None
        self.all_evaluations = []
        self.evaluation_count = 0

    def _get_thickness_samples(self):
        if self._thickness_samples is None or len(self._thickness_samples) != self.n_samples:
            samples = np.random.normal(self.thickness_mean, self.thickness_std, self.n_samples)
            self._thickness_samples = np.clip(samples, self.thickness_bounds[0], self.thickness_bounds[1])
        return self._thickness_samples

    def __call__(self, params, return_std=True):
        # 简化后的评估逻辑
        # 处理参数 - 确保转换为numpy数组
        if isinstance(params, torch.Tensor):
            params = params.detach().cpu().numpy()
        else:
            params = np.array(params)
        if params.ndim > 1:
            params = params.flatten()
        
        cache_key = self._get_cache_key(params)
        if cache_key in self.cache:
            return self.cache[cache_key]

        thickness_samples = self._get_thickness_samples()
        means, stds = [], []

        for thickness_val in thickness_samples:
            full_params = np.append(params, thickness_val)
            mean_val, std_val = super().__call__(full_params, return_std=True)
            means.append(mean_val)
            stds.append(std_val)

        avg_mean = np.mean(means)
        model_uncertainty = np.mean(stds)
        thickness_variance = np.var(means)
        total_std = np.sqrt(model_uncertainty ** 2 + thickness_variance)

        result = (float(avg_mean), float(total_std)) if return_std else float(avg_mean)
        self.cache[cache_key] = result
        
        # Store evaluation result
        self._store_evaluation_4params(params, avg_mean, total_std)
        self.evaluation_count += 1
        
        # Periodic cleanup
        if self.evaluation_count % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return result

    def _store_evaluation_4params(self, params_4d, mean_val, std_val):
        """Store evaluation with 4 parameters"""
        params_5d = np.append(params_4d, self.thickness_mean)

        eval_dict = {
            'log_oxygen_pressure': params_5d[0],  # New first column
            'oxygen_pressure': 10 ** params_5d[0],  # Original oxygen pressure column
            'laser_energy_density': params_5d[1],
            'temperature': params_5d[2],
            'frequency': params_5d[3],
            'thickness': params_5d[4],
            'mean': mean_val,
            'std': std_val
        }
        self.all_evaluations.append(eval_dict)

    def save_results(self, all_evaluations, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # all_evaluations 是 dict 列表，直接构造 DataFrame
        df = pd.DataFrame(all_evaluations)

        csv_path = save_dir / "optimization_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")


class AllVariablesObjectiveFunction:
    """
    Mode 1: Optimize all 5 variables (oxygen_pressure, laser_energy_density, temperature, frequency, thickness)
    """

    def __init__(self, base_objective_fn, cache_size=1000):
        self.base_fn = base_objective_fn
        self.cache = LRUCache(maxsize=cache_size)
        self.evaluation_count = 0
        self.param_processor = base_objective_fn.param_processor
        self.all_evaluations = []

    def __call__(self, params: torch.Tensor, return_std: bool = True):
        """Evaluation function - all 5 parameters"""
        # Parameter handling
        if isinstance(params, torch.Tensor):
            params_np = params.detach().cpu().numpy()
        else:
            params_np = np.array(params)

        if params_np.ndim > 1:
            params_np = params_np.flatten()

        if len(params_np) != 5:
            raise ValueError(f"Expected 5 parameters, got {len(params_np)}")

        # Check cache
        cache_key = tuple(np.round(params_np, 6))
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        # Directly evaluate 5 parameters
        try:
            mean_val, std_val = self.base_fn(params_np, return_std=True)
        except Exception as e:
            print(f"Warning: Evaluation failed: {e}")
            result = (0.0, 1.0) if return_std else 0.0
            self.cache.put(cache_key, result)
            return result

        self.evaluation_count += 1

        if self.evaluation_count % 50 == 0:
            print(f"Evaluation {self.evaluation_count}: Mean={mean_val:.4f}, Std={std_val:.4f}")

        result = (float(mean_val), float(std_val)) if return_std else float(mean_val)
        self.cache.put(cache_key, result)

        # Store evaluation result
        self._store_evaluation_5params(params_np, mean_val, std_val)

        # Periodic cleanup
        if self.evaluation_count % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return result

    def _store_evaluation_5params(self, params_5d, mean_val, std_val):
        """Store evaluation results for 5 parameters"""
        eval_dict = {
            'log_oxygen_pressure': params_5d[0],
            'oxygen_pressure': 10 ** params_5d[0],
            'laser_energy_density': params_5d[1],
            'temperature': params_5d[2],
            'frequency': params_5d[3],
            'thickness': params_5d[4],
            'mean': mean_val,
            'std': std_val
        }
        self.all_evaluations.append(eval_dict)

    def save_results(self, all_evaluations, save_dir):
        """Save results"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(all_evaluations)
        csv_path = save_dir / "optimization_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

        # Assuming the base function has this method or we define it here
        # For simplicity in this merged script, we'll skip the plots creation part
        # unless the full utils module is available.
        # self.create_parameter_plots(df, save_dir)


class FixedFrequencyObjectiveFunction:
    """
    Mode 2: Fix frequency, optimize other 4 variables (oxygen_pressure, laser_energy_density, temperature, thickness)
    """

    def __init__(self, base_objective_fn, fixed_frequency, cache_size=1000):
        self.base_fn = base_objective_fn
        self.fixed_frequency = fixed_frequency
        self.cache = LRUCache(maxsize=cache_size)
        self.evaluation_count = 0
        self.param_processor = base_objective_fn.param_processor
        self.all_evaluations = []

    def __call__(self, params: torch.Tensor, return_std: bool = True):
        """Evaluation function - 4 parameters + fixed frequency"""
        # Parameter handling
        if isinstance(params, torch.Tensor):
            params_np = params.detach().cpu().numpy()
        else:
            params_np = np.array(params)

        if params_np.ndim > 1:
            params_np = params_np.flatten()

        if len(params_np) != 4:
            raise ValueError(f"Expected 4 parameters, got {len(params_np)}")

        # Check cache
        cache_key = tuple(np.round(params_np, 6))
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        # Insert fixed frequency (at position 4, index 3)
        full_params = np.insert(params_np, 3, self.fixed_frequency)

        # Evaluate 5 parameters
        try:
            mean_val, std_val = self.base_fn(full_params, return_std=True)
        except Exception as e:
            print(f"Warning: Evaluation failed: {e}")
            result = (0.0, 1.0) if return_std else 0.0
            self.cache.put(cache_key, result)
            return result

        self.evaluation_count += 1

        if self.evaluation_count % 50 == 0:
            print(f"Evaluation {self.evaluation_count}: Mean={mean_val:.4f}, Std={std_val:.4f}")

        result = (float(mean_val), float(std_val)) if return_std else float(mean_val)
        self.cache.put(cache_key, result)

        # Store evaluation result
        self._store_evaluation_4params(params_np, mean_val, std_val)

        # Periodic cleanup
        if self.evaluation_count % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return result

    def _store_evaluation_4params(self, params_4d, mean_val, std_val):
        """Store evaluation results for 4 parameters + fixed frequency"""
        # params_4d: [oxygen_pressure, laser_energy_density, temperature, thickness]
        params_5d = np.insert(params_4d, 3, self.fixed_frequency)

        eval_dict = {
            'log_oxygen_pressure': params_5d[0],
            'oxygen_pressure': 10 ** params_5d[0],
            'laser_energy_density': params_5d[1],
            'temperature': params_5d[2],
            'frequency': params_5d[3],  # Fixed value
            'thickness': params_5d[4],
            'mean': mean_val,
            'std': std_val
        }
        self.all_evaluations.append(eval_dict)

    def save_results(self, all_evaluations, save_dir):
        """Save results"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(all_evaluations)
        csv_path = save_dir / "optimization_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

        # Assuming the base function has this method or we define it here
        # For simplicity in this merged script, we'll skip the plots creation part
        # unless the full utils module is available.
        # self.create_parameter_plots(df, save_dir)


class FixedFrequencyUncertainThicknessObjectiveFunction:
    """
    Mode: Fix frequency, optimize 3 variables (oxygen, laser, temp), thickness marginalized over N(mean, std)
    """
    def __init__(self, base_objective_fn, fixed_frequency, thickness_mean, thickness_std,
                 n_samples=20, thickness_bounds=(30, 200), cache_size=2000):
        self.base_fn = base_objective_fn
        self.fixed_frequency = fixed_frequency
        self.thickness_mean = thickness_mean
        self.thickness_std = thickness_std
        self.n_samples = n_samples
        self.thickness_bounds = thickness_bounds
        self.cache = LRUCache(maxsize=cache_size)
        self.evaluation_count = 0
        self.param_processor = base_objective_fn.param_processor
        self.all_evaluations = []
        self._thickness_samples = None

    def _get_thickness_samples(self):
        if self._thickness_samples is None or len(self._thickness_samples) != self.n_samples:
            samples = np.random.normal(self.thickness_mean, self.thickness_std, self.n_samples)
            self._thickness_samples = np.clip(samples, self.thickness_bounds[0], self.thickness_bounds[1])
        return self._thickness_samples

    def __call__(self, params, return_std=True):
        if isinstance(params, torch.Tensor):
            params_np = params.detach().cpu().numpy()
        else:
            params_np = np.array(params)
        if params_np.ndim > 1:
            params_np = params_np.flatten()
        if len(params_np) != 3:
            raise ValueError(f"Expected 3 parameters, got {len(params_np)}")

        cache_key = tuple(np.round(params_np, 6))
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        thickness_samples = self._get_thickness_samples()
        means, stds = [], []
        for thick_val in thickness_samples:
            # 拼接5维参数: [oxygen, laser, temp, freq, thick]
            full_params = np.array([params_np[0], params_np[1], params_np[2], self.fixed_frequency, thick_val])
            try:
                m, s = self.base_fn(full_params, return_std=True)
                means.append(m)
                stds.append(s)
            except Exception as e:
                print(f"Warning: Evaluation failed for thick={thick_val:.1f}: {e}")
                means.append(0.0)
                stds.append(1.0)

        avg_mean = np.mean(means)
        model_uncertainty = np.mean(stds)
        thickness_variance = np.var(means)
        total_std = np.sqrt(model_uncertainty ** 2 + thickness_variance)

        result = (float(avg_mean), float(total_std)) if return_std else float(avg_mean)
        self.cache.put(cache_key, result)
        self._store_evaluation_3params(params_np, avg_mean, total_std)
        self.evaluation_count += 1

        if self.evaluation_count % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return result

    def _store_evaluation_3params(self, params_3d, mean_val, std_val):
        full_params = np.append(params_3d, [self.fixed_frequency, self.thickness_mean])
        eval_dict = {
            'log_oxygen_pressure': full_params[0],
            'oxygen_pressure': 10 ** full_params[0],
            'laser_energy_density': full_params[1],
            'temperature': full_params[2],
            'frequency': full_params[3],
            'thickness': full_params[4],
            'mean': mean_val,
            'std': std_val
        }
        self.all_evaluations.append(eval_dict)

    def save_results(self, all_evaluations, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(all_evaluations)
        csv_path = save_dir / "optimization_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")


def run_optimization_for_trial(trial_id: str, mode: str, model_type: str, base_output_dir: str, HybridModel,
                               optim_mode: str, fixed_frequency: float = 5.0,
                               thickness_mean: float = 20.0,
                               thickness_std: float = 10.0,
                               n_thickness_samples: int = 20,
                               use_gpu: bool = True):
    """
    Optimized trial execution function, supporting three optimization modes.
    Args:
        trial_id: Trial identifier
        model_dir: Directory containing the model
        results_dir: Directory to save results
        HybridModel: HybridModel class
        optim_mode: Optimization mode ('optim_all', 'fix_freq', 'fix_thick')
        fixed_frequency: Fixed frequency value for fix_freq
        thickness_mean: Mean value for thickness distribution (for fix_thick)
        thickness_std: Standard deviation for thickness distribution (for fix_thick)
        n_thickness_samples: Number of samples per evaluation (for fix_thick)
        use_gpu: Whether to use GPU for optimization
    """
    # Set result directory name based on mode
    # 参数边界与训练数据范围对齐，避免模型外推导致最优解落在边界上
    # 参数边界与训练数据范围对齐，避免模型外推导致最优解落在边界上
    if optim_mode == 'optim_all':
        mode_suffix = 'all_5vars'
        param_bounds = {
            'oxygen_pressure': (-4, -0.301),
            'laser_energy_density': (1.5, 3.0),
            'temperature': (500, 800),
            'frequency': (4, 10),
            'thickness': (30, 200),
        }
    elif optim_mode == 'fix_freq':
        mode_suffix = f'fix_freq_{fixed_frequency}'
        param_bounds = {
            'oxygen_pressure': (-4, -0.301),
            'laser_energy_density': (1.5, 3.0),
            'temperature': (500, 800),
            'thickness': (30, 200),
        }
    elif optim_mode == 'fix_freq_thick':
        mode_suffix = f'fix_freq_{fixed_frequency}_thick_uncertain'
        param_bounds = {
            'oxygen_pressure': (-4, -0.301),
            'laser_energy_density': (1.5, 3.0),
            'temperature': (500, 800),
        }
    else:  # fix_thick
        mode_suffix = 'thickness_uncertain'
        param_bounds = {
            'oxygen_pressure': (-4, -0.301),
            'laser_energy_density': (1.5, 3.0),
            'temperature': (500, 800),
            'frequency': (4, 10),
        }

    # Load model
    base_path = f"./{mode}/XGB_BNN_{model_type}_hybrid_model"
    trial_model_dir = Path(base_path) / "pareto_solution" / trial_id / "model"

    # Check if already optimized
    results_dir = Path(base_output_dir) / f"model_optim_prediction_results_{trial_id}_{mode_suffix}"
    if results_dir.exists() and any(results_dir.iterdir()):
        print(f"\n{'=' * 60}\nTrial {trial_id} ({optim_mode}) already optimized - Skipping\n{'=' * 60}")
        return None, None

    if not trial_model_dir.exists():
        raise FileNotFoundError(f"Trial model directory does not exist: {trial_model_dir}")

    print(f"\n{'=' * 60}\nOptimizing trial model: {trial_id} (Mode: {optim_mode})\n{'=' * 60}")

    if optim_mode == 'optim_all':
        print(f"Optimizing all 5 variables")
    elif optim_mode == 'fix_freq':
        print(f"Fixed frequency: {fixed_frequency}")
    elif optim_mode == 'fix_freq_thick':
        print(f"Fixed frequency: {fixed_frequency}")
        print(f"Thickness uncertainty: N({thickness_mean:.2f}, {thickness_std:.2f}²) [marginalized]")
    else:  # fix_thick
        print(f"Thickness uncertainty: N({thickness_mean:.2f}, {thickness_std:.2f}²)")

    print(f"GPU acceleration: {use_gpu}")

    hybrid_model = HybridModel.load_model(str(trial_model_dir))
    print("Hybrid model loaded successfully!")

    processor = hybrid_model.processor
    if processor is None:
        raise ValueError("Model's processor is None")

    # Move BNN model to appropriate device
    if use_gpu and torch.cuda.is_available() and hybrid_model.bnn_model is not None:
        hybrid_model.bnn_model = hybrid_model.bnn_model.to(device)
        print(f"BNN model moved to {device}")

    base_objective_fn = CachedObjectiveFunction(
        hybrid_model=hybrid_model,
        processor=processor,
        max_cache_size=2000
    )

    # Choose objective function based on mode
    if optim_mode == 'optim_all':
        objective_fn = AllVariablesObjectiveFunction(
            base_objective_fn=base_objective_fn,
            cache_size=2000
        )
    elif optim_mode == 'fix_freq':
        objective_fn = FixedFrequencyObjectiveFunction(
            base_objective_fn=base_objective_fn,
            fixed_frequency=fixed_frequency,
            cache_size=2000
        )
    elif optim_mode == 'fix_freq_thick':
        objective_fn = FixedFrequencyUncertainThicknessObjectiveFunction(
            base_objective_fn=base_objective_fn,
            fixed_frequency=fixed_frequency,
            thickness_mean=thickness_mean,
            thickness_std=thickness_std,
            n_samples=n_thickness_samples,
            thickness_bounds=(30, 200),
            cache_size=2000
        )
    else:  # fix_thick
        objective_fn = UncertainThicknessObjectiveFunction(
            hybrid_model=hybrid_model,
            processor=processor,
            thickness_mean=thickness_mean,
            thickness_std=thickness_std,
            n_samples=n_thickness_samples,
            thickness_bounds=(30, 200),
            cache_size=2000,
        )

    # Grid search samples are NOT passed to Bayesian optimization to avoid memory crashes
    # Custom optimizer
    class CustomBayesianOptimizer(BayesianOptimizer):
        def _store_evaluation(self, point, mean_val, std_val):
            pass

        def get_all_evaluations(self):
            return objective_fn.all_evaluations

    # Initialize optimizer
    optimizer = CustomBayesianOptimizer(
        objective_fn=objective_fn,
        param_bounds=param_bounds,
        n_initial=5000,
        n_sobol=min(10 ** len(param_bounds), 21201 // len(param_bounds)),
        seed=42
    )

    # Run optimization
    print("\nStarting optimization...")
    best_result = optimizer.optimize(n_iterations=150, verbose=True)

    # Clear GPU memory after optimization
    if use_gpu:
        torch.cuda.empty_cache()
        gc.collect()

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    all_evaluations = optimizer.get_all_evaluations()
    objective_fn.save_results(all_evaluations, results_dir)
    try:
        create_and_save_all_visualizations(
            optimizer=optimizer,
            all_evaluations=objective_fn.all_evaluations,
            results_dir=results_dir,
            optim_mode=optim_mode,
            trial_id=trial_id,
            fixed_frequency=fixed_frequency,
            thickness_mean=thickness_mean,
            thickness_std=thickness_std
        )
    except Exception as e:
        print(f"⚠ 生成可视化文件时发生错误: {e}")

    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"Best Value: {best_result['best_value']:.6f} ± {best_result['best_std']:.6f}")
    print(f"\nOptimization Mode: {optim_mode}")
    print("\nBest Parameters:")

    param_names = list(param_bounds.keys())
    best_params_full = best_result['best_params']

    # Display parameters based on mode
    if optim_mode == 'optim_all':
        # 5 parameters
        for name, value in zip(param_names, best_params_full):
            if name == 'oxygen_pressure':
                print(f"  {name}: {10 ** value:.2e} (log: {value:.4f})")
            elif name == 'frequency':
                print(f"  {name}: {int(round(value))}")
            else:
                print(f"  {name}: {value:.2f}")
    elif optim_mode == 'fix_freq':
        # 4 parameters + fixed frequency
        param_map = {
            'oxygen_pressure': best_params_full[0],
            'laser_energy_density': best_params_full[1],
            'temperature': best_params_full[2],
            'thickness': best_params_full[3]
        }
        for name in ['oxygen_pressure', 'laser_energy_density', 'temperature']:
            value = param_map[name]
            if name == 'oxygen_pressure':
                print(f"  {name}: {10 ** value:.2e} (log: {value:.4f})")
            else:
                print(f"  {name}: {value:.2f}")
        print(f"  frequency: {int(round(fixed_frequency))} [FIXED]")
        print(f"  thickness: {param_map['thickness']:.2f}")
    elif optim_mode == 'fix_freq_thick':
        param_map = {
            'oxygen_pressure': best_params_full[0],
            'laser_energy_density': best_params_full[1],
            'temperature': best_params_full[2]
        }
        for name in ['oxygen_pressure', 'laser_energy_density', 'temperature']:
            value = param_map[name]
            if name == 'oxygen_pressure':
                print(f"  {name}: {10 ** value:.2e} (log: {value:.4f})")
            else:
                print(f"  {name}: {value:.2f}")
        print(f"  frequency: {int(round(fixed_frequency))} [FIXED]")
        print(f"  thickness: N({thickness_mean:.2f}, {thickness_std:.2f}²) [marginalized]")
    else:  # fix_thick
        # 4 parameters + thickness uncertainty
        for name, value in zip(param_names, best_params_full):
            if name == 'oxygen_pressure':
                print(f"  {name}: {10 ** value:.2e} (log: {value:.4f})")
            elif name == 'frequency':
                print(f"  {name}: {int(round(value))}")
            else:
                print(f"  {name}: {value:.2f}")
        print(f"  thickness: N({thickness_mean:.2f}, {thickness_std:.2f}²) [marginalized]")

    # Display top results
    print("\nTop 10 Results:")
    print("-" * 100)
    print(f"{'Rank': <5} | {'Value': <10} | {'Total_Std': <10} | {'Parameters'}")
    print("-" * 100)

    top_results = optimizer.get_top_results(10)
    for result in top_results:
        if optim_mode == 'optim_all':
            linear_oxygen = 10 ** result['parameters'][0]
            laser_energy = result['parameters'][1]
            temperature = result['parameters'][2]
            frequency = int(round(result['parameters'][3]))
            thickness = result['parameters'][4]
            param_str = f"{linear_oxygen:.2e}, {laser_energy:.2f}, {temperature:.1f}, {frequency}, {thickness:.1f}"
        elif optim_mode == 'fix_freq':
            linear_oxygen = 10 ** result['parameters'][0]
            laser_energy = result['parameters'][1]
            temperature = result['parameters'][2]
            thickness = result['parameters'][3]
            param_str = (f"{linear_oxygen:.2e}, {laser_energy:.2f}, {temperature:.1f},  "
                         f"freq={int(round(fixed_frequency))}[FIXED], {thickness:.1f}")
        elif optim_mode == 'fix_freq_thick':
            linear_oxygen = 10 ** result['parameters'][0]
            laser_energy = result['parameters'][1]
            temperature = result['parameters'][2]
            param_str = (f"{linear_oxygen:.2e}, {laser_energy:.2f}, {temperature:.1f},  "
                         f"freq={int(round(fixed_frequency))}[FIXED], thick~N({thickness_mean:.1f},{thickness_std:.1f}²)")
        else:  # fix_thick
            linear_oxygen = 10 ** result['parameters'][0]
            laser_energy = result['parameters'][1]
            temperature = result['parameters'][2]
            frequency = int(round(result['parameters'][3]))
            param_str = (f"{linear_oxygen:.2e}, {laser_energy:.2f}, {temperature:.1f}, {frequency},  "
                         f"thickness~N({thickness_mean:.1f},{thickness_std:.1f}²)")

        print(f"{result['rank']: <5} | {result['value']: <10.6f} | {result['std']: <10.4f} | {param_str}")

    return optimizer, best_result


def find_trial_models(mode: str, model_type: str):
    """Find trial models from pareto_frontier.csv"""
    base_path = Path(f"./{mode}/XGB_BNN_{model_type}_hybrid_model") / "pareto_solution"
    csv_path = base_path / "pareto_frontier.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Pareto frontier CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    trial_numbers = df['trial_number'].astype(int).tolist()

    valid_trials = []
    for trial_num in trial_numbers:
        trial_dir = base_path / f"trial_{trial_num}"
        model_dir = trial_dir / "model"
        if model_dir.exists():
            model_files = list(model_dir.glob("*.pth")) + list(model_dir.glob("*.pt"))
            if model_files:
                valid_trials.append(f"trial_{trial_num}")

    print(f"Found {len(valid_trials)} valid trial models")
    return valid_trials


def run_optimization_on_pareto_solutions(mode: str, model_type: str, base_output_dir: str, HybridModel,
                                         optim_mode: str = 'fix_thick',
                                         fixed_frequency: float = 5.0,
                                         thickness_mean: float = 20.0,
                                         thickness_std: float = 10.0,
                                         n_thickness_samples: int = 20,
                                         use_gpu: bool = True):
    """
    Main function to run optimization on Pareto solutions.
    """
    print(f"\n{'=' * 60}")
    print("🚀 Starting Optimization on Pareto Solutions")
    print(f"{'=' * 60}\n")

    # Find trial models (which should now be the Pareto solutions)
    trial_models = find_trial_models(mode, model_type)

    if not trial_models:
        print("No trial models found in pareto_solution directory. Exiting.")
        return

    print(f"\nFound {len(trial_models)} trial models in pareto_solution/:")
    print(f"Optimization mode: {optim_mode}")
    if optim_mode == 'optim_all':
        print(f"Optimizing all 5 variables")
    elif optim_mode == 'fix_freq':
        print(f"Fixed frequency: {fixed_frequency}")
    else:  # fix_thick
        print(f"Thickness uncertainty: N({thickness_mean:.2f}, {thickness_std:.2f}²)")
    print(f"GPU acceleration: {use_gpu}")

    for i, trial_id in enumerate(trial_models, 1):
        if optim_mode == 'optim_all':
            mode_suffix = 'all_5vars'
        elif optim_mode == 'fix_freq':
            mode_suffix = f'fix_freq_{fixed_frequency}'
        elif optim_mode == 'fix_freq_thick':  # <-- 新增
            mode_suffix = f'fix_freq_{fixed_frequency}_thick_uncertain'
        else:
            mode_suffix = 'thickness_uncertain'

        check_dir = Path(base_output_dir) / f"model_optim_prediction_results_{trial_id}_{mode_suffix}"
        status = "(Already optimized)" if check_dir.exists() and any(check_dir.iterdir()) else ""
        print(f"{i}. {trial_id} {status}")

    print("\n" + "=" * 60)
    print("STARTING OPTIMIZATION PROCESS")
    print("=" * 60)

    for trial_id in trial_models:
        try:
            optimizer, best_result = run_optimization_for_trial(
                trial_id, mode, model_type, base_output_dir, HybridModel,  # 传入 base_output_dir
                optim_mode=optim_mode,
                fixed_frequency=fixed_frequency,
                thickness_mean=thickness_mean,
                thickness_std=thickness_std,
                n_thickness_samples=n_thickness_samples,
                use_gpu=use_gpu
            )
            if optimizer and best_result:
                print(f"✓ Optimization completed for {trial_id}")
            else:
                print(f"! Optimization skipped or failed for {trial_id}")
        except Exception as e:
            print(f"✗ Error optimizing {trial_id}: {str(e)}")
            continue
        finally:
            # Clear GPU memory after each trial
            if use_gpu:
                torch.cuda.empty_cache()
                gc.collect()

    print("\n" + "=" * 60)
    print("FINISHED Optimization on Pareto Solutions")
    print("=" * 60 + "\n")


# --- END OF optim_prediction.py CONTENT ---

# Main execution for standalone use
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Bayesian optimization on Pareto solutions.')
    parser.add_argument('--mode', type=str, required=True, choices=['tradition', 'LLM'])
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['attention', 'series', 'uncertainty_1', 'uncertainty_2'])
    parser.add_argument('--optim_mode', type=str, default='fix_thick', choices=['optim_all', 'fix_freq', 'fix_thick', 'fix_freq_thick'])
    parser.add_argument('--fixed_frequency', type=float, default=5.0)
    parser.add_argument('--thickness_mean', type=float, default=20.0)
    parser.add_argument('--thickness_std', type=float, default=10.0)
    parser.add_argument('--n_thickness_samples', type=int, default=20)
    parser.add_argument('--use_gpu', action='store_true')

    args = parser.parse_args()

    base_path = f"./{args.mode}/XGB_BNN_{args.model_type}_hybrid_model"
    model_path = f"./{args.mode}/XGB_BNN_{args.model_type}_hybrid_model"
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), model_path))

    try:
        from model import HybridModel

        print(f"Successfully imported HybridModel from {model_path}/model.py")
    except ImportError as e:
        print(f"ERROR: Could not import HybridModel from '{model_path}/model.py'")
        print(f"Import error: {e}")
        sys.exit(1)

    base_output_dir = os.path.join(base_path, "pareto_solution")

    print(f"\n--- Bayesian Optimization (Mode: {args.optim_mode}) ---")
    run_optimization_on_pareto_solutions(
        mode=args.mode,
        model_type=args.model_type,
        base_output_dir=base_output_dir,
        HybridModel=HybridModel,
        optim_mode=args.optim_mode,
        fixed_frequency=args.fixed_frequency,
        thickness_mean=args.thickness_mean,
        thickness_std=args.thickness_std,
        n_thickness_samples=args.n_thickness_samples,
        use_gpu=args.use_gpu
    )

    print("\n✅ Bayesian optimization completed!")