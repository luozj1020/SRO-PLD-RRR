import os
import re
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List
from itertools import combinations
from scipy.optimize import minimize

# 参数边界定义
BOUND = {
    'Oxygen pressure': [-4, 0],
    'Laser energy density': [0.5, 5],
    'Temperature': [500, 900],
    'Frequency': [0, 10],
    'Thickness': [0, 200]
}


def sigmoid(x):
    """Sigmoid函数"""
    return 1 / (1 + np.exp(-x))


class GaussianSuperposition:
    """复合高斯函数计算类"""

    def __init__(self, sobol_csv_path='./data/sobol_samples_results.csv'):
        """
        初始化复合高斯函数

        Args:
            sobol_csv_path: sobol样本数据文件路径
        """
        self.param_bounds = {
            'oxygen_pressure': (-4, 0),
            'laser_energy_density': (0.5, 5),
            'temperature': (500, 900),
            'frequency': (0, 10),
            'thickness': (0, 200),
        }

        # 读取并处理sobol数据
        self._load_and_process_sobol_data(sobol_csv_path)

        # 计算所有高斯函数
        self._compute_all_gaussians()

    def _load_and_process_sobol_data(self, csv_path):
        """加载并处理sobol样本数据"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"找不到sobol样本文件: {csv_path}")

        df = pd.read_csv(csv_path)

        # 提取五维向量数据
        data = df[['oxygen_pressure', 'laser_energy_density', 'temperature',
                   'frequency', 'thickness(measure)']].copy()

        # oxygen_pressure维度取对数
        data['oxygen_pressure'] = np.log10(data['oxygen_pressure'])

        # 归一化
        self.normalized_data = self._normalize_data(data)
        self.all_points = self.normalized_data.values

        print(f"加载Sobol数据: {len(self.all_points)} 个点, {self.all_points.shape[1]} 维")

    def _normalize_data(self, data):
        """将数据归一化到[0,1]范围"""
        normalized_data = data.copy()
        param_names = ['oxygen_pressure', 'laser_energy_density', 'temperature',
                       'frequency', 'thickness(measure)']
        bounds_keys = ['oxygen_pressure', 'laser_energy_density', 'temperature',
                       'frequency', 'thickness']

        for data_col, bounds_key in zip(param_names, bounds_keys):
            min_val, max_val = self.param_bounds[bounds_key]
            normalized_data[data_col] = (data[data_col] - min_val) / (max_val - min_val)

        return normalized_data

    def _compute_bounding_sphere(self, points):
        """计算给定点集的最小外接球"""

        def objective(center):
            distances = np.linalg.norm(points - center, axis=1)
            return np.max(distances)

        initial_center = np.mean(points, axis=0)
        result = minimize(objective, initial_center, method='Nelder-Mead')
        center = result.x
        radius = objective(center)

        return center, radius

    def _create_gaussian_function(self, center, radius):
        """创建高斯函数"""
        A = 1.0
        sigma_squared = radius ** 2 / (2 * np.log(4))

        def gaussian_5d(x):
            distance_sq = np.sum((x - center) ** 2, axis=-1)
            return A * np.exp(-distance_sq / (2 * sigma_squared))

        return gaussian_5d

    def _compute_all_gaussians(self):
        """计算所有组合的高斯函数"""
        self.all_gaussians = []
        n_total_points = len(self.all_points)

        print("开始计算复合高斯函数...")

        # 从n_total_points个点到(n_total_points-2)个点
        for n_points in range(n_total_points, max(n_total_points - 3, 1), -1):
            point_combinations = list(combinations(range(n_total_points), n_points))

            for combo in point_combinations:
                combo_points = self.all_points[list(combo), :]
                center, radius = self._compute_bounding_sphere(combo_points)
                gaussian_func = self._create_gaussian_function(center, radius)
                self.all_gaussians.append(gaussian_func)

        self.n_total_gaussians = len(self.all_gaussians)
        print(f"总共生成了 {self.n_total_gaussians} 个高斯函数")

    def normalize_point(self, point_dict):
        """
        将原始参数点归一化

        Args:
            point_dict: 包含5个参数的字典
                {'oxygen_pressure', 'laser_energy_density', 'temperature',
                 'frequency', 'thickness'}

        Returns:
            归一化后的5维numpy数组
        """
        normalized = np.zeros(5)
        param_keys = ['oxygen_pressure', 'laser_energy_density', 'temperature',
                      'frequency', 'thickness']

        for i, key in enumerate(param_keys):
            value = point_dict[key]

            # oxygen_pressure取对数
            if key == 'oxygen_pressure':
                value = np.log10(value)

            min_val, max_val = self.param_bounds[key]
            normalized[i] = (value - min_val) / (max_val - min_val)

        return normalized

    def compute_gaussian_value(self, point_dict):
        """
        计算给定点的归一化复合高斯函数值

        Args:
            point_dict: 包含5个参数的字典

        Returns:
            归一化的复合高斯函数值 (0-1之间)
        """
        # 归一化输入点
        normalized_point = self.normalize_point(point_dict)

        # 计算所有高斯函数的叠加
        total = 0.0
        for gaussian in self.all_gaussians:
            total += gaussian(normalized_point.reshape(1, -1))[0]

        # 除以高斯函数数量进行归一化
        normalized_value = total / self.n_total_gaussians

        return normalized_value


class ModelScorer:
    def __init__(self, mode: str, model_type: str = None, alpha: float = 0.7,
                 beta: float = 0.3, sobol_csv_path='./data/sobol_samples_results.csv'):
        """
        初始化模型评分器

        Args:
            mode: 模式选择，'tradition' 或 'LLM'
            model_type: 指定要评估的模型类型，如果为None则评估所有模型
            alpha: 分数C的权重参数α，默认0.7
            beta: 分数C的权重参数β，默认0.3
            sobol_csv_path: sobol样本数据文件路径
        """
        if mode not in ['tradition', 'LLM']:
            raise ValueError("mode必须是'tradition'或'LLM'")
        if abs(alpha + beta - 1.0) > 1e-10:
            raise ValueError("alpha + beta 必须等于1")

        self.mode = mode
        self.alpha = alpha
        self.beta = beta
        self.base_path = f'./{mode}'

        # 如果指定了model_type，只评估该类型；否则评估所有类型
        if model_type:
            if model_type not in ['series', 'attention', 'uncertainty_1', 'uncertainty_2']:
                raise ValueError("model_type必须是'series', 'attention', 'uncertainty_1'或'uncertainty_2'之一")
            self.model_types = [model_type]
        else:
            self.model_types = ['series', 'attention', 'uncertainty_1', 'uncertainty_2']

        self.param_names = list(BOUND.keys())
        self.results = []

        # 初始化复合高斯函数计算器
        try:
            self.gaussian_calculator = GaussianSuperposition(sobol_csv_path)
            print("复合高斯函数初始化成功")
        except Exception as e:
            print(f"警告: 复合高斯函数初始化失败: {e}")
            self.gaussian_calculator = None

    def calculate_indicator_function(self, y_vector: np.ndarray, 
                                     bayesian_buffer: float = 0.02,
                                     tolerance: float = 0.05) -> float:
        """
        计算向量中边缘值的比例

        Args:
            y_vector: 5维输入参数向量（已归一化到[0,1]）
            bayesian_buffer: 贝叶斯优化的边界缓冲（默认2%，与bayesian_optimization.py对齐）
            tolerance: 容差比例（默认5%，仅在非贝叶斯优化场景使用）

        Returns:
            边缘值比例（边缘值数量/向量长度）
        
        Note:
            由于贝叶斯优化过程中添加了2%边界缓冲（归一化空间[0.02, 0.98]），
            只有当参数超出缓冲区域时才施加惩罚，缓冲区内不惩罚。
        """
        boundary_count = 0
        for i, param_name in enumerate(self.param_names):
            lower, upper = BOUND[param_name]
            range_span = upper - lower
            
            # 将归一化值转换回原始值
            normalized_value = y_vector[i]
            
            # 贝叶斯优化的缓冲边界（归一化空间）
            buffer_lower = bayesian_buffer  # 0.02
            buffer_upper = 1.0 - bayesian_buffer  # 0.98
            
            # 检查是否超出缓冲区域
            if (normalized_value < buffer_lower) or (normalized_value > buffer_upper):
                boundary_count += 1

        return boundary_count / len(y_vector)

    def _normalize_raw_params(self, raw_params: np.ndarray) -> np.ndarray:
        """
        将原始参数值归一化到[0,1]范围
        
        Args:
            raw_params: 5维原始参数向量 [log_oxygen_pressure, laser_energy_density, 
                                          temperature, frequency, thickness]
        
        Returns:
            归一化后的5维向量
        """
        normalized = np.zeros(5)
        param_keys = ['log_oxygen_pressure', 'laser_energy_density', 'temperature', 
                      'frequency', 'thickness']
        
        # 注意：log_oxygen_pressure的边界是(-4, 0)，对应原始BOUND中的Oxygen pressure
        bounds_mapping = {
            'log_oxygen_pressure': (-4, 0),  # log10(0.0001) ~ log10(1)
            'laser_energy_density': (0.5, 5),
            'temperature': (500, 900),
            'frequency': (0, 10),
            'thickness': (0, 200)
        }
        
        for i, param_name in enumerate(param_keys):
            lower, upper = bounds_mapping[param_name]
            normalized[i] = (raw_params[i] - lower) / (upper - lower)
            # 确保归一化值在[0,1]范围内
            normalized[i] = np.clip(normalized[i], 0, 1)
        
        return normalized

    def calculate_bayesian_edge_penalty(self, evaluation_df: pd.DataFrame) -> float:
        """
        计算贝叶斯优化边缘惩罚 (D)
        
        注意：由于贝叶斯优化添加了2%边界缓冲，只有当参数超出缓冲区域
        （归一化空间 < 0.02 或 > 0.98）时才施加惩罚

        Args:
            evaluation_df: optimization_results.csv数据
        """
        # 取后150条数据
        bayesian_data = evaluation_df.tail(150)

        if len(bayesian_data) == 0:
            return 0.0

        # 计算积分
        numerator = 0.0
        for idx, row in enumerate(bayesian_data.iterrows()):
            row_data = row[1]
            # 提取5维原始向量
            raw_vector = np.array([
                row_data.get('log_oxygen_pressure', 0),
                row_data.get('laser_energy_density', 0),
                row_data.get('temperature', 0),
                row_data.get('frequency', 0),
                row_data.get('thickness', 0)
            ])
            
            # 归一化后再计算边缘惩罚
            normalized_vector = self._normalize_raw_params(raw_vector)
            I_y = self.calculate_indicator_function(normalized_vector)
            numerator += I_y * np.log(idx + 1)

        # 计算分母：∫ln(x+1)dx from 0 to 150
        denominator = 150 * np.log(151) - 150 + np.log(151) - (0 * np.log(1) - 0 + np.log(1))

        penalty_D = numerator / denominator if denominator != 0 else 0

        return penalty_D

    def calculate_optimal_edge_penalty(self, evaluation_df: pd.DataFrame) -> float:
        """
        计算最优值边缘惩罚 (E)
        
        注意：由于贝叶斯优化添加了2%边界缓冲，只有当参数超出缓冲区域
        （归一化空间 < 0.02 或 > 0.98）时才施加惩罚

        Args:
            evaluation_df: optimization_results.csv数据
        """
        # 按mean列降序排序
        sorted_df = evaluation_df.sort_values(by='mean', ascending=False)

        # 取前100个数据
        top_100 = sorted_df.head(100)

        if len(top_100) == 0:
            return 0.0

        penalty_E = 0
        for _, row in top_100.iterrows():
            raw_vector = np.array([
                row.get('log_oxygen_pressure', 0),
                row.get('laser_energy_density', 0),
                row.get('temperature', 0),
                row.get('frequency', 0),
                row.get('thickness', 0)
            ])
            
            # 归一化后再计算边缘惩罚
            normalized_vector = self._normalize_raw_params(raw_vector)
            boundary_ratio = self.calculate_indicator_function(normalized_vector)
            penalty_E += boundary_ratio

        penalty_E /= len(top_100)

        return penalty_E

    def get_pareto_metrics(self, model_type: str, trial_number: int) -> Dict:
        """
        从pareto_frontier.csv或consolidated_evaluation_results.xlsx读取模型性能指标

        Args:
            model_type: 模型类型
            trial_number: trial编号

        Returns:
            包含sto_r2, stability_score, secondary_score, experiment_data_r2, original_sto_r2的字典
        """
        model_folder = os.path.join(self.base_path, f'XGB_BNN_{model_type}_hybrid_model', 'pareto_solution')

        # 首先尝试读取pareto_frontier.csv
        pareto_file = os.path.join(model_folder, 'pareto_frontier.csv')

        if os.path.exists(pareto_file):
            try:
                df = pd.read_csv(pareto_file)
                row = df[df['trial_number'] == trial_number]

                if len(row) > 0:
                    row = row.iloc[0]
                    return {
                        'sto_r2': row['sto_r2_trial'],
                        'stability_score': row['stability_score'],
                        'secondary_score': row['secondary_score'],
                        'experiment_data_r2': row.get('experiment_data_r2', 0),
                        'original_sto_r2': row.get('original_sto_r2', 0)
                    }
            except Exception as e:
                print(f"Warning reading {pareto_file}: {e}")

        # 如果pareto_frontier.csv不存在或未找到数据，尝试读取consolidated_evaluation_results.xlsx
        consolidated_file = os.path.join(model_folder, 'consolidated_evaluation_results.xlsx')

        if os.path.exists(consolidated_file):
            try:
                df = pd.read_excel(consolidated_file)
                row = df[df['trial_number'] == trial_number]

                if len(row) > 0:
                    row = row.iloc[0]
                    return {
                        'sto_r2': row.get('sto_r2_trial', row.get('sto_r2', 0)),
                        'stability_score': row.get('stability_score', 0),
                        'secondary_score': row.get('secondary_score', 0),
                        'experiment_data_r2': row.get('experiment_data_r2', 0),
                        'original_sto_r2': row.get('original_sto_r2', 0)
                    }
            except Exception as e:
                print(f"Warning reading {consolidated_file}: {e}")

        print(f"Warning: Could not find metrics for trial_number {trial_number} in {model_folder}")
        return None

    def calculate_score_C(self, experiment_data_r2: float, original_sto_r2: float) -> float:
        """
        计算分数C: α * min(experiment_data_r2, original_sto_r2) + β * max(0, experiment_data_r2 - original_sto_r2)

        Args:
            experiment_data_r2: 实验数据R²
            original_sto_r2: 原始STO R²

        Returns:
            分数C的值
        """
        min_score = min(experiment_data_r2, original_sto_r2)
        diff_score = max(0, experiment_data_r2 - original_sto_r2)

        score_C = self.alpha * min_score + self.beta * diff_score
        return score_C

    def get_physical_evaluation(self, model_type: str, trial_number: int) -> tuple:
        """
        从Excel文件中读取物理评估结果，并计算复合高斯函数修正

        Args:
            model_type: 模型类型
            trial_number: trial编号

        Returns:
            (rationality, synergy, avg_gaussian_penalty)元组
        """
        # 定义可能的Excel文件路径
        potential_excel_paths = []

        # 方案1: 模式文件夹下的通用文件 (例如 ./tradition/ml_evaluation_results.xlsx)
        general_excel_path = f'./{self.mode}/ml_evaluation_results.xlsx'
        potential_excel_paths.append(('通用文件 (模式目录)', general_excel_path))

        # 方案2: 模型文件夹下的专用文件 (例如 ./tradition/XGB_BNN_series_hybrid_model/ml_evaluation_results_series.xlsx)
        # 这与原始逻辑对应
        specific_excel_path = f'./{self.mode}/XGB_BNN_{model_type}_hybrid_model/ml_evaluation_results_{model_type}.xlsx'
        potential_excel_paths.append((f'专用文件 ({model_type})', specific_excel_path))

        # 方案3: 模型文件夹下的通用文件名 (可能由早期版本脚本生成)
        specific_general_excel_path = f'./{self.mode}/XGB_BNN_{model_type}_hybrid_model/ml_evaluation_results.xlsx'
        potential_excel_paths.append((f'模型目录通用文件 ({model_type})', specific_general_excel_path))

        # 目标工作表名称模式：使用正则表达式匹配所有包含 trial_{trial_number} 的工作表
        target_pattern = re.compile(fr'^{model_type}-trial_{trial_number}(?:_|$)', re.IGNORECASE)

        df = None
        used_path = None
        used_sheet = None

        for path_desc, excel_path in potential_excel_paths:
            if not os.path.exists(excel_path):
                # print(f"调试信息: 文件不存在，跳过 {path_desc}: {excel_path}")
                continue

            try:
                # 获取Excel文件的所有工作表名称
                excel_file = pd.ExcelFile(excel_path)
                sheet_names = excel_file.sheet_names

                # 查找匹配的工作表
                matching_sheets = [sheet for sheet in sheet_names if target_pattern.match(sheet)]

                if not matching_sheets:
                    # 如果没有精确匹配，尝试更宽松的匹配：包含 trial_{trial_number}
                    loose_pattern = re.compile(fr'trial_{trial_number}', re.IGNORECASE)
                    matching_sheets = [sheet for sheet in sheet_names if loose_pattern.search(sheet)]

                if not matching_sheets:
                    continue

                # 使用第一个匹配的工作表
                target_sheet_name = matching_sheets[0]
                df = pd.read_excel(excel_file, sheet_name=target_sheet_name)
                used_path = excel_path
                used_sheet = target_sheet_name
                # print(f"调试信息: 成功从 {path_desc} 读取工作表 {target_sheet_name}")
                break  # 成功读取，跳出循环

            except Exception as e:
                # print(f"调试信息: 从 {path_desc} 读取工作表失败: {e}")
                continue  # 尝试下一个文件路径

        if df is None or df.empty:
            # 如果所有尝试都失败，打印详细警告并返回默认值，避免整个trial评分失败
            print(f"警告: 无法为 {self.mode}/{model_type}/trial_{trial_number} 找到物理评估数据。"
                  f"已尝试匹配包含 'trial_{trial_number}' 的工作表:")
            for desc, path in potential_excel_paths:
                exists = "存在" if os.path.exists(path) else "不存在"
                print(f"  - {desc}: {path} [{exists}]")
            if used_path:
                try:
                    excel_file = pd.ExcelFile(used_path)
                    print(f"  可用工作表: {excel_file.sheet_names}")
                except:
                    pass
            print(f"  将使用默认值 (rationality=0, synergy=0, gaussian_penalty=0) 继续评分。")
            print(f"  请确保已运行 `consolidate&physics_evaluate.py` 并生成了相应的评估结果文件。")
            return 0.0, 0.0, 0.0  # 返回默认值，允许评分继续

        # 如果成功读取数据，打印成功信息
        if used_path and used_sheet:
            print(
                f"信息: 成功为 {self.mode}/{model_type}/trial_{trial_number} 从 {used_path} 加载物理评估数据 (工作表: {used_sheet})")

        # 初始化列表存储修正后的值
        corrected_rationality_list = []
        corrected_synergy_list = []
        gaussian_values = []

        # 如果复合高斯函数计算器可用，计算每行的高斯惩罚并应用到rationality和synergy
        if self.gaussian_calculator is not None:
            for _, row in df.iterrows():
                try:
                    point_dict = {
                        'oxygen_pressure': row['oxygen_pressure'],
                        'laser_energy_density': row['laser_energy_density'],
                        'temperature': row['temperature'],
                        'frequency': row['frequency'],
                        'thickness': row['thickness']
                    }

                    # 计算复合高斯函数值
                    gaussian_val = self.gaussian_calculator.compute_gaussian_value(point_dict)
                    gaussian_values.append(gaussian_val)

                    # 对当前行的rationality和synergy应用高斯惩罚修正
                    corrected_rationality = row['parameter_rationality'] * (1 - gaussian_val)
                    corrected_synergy = row['parameter_synergy'] * (1 - gaussian_val)

                    corrected_rationality_list.append(corrected_rationality)
                    corrected_synergy_list.append(corrected_synergy)

                except Exception as e:
                    print(f"警告: 计算高斯值失败 for row: {e}")
                    # 如果某行计算失败，使用原始值
                    corrected_rationality_list.append(row['parameter_rationality'])
                    corrected_synergy_list.append(row['parameter_synergy'])
                    gaussian_values.append(0.0)
                    continue
        else:
            # 如果高斯计算器不可用，使用原始值
            corrected_rationality_list = df['parameter_rationality'].tolist()
            corrected_synergy_list = df['parameter_synergy'].tolist()
            gaussian_values = [0.0] * len(df)

        # 计算修正后的平均值
        rationality = np.mean(corrected_rationality_list) if corrected_rationality_list else 0.0
        synergy = np.mean(corrected_synergy_list) if corrected_synergy_list else 0.0
        avg_gaussian_penalty = np.mean(gaussian_values) if gaussian_values else 0.0

        return rationality, synergy, avg_gaussian_penalty

    def calculate_model_score(self, model_type: str, trial_number: int, folder_name: str) -> Dict:
        """
        计算单个模型的综合评分

        Args:
            model_type: 模型类型
            trial_number: trial编号
            folder_name: 优化结果文件夹名称

        Returns:
            评分结果字典
        """
        result = {
            'mode': self.mode,
            'model_type': model_type,
            'trial_number': trial_number,
            'folder_name': folder_name
        }

        # A: 读取性能指标
        metrics = self.get_pareto_metrics(model_type, trial_number)

        if metrics is None:
            return None

        A_1 = metrics['sto_r2']
        A_3 = metrics['stability_score']

        # 根据mode处理secondary_score
        if self.mode == 'tradition':
            A_2 = metrics['secondary_score']
        else:  # LLM
            A_2 = sigmoid(1 / metrics['secondary_score'])

        result['STO_R2'] = A_1
        result['A_2'] = A_2
        result['Stability'] = A_3

        # 计算score_A
        score_A = ((A_1 + A_2) / 2) * A_3
        result['score_A'] = score_A

        # C: 计算新的分数C
        experiment_r2 = metrics['experiment_data_r2']
        original_r2 = metrics['original_sto_r2']
        score_C = self.calculate_score_C(experiment_r2, original_r2)
        result['score_C'] = score_C
        result['experiment_data_r2'] = experiment_r2
        result['original_sto_r2'] = original_r2

        # B: 读取物理评估结果并应用高斯修正
        rationality, synergy, gaussian_penalty = self.get_physical_evaluation(model_type, trial_number)

        if rationality is None:
            print(f"Warning: Could not get physical evaluation for {model_type}-trial_{trial_number}")
            return None

        # 应用高斯惩罚修正
        gaussian_correction = 1 - gaussian_penalty

        result['parameter_rationality'] = rationality / 10
        result['parameter_synergy'] = synergy / 10
        result['gaussian_penalty'] = gaussian_penalty
        result['gaussian_correction'] = gaussian_correction

        B_1 = (rationality / 10) * gaussian_correction
        B_2 = (synergy / 10) * gaussian_correction

        # 计算score_B
        score_B = (B_1 + B_2) / 2
        result['score_B'] = score_B
        result['score_B_corrected_rationality'] = B_1
        result['score_B_corrected_synergy'] = B_2

        # D & E: 读取evaluation_results.csv - 使用传入的文件夹名称
        eval_file = os.path.join(
            self.base_path,
            f'XGB_BNN_{model_type}_hybrid_model',
            'pareto_solution',
            folder_name,  # 使用传入的文件夹名称，而不是固定后缀
            'optimization_results.csv'  # 注意：这里应该是 optimization_results.csv，不是 evaluation_results.csv
        )

        try:
            eval_df = pd.read_csv(eval_file)
        except Exception as e:
            print(f"Warning: Could not read {eval_file}: {e}")
            return None

        # D: 贝叶斯优化边缘惩罚
        penalty_D = self.calculate_bayesian_edge_penalty(eval_df)
        result['bayesian_edge_penalty_D'] = penalty_D

        # E: 最优值边缘惩罚
        penalty_E = self.calculate_optimal_edge_penalty(eval_df)
        result['optimal_edge_penalty_E'] = penalty_E

        # 计算最终得分：根据mode使用不同的权重公式
        if self.mode == 'tradition':
            # tradition模式权重：A和B更重要，C次之
            final_score = 0.3 * score_A + 0.3 * score_B + 0.4 * score_C - 0.5 * penalty_D - 0.5 * penalty_E
        else:  # LLM模式
            # LLM模式权重：C更重要（实验验证），A次之，B再次之
            final_score = 0.2 * score_A + 0.1 * score_B + 0.7 * score_C - 0.5 * penalty_D - 0.5 * penalty_E
        
        result['final_score'] = final_score

        return result

    def scan_and_score_all_models(self):
        """
        掃描所有模型並計算評分
        """
        for model_type in self.model_types:
            # 正確的搜索路徑：應在 pareto_solution 子目錄下
            search_base_path = os.path.join(self.base_path, f'XGB_BNN_{model_type}_hybrid_model', 'pareto_solution')

            if not os.path.exists(search_base_path):
                print(f"Warning: Pareto solutions directory not found: {search_base_path}")
                continue

            # 智能扫描：匹配 pareto_solution 下所有符合模式的优化结果文件夹
            # 模式示例：model_optim_prediction_results_trial_1_thickness_uncertain
            #          model_optim_prediction_results_trial_2_all_5vars
            #          model_optim_prediction_results_trial_3_fix_freq_4.0
            pattern = re.compile(r'^model_optim_prediction_results_trial_(\d+)_.+')

            trial_folders = []
            for f in os.listdir(search_base_path):
                folder_path = os.path.join(search_base_path, f)
                if os.path.isdir(folder_path) and pattern.match(f):
                    trial_folders.append(f)

            if not trial_folders:
                print(f"Warning: No trial folders found in {search_base_path}")
                continue

            # 提取trial編號
            trial_info = []
            for folder in trial_folders:
                try:
                    # 使用正则表达式从文件夹名中提取trial编号和完整文件夹名
                    match = re.match(r'^model_optim_prediction_results_trial_(\d+)_(.+)', folder)
                    if match:
                        trial_number = int(match.group(1))
                        suffix = match.group(2)  # 获取后缀部分
                        trial_info.append((trial_number, folder, suffix))
                    else:
                        print(f"Warning: Could not extract trial number from folder name: {folder}")
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not extract trial number from {folder}: {e}")
                    continue

            # 按trial编号分组，处理每个trial的所有优化模式
            trial_groups = {}
            for trial_number, folder, suffix in trial_info:
                if trial_number not in trial_groups:
                    trial_groups[trial_number] = []
                trial_groups[trial_number].append((folder, suffix))

            # 对每个trial进行处理
            for trial_number, folders in trial_groups.items():
                # 默认使用 thickness_uncertain 模式，如果没有则使用第一个找到的文件夹
                target_folder = None
                target_suffix = None

                # 优先寻找 thickness_uncertain 后缀
                for folder, suffix in folders:
                    if 'thickness_uncertain' in suffix:
                        target_folder = folder
                        target_suffix = suffix
                        break

                # 如果没有找到 thickness_uncertain，使用第一个文件夹
                if target_folder is None:
                    target_folder, target_suffix = folders[0]

                print(f"\nProcessing: {self.mode}/{model_type}/trial_{trial_number} (mode: {target_suffix})")

                # 修改：传递文件夹名称给 calculate_model_score
                result = self.calculate_model_score(model_type, trial_number, target_folder)

                if result:
                    # 添加后缀信息到结果中
                    result['optimization_mode'] = target_suffix
                    self.results.append(result)
                    print(f"Score A: {result['score_A']:.4f}, "
                          f"Score B: {result['score_B']:.4f} (高斯修正: {result['gaussian_correction']:.4f}), "
                          f"Score C: {result['score_C']:.4f}, "
                          f"Final Score: {result['final_score']:.4f}")

        return pd.DataFrame(self.results)

    def save_results(self, output_file: str = None):
        """
        保存结果到CSV文件
        """
        if not self.results:
            print("No results to save!")
            return

        if output_file is None:
            if len(self.model_types) == 1:
                # 单个模型类型时，保存到对应的模型文件夹
                model_folder = os.path.join(self.base_path, f'XGB_BNN_{self.model_types[0]}_hybrid_model')
                output_file = os.path.join(model_folder,
                                           f'model_comprehensive_scores_{self.mode}_{self.model_types[0]}.csv')
            else:
                # 多个模型类型时，保存到当前目录
                output_file = f'model_comprehensive_scores_{self.mode}.csv'

        df = pd.DataFrame(self.results)
        df = df.sort_values(by='final_score', ascending=False)
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")

        return df

    def save_top_solutions_by_category(self, output_file: str = None):
        """
        保存每个model_type中分数最高的trial，并将对应的optimization_results.csv按mean列降序保存到对应sheet中

        Args:
            output_file: 输出文件名
        """
        if not self.results:
            print("No results to save!")
            return

        if output_file is None:
            if len(self.model_types) == 1:
                # 单个模型类型时，保存到对应的模型文件夹
                model_folder = os.path.join(self.base_path, f'XGB_BNN_{self.model_types[0]}_hybrid_model')
                output_file = os.path.join(model_folder,
                                           f'top_solutions_{self.mode}_{self.model_types[0]}.xlsx')
            else:
                # 多个模型类型时，保存到当前目录
                output_file = f'top_solutions_by_category_{self.mode}.xlsx'

        df = pd.DataFrame(self.results)

        # 按model_type分组，找到每个组中final_score最高的行
        top_solutions = df.loc[df.groupby('model_type')['final_score'].idxmax()]

        # 排序
        top_solutions = top_solutions.sort_values('model_type')

        # 选择需要的列
        columns_to_keep = [
            'mode', 'model_type', 'trial_number', 'folder_name', 'optimization_mode', 'score_A', 'score_B', 'score_C',
            'gaussian_penalty', 'gaussian_correction',
            'bayesian_edge_penalty_D', 'optimal_edge_penalty_E', 'final_score'
        ]
        top_solutions = top_solutions[columns_to_keep]

        # 使用ExcelWriter创建多sheet的Excel文件
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # 保存top_solutions到第一个sheet
            top_solutions.to_excel(writer, sheet_name='Top_Solutions', index=False)

            # 为每个top solution添加对应的optimization_results.csv到单独的sheet
            for _, row in top_solutions.iterrows():
                model_type = row['model_type']
                trial_number = row['trial_number']
                folder_name = row['folder_name']

                # 构建optimization_results.csv文件路径 - 使用folder_name
                eval_file_path = os.path.join(
                    self.base_path,
                    f'XGB_BNN_{model_type}_hybrid_model',
                    'pareto_solution',
                    folder_name,  # 使用folder_name，不是固定后缀
                    'optimization_results.csv'  # 修改为 optimization_results.csv
                )

                # 检查文件是否存在
                if os.path.exists(eval_file_path):
                    try:
                        # 读取optimization_results.csv
                        eval_df = pd.read_csv(eval_file_path)

                        # 按mean列降序排序
                        eval_df_sorted = eval_df.sort_values(by='mean', ascending=False)

                        # 创建sheet名称（确保sheet名有效，不超过31字符）
                        sheet_name = f"{model_type}_trial_{trial_number}"
                        if len(sheet_name) > 31:
                            sheet_name = sheet_name[:31]

                        # 保存到Excel
                        eval_df_sorted.to_excel(writer, sheet_name=sheet_name, index=False)
                        print(f"Saved evaluation results for {model_type} trial {trial_number} to sheet: {sheet_name}")

                    except Exception as e:
                        print(f"Error processing evaluation results for {model_type} trial {trial_number}: {e}")
                else:
                    print(f"Warning: Evaluation results file not found: {eval_file_path}")

        print(f"\nTop solutions saved to {output_file}")
        return top_solutions


def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description='模型评估系统')
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['tradition', 'LLM'],
        help='选择模式：tradition 或 LLM'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default=None,
        choices=['series', 'attention', 'uncertainty_1', 'uncertainty_2'],
        help='指定要评估的模型类型，如果不指定则评估所有模型类型'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.3,
        help='分数C的权重参数α，默认0.3'
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=0.7,
        help='分数C的权重参数β，默认0.7'
    )

    args = parser.parse_args()

    scorer = ModelScorer(args.mode, args.model_type, args.alpha, args.beta)
    results_df = scorer.scan_and_score_all_models()

    if len(results_df) == 0:
        print("未找到任何有效结果！")
        return

    final_df = scorer.save_results()
    top_solutions_df = scorer.save_top_solutions_by_category()

    print("\n" + "=" * 80)
    model_info = f" - {args.model_type}" if args.model_type else ""
    print(f"评分汇总统计 (Mode: {args.mode}{model_info}, α={args.alpha}, β={args.beta})")
    print("=" * 80)
    print(f"\n总共评估模型数: {len(results_df)}")
    print(f"最高分: {results_df['final_score'].max():.4f}")
    print(f"最低分: {results_df['final_score'].min():.4f}")
    print(f"平均分: {results_df['final_score'].mean():.4f}")
    print(f"中位数: {results_df['final_score'].median():.4f}")

    print("\n前5名模型:")
    top_5 = results_df.nlargest(5, 'final_score')[
        ['model_type', 'trial_number', 'score_A', 'score_B', 'score_C', 'final_score']
    ]
    print(top_5.to_string(index=False))

    print("\n" + "=" * 80)
    print("每个类别的最佳trial")
    print("=" * 80)
    print(top_solutions_df.to_string(index=False))


if __name__ == "__main__":
    main()
