import glob
import pandas as pd
import os
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
import re
from pathlib import Path
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def consolidate_csv(input_dir: str, output_dir: str, model_type: Optional[str] = None) -> str:
    """
    Consolidate multiple CSV files into one Excel file

    Args:
        input_dir: Root directory containing pareto_solution
        output_dir: Output directory for results
        model_type: Specific XGB_BNN_* directory name to process (e.g., 'series')
                   If None, process all XGB_BNN_* directories

    Returns:
        Path to output Excel file
    """
    # 修改：确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 修改：根据是否有model_type设置不同的输出文件名
    if model_type:
        output_excel = os.path.join(output_dir, f"consolidated_evaluation_results_{model_type}.xlsx")
    else:
        output_excel = os.path.join(output_dir, "consolidated_evaluation_results.xlsx")

    if model_type:
        # 关键修改点：input_dir 此时已指向 './{mode}/XGB_BNN_{model_type}_hybrid_model'
        # 我们需要在其下找到 pareto_solution 文件夹
        pattern = os.path.join(input_dir, 'pareto_solution',  # 此处路径顺序改变
                               'model_optim_prediction_results_trial_*',
                               'optimization_results.csv')
        csv_files = glob.glob(pattern)

        if not csv_files:
            logger.warning(f"No CSV files found in {input_dir}/pareto_solution using pattern: {pattern}")
            # 可以尝试一个备选模式，但主要逻辑已修改
            alt_pattern = os.path.join(input_dir, '**', 'pareto_solution',
                                       'model_optim_prediction_results_trial_*',
                                       'optimization_results.csv')
            csv_files = glob.glob(alt_pattern, recursive=True)
    else:
        # 当未指定 model_type 时，遍历所有可能的模型目录
        pattern = os.path.join(input_dir, '**', 'pareto_solution',  # 使用 ** 进行递归查找
                               'model_optim_prediction_results_trial_*',
                               'optimization_results.csv')
        csv_files = glob.glob(pattern, recursive=True)

    if not csv_files:
        logger.warning(f"No CSV files found for consolidation")
        return output_excel

    with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
        for csv_path in csv_files:
            try:
                path_parts = Path(csv_path).parts
                # 方案：优先使用函数参数 model_type，如果未提供，则尝试从路径中解析（旧逻辑）
                model_idx_for_sheet = model_type  # 直接使用传入的模型类型，如 ‘series’
                trial_idx = path_parts[-2].replace('model_optim_prediction_results_trial_', '').replace(
                    '_thickness_uncertain', '')
                sheet_name = f"{model_idx_for_sheet}-trial_{trial_idx}"[:31]  # 将生成如 ‘series-trial_1608’

                df = pd.read_csv(csv_path)
                if 'mean' in df.columns:
                    df = df.sort_values(by='mean', ascending=False).head(100)
                else:
                    df = df.head(100)

                df.to_excel(writer, sheet_name=sheet_name, index=False)
                logger.info(f"Processed: {csv_path} -> Sheet: {sheet_name}")

            except Exception as e:
                logger.error(f"Error processing {csv_path}: {str(e)}")

    logger.info(f"Consolidation complete! Results saved to: {output_excel}")
    return output_excel


@dataclass
class EvaluationResult:
    sheet_name: str
    total_score: float
    parameter_scores: Dict[str, float]
    synergy_score: Dict[str, float]
    rrr_value: float
    parameters: Dict[str, float]
    llm_explanations: Dict[str, List[str]]


class DomainLLMEvaluator:
    def __init__(self, LLM_finetuned_model_path: Optional[str] = None,
                 LLM_base_model_path: Optional[str] = None):
        self.max_retries = 114514
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            bnb_4bit_use_double_quant=True
        )

        try:
            logger.info("Loading tokenizer...")
            # 使用微调模型路径的tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                LLM_finetuned_model_path,  # 参数名修改
                trust_remote_code=True
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("Loading base model with 4-bit quantization...")
            # 直接使用传入的base_model_path
            base_model = AutoModelForCausalLM.from_pretrained(
                LLM_base_model_path,  # 参数名修改
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            logger.info("Loading LoRA adapter...")
            self.llm_model = PeftModel.from_pretrained(
                base_model,
                LLM_finetuned_model_path  # 参数名修改
            )

            # 设置为评估模式
            self.llm_model.eval()

            logger.info("Model initialization complete")

        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise

    def generate_response(self, prompt: str) -> str:
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)

            with torch.no_grad():  # 添加no_grad
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    temperature=0.3,
                    top_p=0.85,
                    repetition_penalty=1.1,
                    do_sample=True
                )

            # 只解码新生成的部分
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            return response

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

    def _extract_score_from_response(self, response: str) -> Optional[float]:
        # 增强正则表达式
        patterns = [
            r'评分[:：]\s*(\d+\.?\d*)',
            r'[Ss]core[:：]\s*(\d+\.?\d*)',
            r'得分[:：]\s*(\d+\.?\d*)',
            r'最终评分[:：]\s*(\d+\.?\d*)',
            r'综合评分[:：]\s*(\d+\.?\d*)',
            r'协调性评分[:：]\s*(\d+\.?\d*)',
            r'分数[:：]\s*(\d+\.?\d*)',
            r'评分为\s*(\d+\.?\d*)',  # 添加更灵活的模式
            r'得分为\s*(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*分',  # 匹配 "8.5分"
        ]

        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    score = float(match.group(1))
                    # 验证分数范围
                    if 0 <= score <= 10:
                        return score
                except (ValueError, TypeError):
                    continue

        # logger.warning(f"Failed to extract score from: {response[:200]}")
        return None

    def generate_and_extract_score(self, prompt: str) -> Tuple[float, str]:
        """
        生成响应并从响应中提取分数

        Args:
            prompt: 输入的提示文本

        Returns:
            元组 (分数, 解释文本)
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Generating response (attempt {attempt + 1}/{self.max_retries})...")
                response = self.generate_response(prompt)

                # 提取分数
                score = self._extract_score_from_response(response)

                if score is not None:
                    return score, response
                else:
                    logger.warning(f"Failed to extract score from response, retrying...")

            except Exception as e:
                logger.error(f"Error in generate_and_extract_score (attempt {attempt + 1}): {str(e)}")

        # 如果所有尝试都失败，返回默认值
        logger.error("All attempts failed, returning default score")
        return 0.0, "Evaluation failed after multiple attempts"


class MLModelEvaluator:
    def __init__(self, domain_llm: DomainLLMEvaluator):
        self.domain_llm = domain_llm
        self.evaluation_weights = {
            'parameter_rationality': 0.50,
            'parameter_synergy': 0.50
        }
        self.param_names = [
            'oxygen_pressure',
            'laser_energy_density',
            'temperature',
            'frequency',
            'thickness'
        ]

    def parse_parameters(self, params_list: List[float]) -> Dict[str, float]:
        return dict(zip(self.param_names, params_list))

    def evaluate_ml_predictions(self, prediction_results: List[Dict]) -> Dict:
        logger.info("Starting ML model prediction evaluation...")
        detailed_evaluations = []

        for i, result in enumerate(prediction_results):
            logger.info(f"Evaluating prediction {i + 1}/{len(prediction_results)}")
            try:
                evaluation = self.evaluate_single_prediction(result)
                detailed_evaluations.append(evaluation)
            except Exception as e:
                logger.error(f"Failed to evaluate prediction: {str(e)}")

        report = self.generate_evaluation_report(detailed_evaluations)
        return {
            'detailed_evaluations': detailed_evaluations,
            'comprehensive_report': report,
        }

    def evaluate_single_prediction(self, prediction_result: Dict) -> EvaluationResult:
        sheet_name = prediction_result.get('sheet_name', 'unknown')
        params = self.parse_parameters(prediction_result['params'])
        rrr_value = prediction_result['value']
        std_dev = prediction_result['std']

        param_scores, param_explanations = self._evaluate_with_retry(
            self.evaluate_parameter_rationality,
            params,
            "Parameter rationality"
        )

        synergy_scores, synergy_explanations = self._evaluate_with_retry(
            self.evaluate_parameter_synergy,
            params,
            "Parameter synergy"
        )

        avg_param_score = self._calculate_robust_average(param_scores)
        avg_synergy_score = self._calculate_robust_average(synergy_scores)

        total_score = (
                avg_param_score * self.evaluation_weights['parameter_rationality'] +
                avg_synergy_score * self.evaluation_weights['parameter_synergy']
        )

        llm_explanations = {
            'parameter_rationality': param_explanations,
            'parameter_synergy': synergy_explanations,
        }

        return EvaluationResult(
            sheet_name=sheet_name,
            total_score=total_score,
            parameter_scores={'average': avg_param_score, 'evaluations': param_scores},
            synergy_score={'average': avg_synergy_score, 'evaluations': synergy_scores},
            rrr_value=rrr_value,
            parameters=params,
            llm_explanations=llm_explanations
        )

    def _evaluate_with_retry(self, func, params: Dict[str, float], name: str) -> Tuple[List[float], List[str]]:
        scores = []
        explanations = []

        for i in range(3):
            logger.info(f"  - {name} evaluation ({i + 1}/3)...")
            try:
                score, explanation = func(params)
                scores.append(score)
                explanations.append(explanation)
            except Exception as e:
                logger.error(f"Evaluation failed: {str(e)}")
                scores.append(0.0)
                explanations.append(f"Evaluation failed: {str(e)}")

        return scores, explanations

    def _calculate_robust_average(self, scores: List[float]) -> float:
        if len(scores) < 3:
            return np.mean(scores)
        sorted_scores = sorted(scores)
        return np.mean(sorted_scores[1:-1])

    def evaluate_parameter_rationality(self, params: Dict[str, float]) -> Tuple[float, str]:
        prompt = f"""您是一位材料科学专家，负责评估SRO薄膜在STO衬底上PLD生长条件的合理性。请根据以下参数组合进行专业评估：

参数组合详情：
- 温度: {params['temperature']:.1f} °C 
- 氧压: {10 ** params['oxygen_pressure']:.2e} mbar 
- 激光能量密度: {params['laser_energy_density']:.3f} J/cm² 
- 激光频率: {params['frequency']:.1f} Hz 
- 薄膜厚度: {params['thickness']:.1f} nm 

评估维度及权重：
1. 参数范围合理性 (权重30%): 每个参数是否在推荐范围内
2. 薄膜质量影响 (权重30%): 参数组合是否有利于获得高结晶质量的SRO薄膜
3. RRR值优化 (权重25%): 参数组合是否有利于获得高RRR值
4. 工艺稳定性 (权重15%): 参数组合是否适合稳定重复的PLD工艺

评分标准：
- 9-10分: 参数组合完全符合所有标准，是理想选择
- 7-8分: 参数组合基本合理，有轻微可优化空间
- 5-6分: 部分参数超出范围，需要调整
- 3-4分: 多个参数存在问题，需要重大调整
- 1-2分: 参数组合完全不可行

请给出：
1. 详细的技术分析（重点分析参数的合理性）
2. 明确的综合评分（格式：综合评分: X.X）
"""
        return self.domain_llm.generate_and_extract_score(prompt)

    def evaluate_parameter_synergy(self, params: Dict[str, float]) -> Tuple[float, str]:
        prompt = f"""您是一位PLD工艺专家，负责评估SRO薄膜在STO衬底上PLD生长条件的协调性。请根据以下参数组合进行专业评估：

参数组合详情：
- 温度: {params['temperature']:.1f} °C
- 氧压: {10 ** params['oxygen_pressure']:.2e} mbar
- 激光能量密度: {params['laser_energy_density']:.3f} J/cm²
- 激光频率: {params['frequency']:.1f} Hz
- 薄膜厚度: {params['thickness']:.1f} nm

关键协同关系分析：
1. 温度-氧压匹配度: 高温通常需要较高氧压补偿氧空位
2. 激光参数协调性: 能量密度与频率的组合是否避免过度轰击或沉积不足
3. 厚度相关参数: 沉积速率是否与厚度要求匹配
4. 整体协同效应: 是否存在相互冲突的参数设置，参数组合是否共同支持高质量薄膜生长

评分标准：
- 9-10分: 参数间完美协同，相互增强
- 7-8分: 良好的协同效果，有轻微优化空间
- 5-6分: 存在可接受的冲突，需要调整
- 3-4分: 明显的参数冲突，影响薄膜质量
- 1-2分: 严重的参数冲突，工艺不可行

请给出：
1. 详细的协同效应分析
2. 明确的协调性评分（格式：协调性评分: X.X）
"""
        return self.domain_llm.generate_and_extract_score(prompt)

    def generate_evaluation_report(self, detailed_evaluations: List[EvaluationResult]) -> Dict:
        if not detailed_evaluations:
            return {}

        total = len(detailed_evaluations)
        avg_score = sum(eval.total_score for eval in detailed_evaluations) / total

        sorted_evals = sorted(detailed_evaluations, key=lambda x: x.total_score, reverse=True)
        best_pred = sorted_evals[0] if sorted_evals else None

        score_ranges = {
            'excellent': sum(1 for e in detailed_evaluations if e.total_score >= 8.0),
            'good': sum(1 for e in detailed_evaluations if 6.0 <= e.total_score < 8.0),
            'fair': sum(1 for e in detailed_evaluations if 4.0 <= e.total_score < 6.0),
            'poor': sum(1 for e in detailed_evaluations if e.total_score < 4.0)
        }

        param_analysis = self._analyze_parameter_trends(detailed_evaluations)

        param_rationality_avg = sum(
            e.parameter_scores['average'] for e in detailed_evaluations) / total

        return {
            'summary': {
                'total_predictions': total,
                'average_score': avg_score,
                'best_prediction': {
                    'score': best_pred.total_score if best_pred else 0.0,
                    'rrr_value': best_pred.rrr_value if best_pred else 0.0,
                    'parameters': best_pred.parameters if best_pred else {}
                },
                'score_distribution': score_ranges
            },
            'parameter_analysis': param_analysis,
            'detailed_scores': {
                'parameter_rationality': param_rationality_avg,
                'parameter_synergy': sum(
                    e.synergy_score['average'] for e in detailed_evaluations) / total,
            }
        }

    def _analyze_parameter_trends(self, evaluations: List[EvaluationResult]) -> Dict:
        if not evaluations:
            return {}

        param_names = list(evaluations[0].parameters.keys())
        all_params = {param: [] for param in param_names}

        for eval in evaluations:
            for param, value in eval.parameters.items():
                all_params[param].append(value)

        param_stats = {}
        for param, values in all_params.items():
            param_stats[param] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'range': np.max(values) - np.min(values)
            }

        return param_stats


def save_evaluation_results(all_results: Dict[str, pd.DataFrame], filename: str,
                            existing_xls: Optional[pd.ExcelFile] = None):
    """
    保存评估结果，同时保留已存在的完整sheet

    Args:
        all_results: 本次处理的结果字典 {sheet_name: dataframe}
        filename: 输出文件名
        existing_xls: 已存在的Excel文件对象（如果有）
    """
    # 如果存在旧文件，先读取所有已完成的sheet
    existing_sheets = {}
    if existing_xls is not None:
        for sheet_name in existing_xls.sheet_names:
            if sheet_name not in all_results:  # 只保留本次未处理的sheet
                try:
                    existing_sheets[sheet_name] = pd.read_excel(existing_xls, sheet_name=sheet_name)
                    logger.info(f"Preserving existing sheet: {sheet_name}")
                except Exception as e:
                    logger.warning(f"Error reading existing sheet {sheet_name}: {str(e)}")

    # 合并已存在的和新处理的结果
    combined_results = {**existing_sheets, **all_results}

    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        for sheet_name, df in combined_results.items():
            # 移除sheet_name列（如果存在）
            if 'sheet_name' in df.columns:
                df = df.drop(columns=['sheet_name'])

            df.to_excel(writer, sheet_name=sheet_name, index=False)

            worksheet = writer.sheets[sheet_name]
            for i, col_name in enumerate(df.columns):
                width = 60 if 'explanation' in col_name else 20
                worksheet.set_column(i, i, width)

    logger.info(f"Results saved to: {filename} ({len(combined_results)} sheets total)")


def compare_dataframes(pred_df: pd.DataFrame, exist_df: pd.DataFrame, key_columns: List[str]) -> pd.DataFrame:
    pred_df['merge_key'] = pred_df[key_columns].apply(tuple, axis=1)
    exist_df['merge_key'] = exist_df[key_columns].apply(tuple, axis=1)

    missing_mask = ~pred_df['merge_key'].isin(exist_df['merge_key'])
    missing_df = pred_df[missing_mask].copy()

    pred_df.drop(columns=['merge_key'], inplace=True)
    exist_df.drop(columns=['merge_key'], inplace=True)
    missing_df.drop(columns=['merge_key'], inplace=True)

    return missing_df


def evaluate_main(LLM_finetuned_model_path: str, LLM_base_model_path: str,
                  output_dir: str, model_type: Optional[str] = None):
    try:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"

        # 修改：根据是否有model_type参数设置不同的文件路径
        if model_type:
            # 如果有model_type，文件保存在XGB_BNN_*_hybrid_model文件夹下
            base_path = output_dir  # 直接使用传入的output_dir
            # 注意：文件名应该包含model_type后缀
            input_file = os.path.join(base_path, f"consolidated_evaluation_results_{model_type}.xlsx")
            output_file = os.path.join(base_path, f"ml_evaluation_results_{model_type}.xlsx")
            temp_file = os.path.join(base_path, f"evaluated_keys_{model_type}.pkl")

            # 确保目录存在
            os.makedirs(base_path, exist_ok=True)
            logger.info(f"Using model-specific output path: {base_path}")
        else:
            # 如果没有model_type，文件保存在output_dir下
            input_file = os.path.join(output_dir, "consolidated_evaluation_results.xlsx")
            output_file = os.path.join(output_dir, "ml_evaluation_results.xlsx")
            temp_file = os.path.join(output_dir, "evaluated_keys.pkl")
            logger.info(f"Using general output path: {output_dir}")

        # 当合并文件不存在时，调用 consolidate_csv
        if not os.path.exists(input_file):
            logger.info("Consolidated file not found, running consolidation...")
            # 关键修改点：确定正确的输入目录路径
            if model_type:
                # 对于特定模型，input_dir 应指向 './{mode}/XGB_BNN_{model_type}_hybrid_model'
                # 根据代码逻辑，output_dir 被设置为该路径，所以可以直接使用
                root_dir_for_consolidate = output_dir  # 例如: ./mode/XGB_BNN_series_hybrid_model
                logger.info(f"Consolidating from model-specific directory: {root_dir_for_consolidate}")
                consolidate_csv(root_dir_for_consolidate, base_path, model_type)
            else:
                # 对于通用情况，input_dir 和 output_dir 可能相同，或需要进一步确认
                # 这里假设 output_dir 就是包含所有 XGB_BNN_* 目录的根目录
                root_dir_for_consolidate = output_dir
                consolidate_csv(root_dir_for_consolidate, output_dir, model_type)

        evaluated_keys = load_evaluated_keys(temp_file)

        logger.info(f"Reading predictions from {input_file}")
        consolidated_xls = pd.ExcelFile(input_file)
        all_results = {}

        output_exists = os.path.exists(output_file)
        output_xls = None
        if output_exists:
            try:
                output_xls = pd.ExcelFile(output_file)
            except Exception as e:
                logger.warning(f"Error reading output file: {str(e)}")
                output_exists = False

        # 如果指定了model_type，只处理对应的sheets
        if model_type:
            # 从model_type生成sheet名称前缀
            sheet_prefix = model_type.replace('XGB_BNN_', '').replace('_hybrid_model', '')
            logger.info(f"Filtering sheets by model directory: {model_type}")
            logger.info(f"Looking for sheet prefix: {sheet_prefix}")

            # 筛选consolidated_xls中的sheets
            filtered_sheets = [s for s in consolidated_xls.sheet_names
                               if s.startswith(sheet_prefix + '-') or s.startswith(sheet_prefix + '_')]

            if not filtered_sheets:
                logger.warning(f"No sheets found matching the specified model directory")
                logger.info(f"Available sheets: {consolidated_xls.sheet_names}")
                return

            logger.info(f"Found {len(filtered_sheets)} sheets to process: {filtered_sheets}")
            sheets_to_check = filtered_sheets
        else:
            sheets_to_check = consolidated_xls.sheet_names

        sheets_to_process = identify_sheets_to_process(consolidated_xls, output_xls, sheets_to_check)

        # 统计需要处理的sheet
        sheets_needing_work = {k: v for k, v in sheets_to_process.items() if v != "complete"}

        if not sheets_needing_work:
            logger.info("✅ All specified sheets are complete. Skipping evaluation.")
            return

        logger.info(f"Found {len(sheets_needing_work)} sheets to process:")
        for sheet, status in sheets_needing_work.items():
            logger.info(f"  - {sheet}: {status}")

        logger.info("Initializing domain LLM evaluator...")
        domain_llm = DomainLLMEvaluator(LLM_finetuned_model_path, LLM_base_model_path)

        logger.info("Initializing ML model evaluator...")
        evaluator = MLModelEvaluator(domain_llm)

        # 只处理需要评估的sheet
        for sheet_name, status in sheets_to_process.items():
            if status == "complete":
                logger.info(f"\nSkipping complete sheet: {sheet_name}")
                continue

            logger.info(f"\nProcessing sheet: {sheet_name} (status: {status})")

            pred_df = pd.read_excel(consolidated_xls, sheet_name=sheet_name).head(10)
            pred_df['sheet_name'] = sheet_name

            results_df = pred_df.copy()

            eval_columns = [
                               'total_score', 'parameter_rationality', 'parameter_synergy'
                           ] + [f'{prefix}_{i}' for prefix in ['param_rationality_score', 'synergy_score',
                                                               'parameter_rationality_explanation',
                                                               'parameter_synergy_explanation']
                                for i in range(1, 4)]

            for col in eval_columns:
                results_df[col] = None

            key_columns = [
                'log_oxygen_pressure', 'oxygen_pressure', 'laser_energy_density',
                'temperature', 'frequency', 'thickness', 'mean', 'std'
            ]

            if output_exists and sheet_name in output_xls.sheet_names:
                existing_df = pd.read_excel(output_xls, sheet_name=sheet_name).head(10)

                missing_df = compare_dataframes(pred_df, existing_df, key_columns)

                if not missing_df.empty:
                    logger.info(f"Found {len(missing_df)} missing rows to evaluate")
                    evaluated_rows = evaluate_missing_rows(missing_df, evaluator, temp_file, evaluated_keys)
                    results_df = pd.concat([existing_df, evaluated_rows]).head(10)
                else:
                    results_df = existing_df

            else:
                logger.info(f"Evaluating all rows in new sheet: {sheet_name}")
                evaluated_rows = evaluate_missing_rows(pred_df, evaluator, temp_file, evaluated_keys)
                results_df = evaluated_rows

            all_results[sheet_name] = results_df

        # 保存结果时传入已存在的Excel对象
        if all_results:
            save_evaluation_results(
                all_results=all_results,
                filename=output_file,
                existing_xls=output_xls  # 传入已存在的Excel对象
            )

        cleanup_temp_file(temp_file)

        logger.info("✅ Evaluation completed successfully!")

    except Exception as e:
        logger.exception(f"Critical error in evaluation: {str(e)}")


def load_evaluated_keys(temp_file: str) -> Set[str]:
    if os.path.exists(temp_file):
        try:
            return set(pd.read_pickle(temp_file))
        except Exception:
            pass
    return set()


def identify_sheets_to_process(consolidated_xls: pd.ExcelFile, output_xls: Optional[pd.ExcelFile],
                               sheets_to_check: Optional[List[str]] = None) -> Dict[str, str]:
    """
    识别需要处理的sheets

    Args:
        consolidated_xls: 输入的Excel文件
        output_xls: 输出的Excel文件（可能存在）
        sheets_to_check: 要检查的sheet列表，如果为None则检查所有sheets
    """
    sheets_to_process = {}
    output_sheets = set(output_xls.sheet_names) if output_xls else set()

    # 确定要检查的sheet列表
    if sheets_to_check is None:
        sheets_to_check = consolidated_xls.sheet_names

    for sheet_name in sheets_to_check:
        if not output_xls or sheet_name not in output_sheets:
            sheets_to_process[sheet_name] = "missing"
            continue

        try:
            output_df = pd.read_excel(output_xls, sheet_name=sheet_name)

            if len(output_df) < 10:
                sheets_to_process[sheet_name] = "incomplete"
                continue

            eval_columns = [
                'total_score', 'parameter_rationality', 'parameter_synergy',
                'param_rationality_score_1', 'param_rationality_score_2', 'param_rationality_score_3',
                'synergy_score_1', 'synergy_score_2', 'synergy_score_3'
            ]

            if any(output_df[col].isnull().any() for col in eval_columns if col in output_df.columns):
                sheets_to_process[sheet_name] = "incomplete"
            else:
                sheets_to_process[sheet_name] = "complete"

        except Exception:
            sheets_to_process[sheet_name] = "incomplete"

    return sheets_to_process


def evaluate_missing_rows(df: pd.DataFrame, evaluator: MLModelEvaluator,
                          temp_file: str, evaluated_keys: Set[str]) -> pd.DataFrame:
    results = []

    for _, row in df.iterrows():
        row_key = "|".join([
            row['sheet_name'],
            str(row['log_oxygen_pressure']),
            str(row['laser_energy_density']),
            str(row['temperature']),
            str(row['frequency']),
            str(row['thickness'])
        ])

        if row_key in evaluated_keys:
            continue

        logger.info(f"Evaluating row: {row_key}")

        try:
            params = [
                row['log_oxygen_pressure'],
                row['laser_energy_density'],
                row['temperature'],
                row['frequency'],
                row['thickness']
            ]

            prediction_result = {
                'value': row['mean'],
                'std': row['std'],
                'params': params,
                'sheet_name': row['sheet_name']
            }

            eval_result = evaluator.evaluate_single_prediction(prediction_result)

            result_row = row.to_dict()
            result_row.update({
                'total_score': eval_result.total_score,
                'parameter_rationality': eval_result.parameter_scores['average'],
                'parameter_synergy': eval_result.synergy_score['average']
            })

            for i in range(3):
                if i < len(eval_result.parameter_scores['evaluations']):
                    result_row[f'param_rationality_score_{i + 1}'] = eval_result.parameter_scores['evaluations'][i]

                if i < len(eval_result.synergy_score['evaluations']):
                    result_row[f'synergy_score_{i + 1}'] = eval_result.synergy_score['evaluations'][i]

                if i < len(eval_result.llm_explanations['parameter_rationality']):
                    result_row[f'parameter_rationality_explanation_{i + 1}'] = \
                        eval_result.llm_explanations['parameter_rationality'][i]

                if i < len(eval_result.llm_explanations['parameter_synergy']):
                    result_row[f'parameter_synergy_explanation_{i + 1}'] = \
                        eval_result.llm_explanations['parameter_synergy'][i]

            results.append(result_row)
            evaluated_keys.add(row_key)

            if len(evaluated_keys) % 5 == 0:
                pd.to_pickle(list(evaluated_keys), temp_file)

        except Exception as e:
            logger.error(f"Error evaluating row: {str(e)}")

    pd.to_pickle(list(evaluated_keys), temp_file)

    return pd.DataFrame(results)


def cleanup_temp_file(temp_file: str):
    if os.path.exists(temp_file):
        try:
            os.remove(temp_file)
            logger.info(f"Temporary file cleaned up: {temp_file}")
        except Exception as e:
            logger.warning(f"Error cleaning up temp file: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PLD Model Evaluation Pipeline')
    parser.add_argument('--mode', type=str, required=True,
                        help='Input directory with pareto_solution subdirectory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (defaults to input directory)')
    parser.add_argument('--LLM_finetuned_model_path', type=str, default='../LLM/pld_finetuned_model',
                        help='Path to the fine-tuned LLM model')
    parser.add_argument('--LLM_base_model_path', type=str, default='../LLM/models/DeepSeek-R1-Distill-Qwen-32B',
                        help='Path to the base LLM model')
    parser.add_argument('--model_type', type=str, default=None,
                        help='Specific XGB_BNN_* directory to evaluate (e.g., series)')

    args = parser.parse_args()

    if args.model_type:
        # 关键修改点：output_dir 指向模型文件夹 (例如 ./mode/XGB_BNN_series_hybrid_model)
        model_folder = f"XGB_BNN_{args.model_type}_hybrid_model"
        output_dir = os.path.join(args.mode, model_folder)  # 结构为: ./{mode}/XGB_BNN_{model_type}_hybrid_model
        logger.info(f"Using model-specific output directory: {output_dir}")
    else:
        # 通用情况保持不变
        output_dir = args.output_dir or args.mode
        logger.info(f"Using general output directory: {output_dir}")

    # 第一步：合并CSV文件
    logger.info("\n" + "=" * 50)
    logger.info("Step 1: Consolidating CSV files")
    # 关键修改点：根据上面设置的 output_dir 来调用 consolidate_csv
    # 此时，如果指定了 model_type，output_dir 就是 './{mode}/XGB_BNN_{model_type}_hybrid_model'
    # 这正是我们要从中读取 pareto_solution 的目录。
    if args.model_type:
        # input_dir 应设置为 output_dir，因为它已经包含了我们需要查找的 pareto_solution 子目录
        consolidate_csv(output_dir, output_dir, args.model_type) # input_dir 和 output_dir 相同
    else:
        # 对于通用情况，input_dir 和 output_dir 都是 output_dir（根目录）
        consolidate_csv(output_dir, output_dir, args.model_type)

    # 第二步：评估
    logger.info("\n" + "=" * 50)
    logger.info("Step 2: Performing physical plausibility evaluation")
    evaluate_main(
        LLM_finetuned_model_path=args.LLM_finetuned_model_path,
        LLM_base_model_path=args.LLM_base_model_path,
        output_dir=output_dir,  # 使用设置好的output_dir
        model_type=args.model_type
    )
