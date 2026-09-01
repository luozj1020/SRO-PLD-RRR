import joblib
import logging
import gc
import os
import sys
import argparse
import importlib.util


# Import model classes and utilities
from utils.model_optim_utils import TuningConfig, create_tuning_pipeline
from utils.model_utils import setup_seed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
setup_seed(42)

# Main execution block
if __name__ == "__main__":
    # 新增命令行参数解析
    parser = argparse.ArgumentParser(description='超参数调优脚本')
    parser.add_argument('--mode', type=str, required=True, choices=['tradition', 'LLM'],
                        help='模式选择: tradition 或 LLM')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['attention', 'series', 'uncertainty_1', 'uncertainty_2'],
                        help='模型类型: attention, series, uncertainty_1 或 uncertainty_2')
    args = parser.parse_args()

    # 动态导入模型
    model_path = f"./{args.mode}/XGB_BNN_{args.model_type}_hybrid_model"
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), model_path))
    from model import HybridModel, ModelConfig

    # Configuration parameters
    # 只需要一个数据文件 - 原模型会自动处理特征
    DATA_PATH = './data/converted_file.xlsx'
    BASE_FEATURES = ['Oxygen pressure', 'Laser energy density', 'Temperature', 'Frequency', 'Thickness']
    TARGET_COLUMN = 'rrr'
    STO_COLUMN = 'Substrate'
    SEQUENCE_GOOD_PATH = './data/extracted_conditions_good.csv' if args.mode == 'LLM' else ''
    SEQUENCE_BAD_PATH = './data/extracted_conditions_bad.csv' if args.mode == 'LLM' else ''

    # Weight configuration for samples
    WEIGHT_CONFIG = {
        'weight_condition': {
            "column": "Substrate",
            "conditions": [
                ("STO", 1.0),
                ("Others", None)  # 将在优化过程中调整
            ]
        }
    }

    # Boundary weight configuration
    BOUNDARY_CONFIG = {
        'gaussian_sigma': 0.8,
        'missing_penalty_rate': 0.95,
        'penalty_sharpness': 1.5,
        'use_mahalanobis': True
    }

    # Hyperparameter tuning configuration
    tuning_config = TuningConfig()
    tuning_config.n_trials = 2048  # 可根据需要调整
    tuning_config.cv_folds = 3
    tuning_config.primary_metric = 'r2'
    tuning_config.direction = 'maximize'
    tuning_config.parallel_backend = 'thread'
    tuning_config.multi_objective_type = 'loss' if args.mode == 'LLM' else 'cv'
    tuning_config.timeout = None  # 可设置超时时间(秒)

    # 修改输出目录，根据命令行参数动态生成
    base_output_dir = f"./{args.mode}/XGB_BNN_{args.model_type}_hybrid_model/pretrain/hyperparameter_tuning_results"

    # 确保输出目录存在
    os.makedirs(base_output_dir, exist_ok=True)

    # Initialize tuning pipeline
    run_tuning = create_tuning_pipeline()

    try:
        logger.info("=" * 60)
        logger.info("开始超参数调优")
        logger.info("=" * 60)
        logger.info(f"模式: {args.mode}")
        logger.info(f"模型类型: {args.model_type}")
        logger.info(f"数据路径: {DATA_PATH}")
        logger.info(f"特征列: {BASE_FEATURES}")
        logger.info(f"目标列: {TARGET_COLUMN}")
        logger.info(f"STO标识列: {STO_COLUMN}")
        logger.info(f"试验次数: {tuning_config.n_trials}")
        logger.info(f"交叉验证折数: {tuning_config.cv_folds}")
        logger.info(f"优化类型: {tuning_config.multi_objective_type}")
        logger.info(f"输出目录: {base_output_dir}")
        logger.info("=" * 60)

        # Execute hyperparameter tuning
        results = run_tuning(
            data_path=DATA_PATH,
            base_features=BASE_FEATURES,
            target_column=TARGET_COLUMN,
            sto_column=STO_COLUMN,
            weight_config=WEIGHT_CONFIG,
            tuning_config=tuning_config,
            output_dir=base_output_dir,
            visualization_output_dir=base_output_dir,
            model_class=HybridModel,
            config_class=ModelConfig,
            use_boundary_weights=True,
            boundary_config=BOUNDARY_CONFIG,
            sequence_good_path=SEQUENCE_GOOD_PATH,
            sequence_bad_path=SEQUENCE_BAD_PATH,
            verbose=True,
            show_progress=True,
        )

        # Print completion message
        print("\n" + "=" * 60)
        print("超参数调优完成!")
        print("=" * 60)

        # Display Pareto frontier information
        tuning_results = results['tuning_results']

        # Print summary statistics
        print("\n" + "=" * 60)
        print("调优统计:")
        print(f"  总试验次数: {tuning_results['n_trials']}")
        print(f"  调优耗时: {tuning_results['tuning_time']:.2f} 秒")
        print(f"  平均每次试验: {tuning_results['tuning_time'] / max(tuning_results['n_trials'], 1):.2f} 秒")

        # Output directory instructions
        print("\n结果文件:")
        print(f"  主目录: {base_output_dir}")
        print(f"  可视化: {base_output_dir}/optimization_history.png")
        print(f"  所有试验: {base_output_dir}/all_trials_results.csv")
        print("=" * 60)

    except Exception as e:
        logger.error(f"调优失败: {str(e)}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        # Clean up resources
        logger.info("清理资源...")
        gc.collect()
        logger.info("完成!")
