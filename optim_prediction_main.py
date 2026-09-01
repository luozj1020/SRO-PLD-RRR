#!/usr/bin/env python3
"""
Main Pipeline: Pareto Frontier Calculation and Bayesian Optimization

This is the main entry point that orchestrates:
1. Combining metrics from pretraining and finetuning
2. Calculating Pareto frontier with optional grid search filtering
3. Running Bayesian optimization on selected Pareto solutions

Usage:
    python optim_prediction_main.py --mode tradition --model_type series \
        --secondary_direction max --min_sol 10 --max_sol 20 \
        --optim_mode fix_freq --fixed_frequency 4.0 --use_gpu --enable_grid_filter
"""

import sys
import os
import argparse

# Import the two main modules
from calculate_pareto_solution import process_trial_data, analyze_pareto_and_copy_models
from bayesian_optimization import run_optimization_on_pareto_solutions

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))


def main():
    """Main execution function orchestrating the pipeline."""
    print("\n" + "="*70)
    print("🤖 Integrated Pipeline: Pareto Frontier → Bayesian Optimization")
    print("="*70)

    parser = argparse.ArgumentParser(description='Execute the combined pipeline.')

    # Basic parameters
    parser.add_argument('--mode', type=str, required=True, choices=['tradition', 'LLM'],
                        help='Mode: tradition or LLM')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['attention', 'series', 'uncertainty_1', 'uncertainty_2'],
                        help='Model type')

    # Pareto frontier parameters
    parser.add_argument('--secondary_direction', type=str, default='max', choices=['max', 'min'],
                        help='Direction for secondary_score (default: max)')
    parser.add_argument('--max_sol', type=int, default=20,
                        help='Maximum number of Pareto solutions (default: 20)')
    parser.add_argument('--no_viz', action='store_true',
                        help='Skip creating visualizations')

    # Grid search filtering parameters
    parser.add_argument('--enable_grid_filter', action='store_true', default=True,
                        help='Enable grid search filtering (default: True)')
    parser.add_argument('--disable_grid_filter', action='store_true',
                        help='Disable grid search filtering')
    parser.add_argument('--eta', type=float, default=1.0,
                        help='Threshold for grid search filtering (default: 1.0)')
    parser.add_argument('--grid_points_per_dim', type=int, default=10,
                        help='Number of grid points per dimension (default: 10)')
    parser.add_argument('--distance_threshold', type=float, default=1.0,
                        help='Distance threshold for "far from S" (default: 0.8)')

    # Bayesian optimization parameters
    parser.add_argument('--optim_mode', type=str,
                        choices=['optim_all', 'fix_freq', 'fix_thick', 'fix_freq_thick'],
                        help='Optimization mode: optim_all (5 vars) | fix_freq (fixed f, optimize 4) | '
                             'fix_thick (thickness uncertainty marginalized, optimize 4) | '
                             'fix_freq_thick (fixed f + thickness uncertainty marginalized, optimize 3)')
    parser.add_argument('--fixed_frequency', type=float,
                        help='Fixed frequency value for fix_freq / fix_freq_thick modes (default: 5.0)')
    parser.add_argument('--thickness_mean', type=float,
                        help='Mean thickness value for fix_thick / fix_freq_thick modes (default: 20.0)')
    parser.add_argument('--thickness_std', type=float,
                        help='Std thickness value for fix_thick / fix_freq_thick modes (default: 10.0)')
    parser.add_argument('--n_thickness_samples', type=int,
                        help='Number of thickness samples for fix_thick / fix_freq_thick modes (default: 20)')

    # GPU acceleration
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU for optimization and grid search')

    args = parser.parse_args()

    mode = args.mode
    model_type = args.model_type
    base_path = f"./{mode}/XGB_BNN_{model_type}_hybrid_model"

    # Step 1: Combine metrics
    print(f"\n--- Step 1: Combining Metrics for {mode}/{model_type} ---")
    if not process_trial_data(mode, model_type):
        print("❌ Step 1 failed. Stopping pipeline.")
        return

    # Step 2: Calculate Pareto frontier
    print(f"\n--- Step 2: Calculating Pareto Frontier for {mode}/{model_type} ---")
    enable_grid_filter = args.enable_grid_filter and not args.disable_grid_filter

    selected_pareto_df = analyze_pareto_and_copy_models(
        mode=mode,
        model_type=model_type,
        secondary_direction=args.secondary_direction,
        max_sol=args.max_sol,
        create_viz=not args.no_viz,
        enable_grid_filter=enable_grid_filter,
        eta=args.eta,
        grid_points_per_dim=args.grid_points_per_dim,
        distance_threshold=args.distance_threshold,
        use_gpu=args.use_gpu
    )

    if selected_pareto_df is None:
        print("❌ Step 2 failed. Stopping pipeline.")
        return

    # Step 3: Bayesian optimization (optional)
    if args.optim_mode:
        print(f"\n--- Step 3: Optimizing Pareto Solutions (Mode: {args.optim_mode}) ---")

        # Import HybridModel
        model_path = f"./{mode}/XGB_BNN_{model_type}_hybrid_model"
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), model_path))

        try:
            from model import HybridModel
            print(f"✓ Successfully imported HybridModel from {model_path}/model.py")
        except ImportError as e:
            print(f"❌ Could not import HybridModel from '{model_path}/model.py'")
            print(f"   Error: {e}")
            print("⚠ Skipping Step 3 due to missing dependency.")
            return

        base_output_dir = os.path.join(base_path, "pareto_solution")
        run_optimization_on_pareto_solutions(
            mode=mode,
            model_type=model_type,
            base_output_dir=base_output_dir,
            HybridModel=HybridModel,
            optim_mode=args.optim_mode,
            fixed_frequency=args.fixed_frequency,
            thickness_mean=args.thickness_mean,
            thickness_std=args.thickness_std,
            n_thickness_samples=args.n_thickness_samples,
            use_gpu=args.use_gpu
        )
    else:
        print("\n⚠ No optimization mode specified. Skipping Step 3.")

    print("\n" + "="*70)
    print("✅ Pipeline completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
