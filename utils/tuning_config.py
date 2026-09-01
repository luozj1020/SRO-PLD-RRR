from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class TuningConfig:
    """Configuration for optimized hyperparameter tuning."""

    # Output verbosity control
    verbose: bool = True
    show_progress: bool = True
    verbose_training: bool = False

    # Settings for initial sampling strategy
    sobol_sampling: bool = True
    sobol_trials: int = 20

    # Hyperparameter search space for XGBoost model
    xgb_search_space: Dict = field(default_factory=lambda: {
        'n_estimators': {'type': 'int', 'low': 10, 'high': 1000},
        'learning_rate': {'type': 'float', 'low': 1e-4, 'high': 1 - 1e-4, 'log': True},
        'reg_alpha': {'type': 'float', 'low': 0.0, 'high': 1.5},
        'reg_lambda': {'type': 'float', 'low': 0.0, 'high': 2.0},
        'min_child_weight': {'type': 'int', 'low': 1, 'high': 10}
    })

    # Hyperparameter search space for Bayesian Neural Network
    bnn_search_space: Dict = field(default_factory=lambda: {
        'first_hidden_dims_pow': {'type': 'int', 'low': 4, 'high': 10},
        'dropout_rates': {
            'type': 'float_list',
            'low': 0.01,
            'high': 0.5,
            'size': 3
        },
        'learning_rate': {'type': 'float', 'low': 1e-4, 'high': 0.5, 'log': True},
        #'weight_decay': {'type': 'float', 'low': 1e-5, 'high': 0.1, 'log': True},
    })

    # Search space for sample weighting parameters
    weight_search_space: Dict = field(default_factory=lambda: {
        'other_weight': {'type': 'float', 'low': 0.5, 'high': 1.0}
    })

    # Core tuning parameters
    n_trials: int = 100
    cv_folds: int = 3
    n_jobs: int = -1
    timeout: Optional[int] = None
    parallel_backend: str = 'thread'  # Options: 'thread' or 'process'

    # Optimization objective configuration
    primary_metric: str = 'r2'
    direction: str = 'maximize'

    # Early stopping criteria
    early_stopping_rounds: int = 15
    min_improvement: float = 0.001

    # Validation dataset configuration
    validation_split_ratio: float = 0.2
    validation_split_random_state: int = 42

    # Memory management settings
    memory_limit_gb: float = 100
    cleanup_frequency: int = 10

    multi_objective_type: str = 'full_cv'
