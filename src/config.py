"""
Configuration module for Student Performance Prediction System.
Contains all configurable parameters for the ML pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    dataset_id: int = 697
    test_size: float = 0.2
    random_state: int = 42
    target_column: str = "Target"
    
    # Class labels mapping
    class_labels: Dict[str, int] = field(default_factory=lambda: {
        "Dropout": 0,
        "Enrolled": 1,
        "Graduate": 2
    })
    
    # Reverse mapping
    class_names: Dict[int, str] = field(default_factory=lambda: {
        0: "Dropout",
        1: "Enrolled",
        2: "Graduate"
    })


@dataclass
class SamplingConfig:
    """Configuration for data sampling/balancing techniques."""
    use_smote: bool = True
    use_adasyn: bool = False
    random_state: int = 42
    k_neighbors: int = 5


@dataclass
class ModelConfig:
    """Configuration for model training."""
    cv_folds: int = 5
    random_state: int = 42
    scoring_metric: str = "f1_macro"
    n_iter_random_search: int = 50
    
    # Standard models to train
    standard_models: List[str] = field(default_factory=lambda: [
        "logistic_regression",
        "svm",
        "decision_tree",
        "random_forest"
    ])
    
    # Boosting models to train
    boosting_models: List[str] = field(default_factory=lambda: [
        "gradient_boosting",
        "xgboost",
        "catboost",
        "logitboost"
    ])


@dataclass
class VisualizationConfig:
    """Configuration for visualization outputs."""
    figure_size: tuple = (12, 8)
    dpi: int = 100
    style: str = "seaborn-v0_8-whitegrid"
    color_palette: str = "Set2"
    save_figures: bool = True
    output_dir: str = "outputs"


@dataclass
class ProjectConfig:
    """Main project configuration combining all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.data.test_size <= 0 or self.data.test_size >= 1:
            raise ValueError("test_size must be between 0 and 1")
        if self.model.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")


# Default configuration instance
default_config = ProjectConfig()
