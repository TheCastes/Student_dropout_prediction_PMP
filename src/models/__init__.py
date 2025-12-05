"""
Models module for Student Performance Prediction System.
Contains all ML model implementations and training utilities.
"""

from .base import (
    ModelInterface,
    BaseModel,
    ModelRegistry,
    ModelFactory
)
from .standard import (
    LogisticRegressionModel,
    SVMModel,
    DecisionTreeModel,
    RandomForestModel
)
from .boosting import (
    GradientBoostingModel,
    XGBoostModel,
    CatBoostModel,
    LogitBoostModel
)
from .trainer import (
    ModelTrainer,
    ModelEvaluator,
    ExperimentRunner
)

__all__ = [
    # Base
    "ModelInterface",
    "BaseModel",
    "ModelRegistry",
    "ModelFactory",
    # Standard models
    "LogisticRegressionModel",
    "SVMModel",
    "DecisionTreeModel",
    "RandomForestModel",
    # Boosting models
    "GradientBoostingModel",
    "XGBoostModel",
    "CatBoostModel",
    "LogitBoostModel",
    # Training utilities
    "ModelTrainer",
    "ModelEvaluator",
    "ExperimentRunner"
]
