"""
Base Model Module.
Defines abstract interfaces and factory pattern for ML models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from abc import ABC, abstractmethod
from sklearn.model_selection import cross_val_score, StratifiedKFold


class ModelInterface(ABC):
    """Abstract interface for all machine learning models."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ModelInterface':
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the model name."""
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        pass
    
    @abstractmethod
    def set_params(self, **params) -> 'ModelInterface':
        """Set model parameters."""
        pass


class BaseModel(ModelInterface):
    """
    Base class for all models.
    Provides common functionality for model training and evaluation.
    """
    
    def __init__(self, model_name: str, random_state: int = 42):
        """
        Initialize the base model.
        
        Args:
            model_name: Name of the model.
            random_state: Random seed for reproducibility.
        """
        self.model_name = model_name
        self.random_state = random_state
        self.model = None
        self._is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        """
        Fit the model to training data.
        
        Args:
            X: Training features.
            y: Training labels.
            
        Returns:
            Self for method chaining.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call _init_model first.")
        
        self.model.fit(X, y)
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict.
            
        Returns:
            Predicted labels.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict.
            
        Returns:
            Class probabilities.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError(
                f"{self.model_name} does not support probability prediction."
            )
    
    def get_name(self) -> str:
        """Get the model name."""
        return self.model_name
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        if self.model is not None:
            return self.model.get_params()
        return {}
    
    def set_params(self, **params) -> 'BaseModel':
        """Set model parameters."""
        if self.model is not None:
            self.model.set_params(**params)
        return self
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: str = 'f1_macro'
    ) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Features.
            y: Labels.
            cv: Number of folds.
            scoring: Scoring metric.
            
        Returns:
            Dictionary with CV results.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized.")
        
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(self.model, X, y, cv=skf, scoring=scoring)
        
        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'all_scores': scores.tolist()
        }
    
    @abstractmethod
    def _init_model(self) -> None:
        """Initialize the underlying sklearn model."""
        pass
    
    @abstractmethod
    def get_hyperparameter_grid(self) -> Dict[str, List[Any]]:
        """Get hyperparameter grid for tuning."""
        pass


class ModelRegistry:
    """Registry for storing and retrieving model classes."""
    
    _models: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, model_class: type) -> None:
        """Register a model class."""
        cls._models[name.lower()] = model_class
    
    @classmethod
    def get(cls, name: str) -> type:
        """Get a registered model class."""
        name = name.lower()
        if name not in cls._models:
            raise ValueError(
                f"Unknown model: {name}. "
                f"Available models: {list(cls._models.keys())}"
            )
        return cls._models[name]
    
    @classmethod
    def available_models(cls) -> List[str]:
        """Get list of available models."""
        return list(cls._models.keys())


class ModelFactory:
    """Factory for creating model instances."""
    
    @staticmethod
    def create(
        model_type: str,
        random_state: int = 42,
        **kwargs
    ) -> ModelInterface:
        """
        Create a model instance.
        
        Args:
            model_type: Type of model to create.
            random_state: Random seed.
            **kwargs: Additional model arguments.
            
        Returns:
            Model instance.
        """
        model_class = ModelRegistry.get(model_type)
        return model_class(random_state=random_state, **kwargs)
    
    @staticmethod
    def create_all_standard(random_state: int = 42) -> Dict[str, ModelInterface]:
        """Create instances of all standard models."""
        from .standard import (
            LogisticRegressionModel,
            SVMModel,
            DecisionTreeModel,
            RandomForestModel
        )
        
        return {
            "Logistic Regression": LogisticRegressionModel(random_state=random_state),
            "SVM": SVMModel(random_state=random_state),
            "Decision Tree": DecisionTreeModel(random_state=random_state),
            "Random Forest": RandomForestModel(random_state=random_state)
        }
    
    @staticmethod
    def create_all_boosting(random_state: int = 42) -> Dict[str, ModelInterface]:
        """Create instances of all boosting models."""
        from .boosting import (
            GradientBoostingModel,
            XGBoostModel,
            CatBoostModel,
            LogitBoostModel
        )
        
        return {
            "Gradient Boosting": GradientBoostingModel(random_state=random_state),
            "XGBoost": XGBoostModel(random_state=random_state),
            "CatBoost": CatBoostModel(random_state=random_state),
            "LogitBoost": LogitBoostModel(random_state=random_state)
        }
