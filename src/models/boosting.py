"""
Boosting Models Module.
Implements boosting algorithms: Gradient Boosting, XGBoost, CatBoost, LogitBoost.
Based on the paper's findings that boosting methods outperform standard algorithms.
"""

import numpy as np
from typing import Dict, Any, List
import warnings

from .base import BaseModel, ModelRegistry


class GradientBoostingModel(BaseModel):
    """Gradient Boosting classifier using sklearn."""
    
    def __init__(self, random_state: int = 42, **kwargs):
        super().__init__("Gradient Boosting", random_state)
        self.kwargs = kwargs
        self._init_model()
    
    def _init_model(self) -> None:
        """Initialize the Gradient Boosting model."""
        from sklearn.ensemble import GradientBoostingClassifier
        
        self.model = GradientBoostingClassifier(
            random_state=self.random_state,
            **self.kwargs
        )
    
    def get_hyperparameter_grid(self) -> Dict[str, List[Any]]:
        """Get hyperparameter grid for tuning."""
        return {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0]
        }
    
    def get_feature_importances(self) -> np.ndarray:
        """Get feature importances."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first.")
        return self.model.feature_importances_


class XGBoostModel(BaseModel):
    """XGBoost (Extreme Gradient Boosting) classifier."""
    
    def __init__(self, random_state: int = 42, **kwargs):
        super().__init__("XGBoost", random_state)
        self.kwargs = kwargs
        self._init_model()
    
    def _init_model(self) -> None:
        """Initialize the XGBoost model."""
        try:
            from xgboost import XGBClassifier
            
            self.model = XGBClassifier(
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='mlogloss',
                n_jobs=-1,
                **self.kwargs
            )
        except ImportError:
            raise ImportError(
                "xgboost package is required. "
                "Install it with: pip install xgboost"
            )
    
    def get_hyperparameter_grid(self) -> Dict[str, List[Any]]:
        """Get hyperparameter grid for tuning."""
        return {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 10],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2, 0.3]
        }
    
    def get_feature_importances(self) -> np.ndarray:
        """Get feature importances."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first.")
        return self.model.feature_importances_


class CatBoostModel(BaseModel):
    """CatBoost classifier - handles categorical features natively."""
    
    def __init__(self, random_state: int = 42, verbose: bool = False, **kwargs):
        super().__init__("CatBoost", random_state)
        self.verbose = verbose
        self.kwargs = kwargs
        self._init_model()
    
    def _init_model(self) -> None:
        """Initialize the CatBoost model."""
        try:
            from catboost import CatBoostClassifier
            
            self.model = CatBoostClassifier(
                random_state=self.random_state,
                verbose=self.verbose,
                thread_count=-1,
                **self.kwargs
            )
        except ImportError:
            raise ImportError(
                "catboost package is required. "
                "Install it with: pip install catboost"
            )
    
    def get_hyperparameter_grid(self) -> Dict[str, List[Any]]:
        """Get hyperparameter grid for tuning."""
        return {
            'iterations': [100, 200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'depth': [4, 6, 8, 10],
            'l2_leaf_reg': [1, 3, 5, 7, 9],
            'border_count': [32, 64, 128],
            'bagging_temperature': [0, 0.5, 1]
        }
    
    def get_feature_importances(self) -> np.ndarray:
        """Get feature importances."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first.")
        return self.model.feature_importances_


class LogitBoostModel(BaseModel):
    """
    LogitBoost classifier.
    Implemented using sklearn's GradientBoostingClassifier with logistic loss.
    """
    
    def __init__(self, random_state: int = 42, **kwargs):
        super().__init__("LogitBoost", random_state)
        self.kwargs = kwargs
        self._init_model()
    
    def _init_model(self) -> None:
        """Initialize the LogitBoost model."""
        from sklearn.ensemble import GradientBoostingClassifier
        
        # LogitBoost is essentially Gradient Boosting with log loss
        self.model = GradientBoostingClassifier(
            random_state=self.random_state,
            loss='log_loss',  # This makes it equivalent to LogitBoost
            **self.kwargs
        )
    
    def get_hyperparameter_grid(self) -> Dict[str, List[Any]]:
        """Get hyperparameter grid for tuning."""
        return {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0]
        }
    
    def get_feature_importances(self) -> np.ndarray:
        """Get feature importances."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first.")
        return self.model.feature_importances_


# Register models
ModelRegistry.register("gradient_boosting", GradientBoostingModel)
ModelRegistry.register("xgboost", XGBoostModel)
ModelRegistry.register("catboost", CatBoostModel)
ModelRegistry.register("logitboost", LogitBoostModel)
