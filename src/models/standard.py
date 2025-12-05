"""
Standard Models Module.
Implements standard ML algorithms: Logistic Regression, SVM, Decision Tree, Random Forest.
"""

import numpy as np
from typing import Dict, Any, List

from .base import BaseModel, ModelRegistry


class LogisticRegressionModel(BaseModel):
    """Logistic Regression classifier."""
    
    def __init__(self, random_state: int = 42, **kwargs):
        super().__init__("Logistic Regression", random_state)
        self.kwargs = kwargs
        self._init_model()
    
    def _init_model(self) -> None:
        """Initialize the Logistic Regression model."""
        from sklearn.linear_model import LogisticRegression
        
        self.model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            multi_class='multinomial',
            solver='lbfgs',
            **self.kwargs
        )
    
    def get_hyperparameter_grid(self) -> Dict[str, List[Any]]:
        """Get hyperparameter grid for tuning."""
        return {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'newton-cg', 'sag'],
            'max_iter': [1000, 2000]
        }


class SVMModel(BaseModel):
    """Support Vector Machine classifier."""
    
    def __init__(self, random_state: int = 42, **kwargs):
        super().__init__("SVM", random_state)
        self.kwargs = kwargs
        self._init_model()
    
    def _init_model(self) -> None:
        """Initialize the SVM model."""
        from sklearn.svm import SVC
        
        self.model = SVC(
            random_state=self.random_state,
            probability=True,
            **self.kwargs
        )
    
    def get_hyperparameter_grid(self) -> Dict[str, List[Any]]:
        """Get hyperparameter grid for tuning."""
        return {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear', 'poly'],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'degree': [2, 3, 4]  # Only for poly kernel
        }


class DecisionTreeModel(BaseModel):
    """Decision Tree classifier."""
    
    def __init__(self, random_state: int = 42, **kwargs):
        super().__init__("Decision Tree", random_state)
        self.kwargs = kwargs
        self._init_model()
    
    def _init_model(self) -> None:
        """Initialize the Decision Tree model."""
        from sklearn.tree import DecisionTreeClassifier
        
        self.model = DecisionTreeClassifier(
            random_state=self.random_state,
            **self.kwargs
        )
    
    def get_hyperparameter_grid(self) -> Dict[str, List[Any]]:
        """Get hyperparameter grid for tuning."""
        return {
            'max_depth': [None, 5, 10, 15, 20, 25],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random']
        }
    
    def get_feature_importances(self) -> np.ndarray:
        """Get feature importances from the fitted tree."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first.")
        return self.model.feature_importances_


class RandomForestModel(BaseModel):
    """Random Forest classifier."""
    
    def __init__(self, random_state: int = 42, **kwargs):
        super().__init__("Random Forest", random_state)
        self.kwargs = kwargs
        self._init_model()
    
    def _init_model(self) -> None:
        """Initialize the Random Forest model."""
        from sklearn.ensemble import RandomForestClassifier
        
        self.model = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=-1,
            **self.kwargs
        )
    
    def get_hyperparameter_grid(self) -> Dict[str, List[Any]]:
        """Get hyperparameter grid for tuning."""
        return {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
    
    def get_feature_importances(self) -> np.ndarray:
        """Get feature importances from the fitted forest."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first.")
        return self.model.feature_importances_


# Register models
ModelRegistry.register("logistic_regression", LogisticRegressionModel)
ModelRegistry.register("svm", SVMModel)
ModelRegistry.register("decision_tree", DecisionTreeModel)
ModelRegistry.register("random_forest", RandomForestModel)
