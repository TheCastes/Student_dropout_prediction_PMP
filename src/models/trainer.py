"""
Model Trainer Module.
Handles model training, hyperparameter tuning, and cross-validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)
import warnings

from .base import ModelInterface, ModelFactory


class ModelTrainer:
    """
    Handles training of individual models with optional hyperparameter tuning.
    """
    
    def __init__(
        self,
        cv_folds: int = 5,
        random_state: int = 42,
        scoring: str = 'f1_macro',
        n_iter: int = 50
    ):
        """
        Initialize the model trainer.
        
        Args:
            cv_folds: Number of cross-validation folds.
            random_state: Random seed for reproducibility.
            scoring: Metric for optimization.
            n_iter: Number of iterations for random search.
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.scoring = scoring
        self.n_iter = n_iter
        
        self.cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=random_state
        )
    
    def train(
        self,
        model: ModelInterface,
        X_train: np.ndarray,
        y_train: np.ndarray,
        tune_hyperparameters: bool = False,
        search_method: str = 'random'
    ) -> Dict[str, Any]:
        """
        Train a model with optional hyperparameter tuning.
        
        Args:
            model: Model to train.
            X_train: Training features.
            y_train: Training labels.
            tune_hyperparameters: Whether to tune hyperparameters.
            search_method: 'grid' or 'random' search.
            
        Returns:
            Dictionary with training results.
        """
        if tune_hyperparameters:
            return self._train_with_tuning(model, X_train, y_train, search_method)
        else:
            return self._train_simple(model, X_train, y_train)
    
    def _train_simple(
        self,
        model: ModelInterface,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Dict[str, Any]:
        """Train model without hyperparameter tuning."""
        # Cross-validation
        cv_results = model.cross_validate(
            X_train, y_train,
            cv=self.cv_folds,
            scoring=self.scoring
        )
        
        # Fit on full training data
        model.fit(X_train, y_train)
        
        return {
            'model': model,
            'cv_mean_score': cv_results['mean_score'],
            'cv_std_score': cv_results['std_score'],
            'cv_all_scores': cv_results['all_scores'],
            'best_params': model.get_params(),
            'tuned': False
        }
    
    def _train_with_tuning(
        self,
        model: ModelInterface,
        X_train: np.ndarray,
        y_train: np.ndarray,
        search_method: str
    ) -> Dict[str, Any]:
        """Train model with hyperparameter tuning."""
        param_grid = model.get_hyperparameter_grid()
        
        if search_method == 'grid':
            search = GridSearchCV(
                model.model,
                param_grid,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=-1,
                verbose=0
            )
        else:
            search = RandomizedSearchCV(
                model.model,
                param_grid,
                n_iter=self.n_iter,
                cv=self.cv,
                scoring=self.scoring,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=0
            )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X_train, y_train)
        
        # Update model with best parameters
        model.model = search.best_estimator_
        model._is_fitted = True
        
        return {
            'model': model,
            'cv_mean_score': search.best_score_,
            'cv_std_score': search.cv_results_['std_test_score'][search.best_index_],
            'cv_all_scores': None,
            'best_params': search.best_params_,
            'tuned': True
        }


class ModelEvaluator:
    """
    Evaluates trained models on test data.
    Computes comprehensive metrics including per-class metrics.
    """
    
    def __init__(self, class_names: Optional[Dict[int, str]] = None):
        """
        Initialize the evaluator.
        
        Args:
            class_names: Mapping of class indices to names.
        """
        self.class_names = class_names or {0: "Class 0", 1: "Class 1", 2: "Class 2"}
    
    def evaluate(
        self,
        model: ModelInterface,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate a model on test data.
        
        Args:
            model: Trained model.
            X_test: Test features.
            y_test: Test labels.
            
        Returns:
            Dictionary with evaluation metrics.
        """
        y_pred = model.predict(X_test)
        
        # Overall metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # Per-class metrics
        f1_per_class = f1_score(y_test, y_pred, average=None)
        precision_per_class = precision_score(y_test, y_pred, average=None)
        recall_per_class = recall_score(y_test, y_pred, average=None)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        report = classification_report(
            y_test, y_pred,
            target_names=list(self.class_names.values()),
            output_dict=True
        )
        
        # Build per-class metrics dictionary
        per_class_metrics = {}
        for idx, class_name in self.class_names.items():
            if idx < len(f1_per_class):
                per_class_metrics[class_name] = {
                    'f1_score': f1_per_class[idx],
                    'precision': precision_per_class[idx],
                    'recall': recall_per_class[idx]
                }
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': y_pred
        }
    
    def compare_models(
        self,
        results: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Create a comparison table of model results.
        
        Args:
            results: Dictionary of model names to evaluation results.
            
        Returns:
            DataFrame with comparison.
        """
        comparison_data = []
        
        for model_name, result in results.items():
            row = {
                'Model': model_name,
                'Accuracy': result['accuracy'],
                'F1 (Macro)': result['f1_macro'],
                'F1 (Weighted)': result['f1_weighted']
            }
            
            # Add per-class F1 scores
            for class_name, metrics in result['per_class_metrics'].items():
                row[f'F1 {class_name}'] = metrics['f1_score']
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('F1 (Macro)', ascending=False)
        return df


class ExperimentRunner:
    """
    Orchestrates the complete training and evaluation experiment.
    Runs multiple models and compares results.
    """
    
    def __init__(
        self,
        trainer: Optional[ModelTrainer] = None,
        evaluator: Optional[ModelEvaluator] = None,
        class_names: Optional[Dict[int, str]] = None
    ):
        """
        Initialize the experiment runner.
        
        Args:
            trainer: ModelTrainer instance.
            evaluator: ModelEvaluator instance.
            class_names: Mapping of class indices to names.
        """
        self.trainer = trainer or ModelTrainer()
        self.evaluator = evaluator or ModelEvaluator(class_names)
        self.class_names = class_names
        
        self.training_results = {}
        self.evaluation_results = {}
    
    def run_experiment(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        models: Dict[str, ModelInterface],
        tune_hyperparameters: bool = False
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run complete experiment with multiple models.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            X_test: Test features.
            y_test: Test labels.
            models: Dictionary of model names to model instances.
            tune_hyperparameters: Whether to tune hyperparameters.
            
        Returns:
            Tuple of (comparison DataFrame, detailed results dictionary).
        """
        print(f"Running experiment with {len(models)} models...")
        print(f"Hyperparameter tuning: {'enabled' if tune_hyperparameters else 'disabled'}")
        print("-" * 50)
        
        for model_name, model in models.items():
            print(f"\nTraining: {model_name}")
            
            try:
                # Train
                train_result = self.trainer.train(
                    model, X_train, y_train,
                    tune_hyperparameters=tune_hyperparameters
                )
                self.training_results[model_name] = train_result
                
                print(f"  CV Score: {train_result['cv_mean_score']:.4f} "
                      f"(+/- {train_result['cv_std_score']:.4f})")
                
                # Evaluate
                eval_result = self.evaluator.evaluate(
                    train_result['model'], X_test, y_test
                )
                self.evaluation_results[model_name] = eval_result
                
                print(f"  Test Accuracy: {eval_result['accuracy']:.4f}")
                print(f"  Test F1 (Macro): {eval_result['f1_macro']:.4f}")
                
            except Exception as e:
                print(f"  Error: {str(e)}")
                continue
        
        print("\n" + "=" * 50)
        
        # Create comparison table
        comparison = self.evaluator.compare_models(self.evaluation_results)
        
        return comparison, self.evaluation_results
    
    def get_best_model(self) -> Tuple[str, ModelInterface]:
        """
        Get the best performing model based on F1 macro score.
        
        Returns:
            Tuple of (model name, model instance).
        """
        if not self.evaluation_results:
            raise RuntimeError("No experiments have been run yet.")
        
        best_name = max(
            self.evaluation_results.keys(),
            key=lambda k: self.evaluation_results[k]['f1_macro']
        )
        
        return best_name, self.training_results[best_name]['model']
