"""
Main Pipeline Module.
Orchestrates the complete ML pipeline for student performance prediction.
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import ProjectConfig, default_config
from src.data import (
    UCIDataLoader,
    DataValidator,
    FeaturePreprocessor,
    LabelEncoderWrapper,
    DataSplitter,
    DataPipeline,
    SamplingManager
)
from src.models import (
    ModelFactory,
    ModelTrainer,
    ModelEvaluator,
    ExperimentRunner
)
from src.visualization import (
    VisualizationConfig,
    DataVisualizer,
    ModelVisualizer,
    ReportGenerator
)
from src.utils import (
    setup_logging,
    save_model,
    save_results,
    create_experiment_directory,
    print_classification_report,
    Timer
)


class StudentPerformancePipeline:
    """
    Main pipeline class that orchestrates all components.
    Provides a high-level API for the complete ML workflow.
    """
    
    def __init__(
        self,
        config: Optional[ProjectConfig] = None,
        output_dir: str = "outputs"
    ):
        """
        Initialize the pipeline.
        
        Args:
            config: Project configuration.
            output_dir: Directory for outputs.
        """
        self.config = config or default_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_loader = UCIDataLoader(self.config.data.dataset_id)
        self.data_validator = DataValidator()
        self.sampling_manager = SamplingManager(self.config.sampling.random_state)
        
        # Initialize visualization
        viz_config = VisualizationConfig(
            figsize=self.config.visualization.figure_size,
            dpi=self.config.visualization.dpi,
            save_path=str(self.output_dir)
        )
        self.data_visualizer = DataVisualizer(viz_config)
        self.model_visualizer = ModelVisualizer(viz_config)
        self.report_generator = ReportGenerator(save_path=str(self.output_dir))
        
        # Data storage
        self.X = None
        self.y = None
        self.processed_data = None
        self.X_train_sampled = None
        self.y_train_sampled = None
        
        # Results storage
        self.comparison_df = None
        self.evaluation_results = None
        self.best_model = None
        
        # Class names mapping
        self.class_names = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
        self.class_names_list = ["Dropout", "Enrolled", "Graduate"]
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load the dataset from UCI repository.
        
        Returns:
            Tuple of (features, target).
        """
        print("\n" + "="*60)
        print("LOADING DATA")
        print("="*60)
        
        with Timer("Data loading"):
            self.X, self.y = self.data_loader.load()
        
        # Get metadata
        metadata = self.data_loader.get_metadata()
        print(f"\nDataset: {metadata.get('name', 'Unknown')}")
        print(f"Samples: {metadata['num_instances']}")
        print(f"Features: {metadata['num_features']}")
        print(f"Classes: {metadata['target_classes']}")
        
        return self.X, self.y
    
    def validate_data(self) -> Dict[str, Any]:
        """
        Validate the loaded data.
        
        Returns:
            Validation results dictionary.
        """
        print("\n" + "="*60)
        print("DATA VALIDATION")
        print("="*60)
        
        if self.X is None:
            raise RuntimeError("Data must be loaded first. Call load_data().")
        
        validation = self.data_validator.validate(self.X, self.y)
        
        print(f"\nTotal samples: {validation['num_samples']}")
        print(f"Total features: {validation['num_features']}")
        print(f"Missing values: {validation['total_missing']}")
        print(f"Duplicate rows: {validation['duplicates']}")
        print(f"\nTarget distribution:")
        for cls, count in validation['target_distribution'].items():
            pct = count / validation['num_samples'] * 100
            print(f"  {cls}: {count} ({pct:.1f}%)")
        
        return validation
    
    def preprocess_data(
        self,
        scaling: str = "standard"
    ) -> Dict[str, Any]:
        """
        Preprocess the data for model training.
        
        Args:
            scaling: Type of scaling to apply.
            
        Returns:
            Dictionary with processed data.
        """
        print("\n" + "="*60)
        print("DATA PREPROCESSING")
        print("="*60)
        
        if self.X is None:
            raise RuntimeError("Data must be loaded first. Call load_data().")
        
        with Timer("Preprocessing"):
            # Create pipeline components
            preprocessor = FeaturePreprocessor(scaling=scaling)
            label_encoder = LabelEncoderWrapper()
            splitter = DataSplitter(
                test_size=self.config.data.test_size,
                random_state=self.config.data.random_state
            )
            
            # Create and run pipeline
            pipeline = DataPipeline(preprocessor, label_encoder, splitter)
            self.processed_data = pipeline.run(self.X, self.y)
        
        print(f"\nTraining samples: {len(self.processed_data['X_train'])}")
        print(f"Test samples: {len(self.processed_data['X_test'])}")
        print(f"Features: {len(self.processed_data['feature_names'])}")
        print(f"Classes: {list(self.processed_data['classes'])}")
        
        return self.processed_data
    
    def apply_sampling(
        self,
        sampler_type: str = "smote"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply sampling to handle class imbalance.
        
        Args:
            sampler_type: Type of sampler to use.
            
        Returns:
            Tuple of (resampled X, resampled y).
        """
        print("\n" + "="*60)
        print(f"APPLYING {sampler_type.upper()} SAMPLING")
        print("="*60)
        
        if self.processed_data is None:
            raise RuntimeError("Data must be preprocessed first. Call preprocess_data().")
        
        X_train = self.processed_data['X_train_scaled'].values
        y_train = self.processed_data['y_train'].values
        
        print(f"\nOriginal training set size: {len(y_train)}")
        print("Original class distribution:")
        for cls, name in self.class_names.items():
            count = np.sum(y_train == cls)
            print(f"  {name}: {count} ({count/len(y_train)*100:.1f}%)")
        
        with Timer("Sampling"):
            self.X_train_sampled, self.y_train_sampled = \
                self.sampling_manager.apply_sampling(X_train, y_train, sampler_type)
        
        print(f"\nResampled training set size: {len(self.y_train_sampled)}")
        print("Resampled class distribution:")
        for cls, name in self.class_names.items():
            count = np.sum(self.y_train_sampled == cls)
            print(f"  {name}: {count} ({count/len(self.y_train_sampled)*100:.1f}%)")
        
        return self.X_train_sampled, self.y_train_sampled
    
    def train_standard_models(
        self,
        tune_hyperparameters: bool = False
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Train all standard models.
        
        Args:
            tune_hyperparameters: Whether to tune hyperparameters.
            
        Returns:
            Tuple of (comparison DataFrame, detailed results).
        """
        print("\n" + "="*60)
        print("TRAINING STANDARD MODELS")
        print("="*60)
        
        models = ModelFactory.create_all_standard(self.config.model.random_state)
        return self._train_models(models, tune_hyperparameters)
    
    def train_boosting_models(
        self,
        tune_hyperparameters: bool = False
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Train all boosting models.
        
        Args:
            tune_hyperparameters: Whether to tune hyperparameters.
            
        Returns:
            Tuple of (comparison DataFrame, detailed results).
        """
        print("\n" + "="*60)
        print("TRAINING BOOSTING MODELS")
        print("="*60)
        
        models = ModelFactory.create_all_boosting(self.config.model.random_state)
        return self._train_models(models, tune_hyperparameters)
    
    def train_all_models(
        self,
        tune_hyperparameters: bool = False
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Train all models (standard + boosting).
        
        Args:
            tune_hyperparameters: Whether to tune hyperparameters.
            
        Returns:
            Tuple of (comparison DataFrame, detailed results).
        """
        print("\n" + "="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        
        models = {}
        models.update(ModelFactory.create_all_standard(self.config.model.random_state))
        models.update(ModelFactory.create_all_boosting(self.config.model.random_state))
        
        return self._train_models(models, tune_hyperparameters)
    
    def _train_models(
        self,
        models: Dict[str, Any],
        tune_hyperparameters: bool
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Internal method to train models."""
        if self.X_train_sampled is None:
            raise RuntimeError("Sampling must be applied first. Call apply_sampling().")
        
        # Setup trainer and evaluator
        trainer = ModelTrainer(
            cv_folds=self.config.model.cv_folds,
            random_state=self.config.model.random_state,
            scoring=self.config.model.scoring_metric,
            n_iter=self.config.model.n_iter_random_search
        )
        evaluator = ModelEvaluator(self.class_names)
        
        # Setup experiment runner
        runner = ExperimentRunner(trainer, evaluator, self.class_names)
        
        # Get test data
        X_test = self.processed_data['X_test_scaled'].values
        y_test = self.processed_data['y_test'].values
        
        with Timer("Model training"):
            self.comparison_df, self.evaluation_results = runner.run_experiment(
                self.X_train_sampled,
                self.y_train_sampled,
                X_test,
                y_test,
                models,
                tune_hyperparameters
            )
        
        # Get best model
        best_name, self.best_model = runner.get_best_model()
        print(f"\nBest model: {best_name}")
        
        return self.comparison_df, self.evaluation_results
    
    def visualize_data(self) -> None:
        """Generate data exploration visualizations."""
        print("\n" + "="*60)
        print("GENERATING DATA VISUALIZATIONS")
        print("="*60)
        
        if self.X is None:
            raise RuntimeError("Data must be loaded first.")
        
        with Timer("Visualization"):
            # Class distribution
            self.data_visualizer.plot_class_distribution(
                self.y, 
                {v: k for k, v in zip(self.class_names_list, self.y.unique())},
                save_name="class_distribution.png"
            )
            print("  - Class distribution saved")
            
            # Correlation matrix
            self.data_visualizer.plot_correlation_matrix(
                self.X,
                save_name="correlation_matrix.png"
            )
            print("  - Correlation matrix saved")
    
    def visualize_results(self) -> None:
        """Generate model evaluation visualizations."""
        print("\n" + "="*60)
        print("GENERATING RESULTS VISUALIZATIONS")
        print("="*60)
        
        if self.comparison_df is None:
            raise RuntimeError("Models must be trained first.")
        
        with Timer("Visualization"):
            # Model comparison
            self.model_visualizer.plot_model_comparison(
                self.comparison_df,
                'F1 (Macro)',
                save_name="model_comparison.png"
            )
            print("  - Model comparison saved")
            
            # Per-class metrics
            self.model_visualizer.plot_per_class_metrics(
                self.evaluation_results,
                'f1_score',
                save_name="per_class_f1.png"
            )
            print("  - Per-class F1 scores saved")
            
            # Confusion matrices
            for model_name, results in self.evaluation_results.items():
                safe_name = model_name.replace(' ', '_').lower()
                self.model_visualizer.plot_confusion_matrix(
                    results['confusion_matrix'],
                    self.class_names_list,
                    title=f"Confusion Matrix - {model_name}",
                    save_name=f"cm_{safe_name}.png"
                )
            print("  - Confusion matrices saved")
    
    def generate_report(self) -> None:
        """Generate a summary report."""
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        
        if self.comparison_df is None:
            raise RuntimeError("Models must be trained first.")
        
        print("\nModel Comparison:")
        print(self.comparison_df.to_string(index=False))
        
        # Save comparison to CSV
        self.comparison_df.to_csv(self.output_dir / "model_comparison.csv", index=False)
        print(f"\nResults saved to {self.output_dir / 'model_comparison.csv'}")
        
        # Print classification report for best model
        best_name = self.comparison_df.iloc[0]['Model']
        print_classification_report(
            self.evaluation_results[best_name]['classification_report'],
            f"Best Model: {best_name}"
        )
    
    def run_full_pipeline(
        self,
        sampler_type: str = "smote",
        tune_hyperparameters: bool = False,
        train_boosting: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline from data loading to evaluation.
        
        Args:
            sampler_type: Type of sampling to use.
            tune_hyperparameters: Whether to tune hyperparameters.
            train_boosting: Whether to include boosting models.
            
        Returns:
            Dictionary with all results.
        """
        print("\n" + "="*60)
        print("STUDENT PERFORMANCE PREDICTION PIPELINE")
        print("="*60)
        
        with Timer("Full pipeline"):
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Validate data
            self.validate_data()
            
            # Step 3: Preprocess data
            self.preprocess_data()
            
            # Step 4: Apply sampling
            self.apply_sampling(sampler_type)
            
            # Step 5: Visualize data
            self.visualize_data()
            
            # Step 6: Train models
            if train_boosting:
                self.train_all_models(tune_hyperparameters)
            else:
                self.train_standard_models(tune_hyperparameters)
            
            # Step 7: Visualize results
            self.visualize_results()
            
            # Step 8: Generate report
            self.generate_report()
        
        return {
            'comparison': self.comparison_df,
            'results': self.evaluation_results,
            'best_model': self.best_model
        }


def main():
    """Main entry point."""
    warnings.filterwarnings('ignore')
    
    # Create pipeline
    pipeline = StudentPerformancePipeline(output_dir="outputs")
    
    # Run full pipeline
    results = pipeline.run_full_pipeline(
        sampler_type="smote",
        tune_hyperparameters=False,
        train_boosting=True
    )
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    
    return results


if __name__ == "__main__":
    main()
