"""
Utility Functions Module.
Contains helper functions for logging, file operations, and reporting.
"""

import logging
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np


def setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level.
        log_file: Optional file path for logging.
        
    Returns:
        Configured logger.
    """
    logger = logging.getLogger('student_prediction')
    logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def save_model(model: Any, filepath: str) -> None:
    """
    Save a trained model to disk.
    
    Args:
        model: Model to save.
        filepath: Path to save the model.
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model(filepath: str) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        filepath: Path to the saved model.
        
    Returns:
        Loaded model.
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """
    Save results dictionary to JSON.
    
    Args:
        results: Results dictionary.
        filepath: Path to save the results.
    """
    # Convert numpy arrays to lists for JSON serialization
    serializable = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        elif isinstance(value, dict):
            serializable[key] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in value.items()
            }
        else:
            serializable[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)


def load_results(filepath: str) -> Dict[str, Any]:
    """
    Load results from JSON file.
    
    Args:
        filepath: Path to the results file.
        
    Returns:
        Results dictionary.
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def generate_experiment_id() -> str:
    """Generate a unique experiment ID based on timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_experiment_directory(
    base_path: str = "experiments",
    experiment_id: Optional[str] = None
) -> Path:
    """
    Create a directory for experiment outputs.
    
    Args:
        base_path: Base path for experiments.
        experiment_id: Optional experiment ID (generated if not provided).
        
    Returns:
        Path to the experiment directory.
    """
    if experiment_id is None:
        experiment_id = generate_experiment_id()
    
    exp_path = Path(base_path) / experiment_id
    exp_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_path / "models").mkdir(exist_ok=True)
    (exp_path / "figures").mkdir(exist_ok=True)
    (exp_path / "reports").mkdir(exist_ok=True)
    
    return exp_path


def format_results_table(
    comparison_df: pd.DataFrame,
    float_format: str = ".4f"
) -> str:
    """
    Format results table for display.
    
    Args:
        comparison_df: Comparison DataFrame.
        float_format: Format string for floats.
        
    Returns:
        Formatted string representation.
    """
    return comparison_df.to_string(float_format=float_format)


def print_classification_report(
    report: Dict[str, Any],
    title: str = "Classification Report"
) -> None:
    """
    Print a formatted classification report.
    
    Args:
        report: Classification report dictionary.
        title: Title for the report.
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print('='*60)
    
    # Headers
    print(f"{'Class':<20} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>10}")
    print('-'*60)
    
    # Per-class metrics
    for key, value in report.items():
        if isinstance(value, dict) and 'precision' in value:
            print(f"{key:<20} {value['precision']:>12.4f} {value['recall']:>12.4f} "
                  f"{value['f1-score']:>12.4f} {value.get('support', 'N/A'):>10}")
    
    print('-'*60)
    
    # Overall metrics
    if 'accuracy' in report:
        print(f"{'Accuracy':<20} {'':<12} {'':<12} {report['accuracy']:>12.4f}")
    if 'macro avg' in report:
        macro = report['macro avg']
        print(f"{'Macro Avg':<20} {macro['precision']:>12.4f} {macro['recall']:>12.4f} "
              f"{macro['f1-score']:>12.4f}")
    if 'weighted avg' in report:
        weighted = report['weighted avg']
        print(f"{'Weighted Avg':<20} {weighted['precision']:>12.4f} {weighted['recall']:>12.4f} "
              f"{weighted['f1-score']:>12.4f}")
    
    print('='*60)


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, *args):
        self.end_time = datetime.now()
        self.elapsed = (self.end_time - self.start_time).total_seconds()
        print(f"{self.description} completed in {self.elapsed:.2f} seconds")


class ProgressTracker:
    """Simple progress tracker for long-running operations."""
    
    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.description = description
        self.current = 0
    
    def update(self, n: int = 1) -> None:
        """Update progress by n steps."""
        self.current += n
        percentage = (self.current / self.total) * 100
        print(f"\r{self.description}: {self.current}/{self.total} ({percentage:.1f}%)", end="")
        
        if self.current >= self.total:
            print()  # New line when complete
    
    def reset(self) -> None:
        """Reset the progress tracker."""
        self.current = 0
