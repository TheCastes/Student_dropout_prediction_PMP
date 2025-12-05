"""
Student Performance Prediction System.
A modular ML system for predicting student academic performance.

Based on the paper:
"Early Prediction of student's Performance in Higher Education: A Case Study"
by Martins et al. (2021)
"""

from .config import ProjectConfig, default_config

__version__ = "1.0.0"
__author__ = "Based on Martins et al. (2021)"

__all__ = [
    "ProjectConfig",
    "default_config"
]
