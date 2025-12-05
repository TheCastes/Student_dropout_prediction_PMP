"""
Data module for Student Performance Prediction System.
Contains data loading, preprocessing, and sampling utilities.
"""

from .loader import UCIDataLoader, DataValidator
from .preprocessor import (
    FeaturePreprocessor,
    LabelEncoderWrapper,
    DataSplitter,
    DataPipeline
)
from .sampling import (
    SamplerFactory,
    SamplingManager,
    SMOTESampler,
    ADASYNSampler,
    NoSampler
)

__all__ = [
    "UCIDataLoader",
    "DataValidator",
    "FeaturePreprocessor",
    "LabelEncoderWrapper",
    "DataSplitter",
    "DataPipeline",
    "SamplerFactory",
    "SamplingManager",
    "SMOTESampler",
    "ADASYNSampler",
    "NoSampler"
]
