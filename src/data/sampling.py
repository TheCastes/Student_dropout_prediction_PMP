"""
Data Sampling Module.
Handles class imbalance through various sampling techniques.
Implements SMOTE and ADASYN as described in the paper.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from abc import ABC, abstractmethod


class SamplerInterface(ABC):
    """Abstract interface for data sampling strategies."""
    
    @abstractmethod
    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and resample the data."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the sampling strategy."""
        pass


class NoSampler(SamplerInterface):
    """No sampling - returns data as is."""
    
    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return data without any modifications."""
        return X, y
    
    def get_name(self) -> str:
        return "No Sampling"


class SMOTESampler(SamplerInterface):
    """
    SMOTE (Synthetic Minority Over-sampling Technique) sampler.
    Creates synthetic samples for minority classes.
    """
    
    def __init__(
        self,
        random_state: int = 42,
        k_neighbors: int = 5,
        sampling_strategy: str = "auto"
    ):
        """
        Initialize SMOTE sampler.
        
        Args:
            random_state: Random seed for reproducibility.
            k_neighbors: Number of nearest neighbors for synthesis.
            sampling_strategy: Strategy for resampling.
        """
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy
        self._smote = None
    
    def _init_smote(self) -> None:
        """Initialize the SMOTE object."""
        try:
            from imblearn.over_sampling import SMOTE
            self._smote = SMOTE(
                random_state=self.random_state,
                k_neighbors=self.k_neighbors,
                sampling_strategy=self.sampling_strategy
            )
        except ImportError:
            raise ImportError(
                "imbalanced-learn package is required for SMOTE. "
                "Install it with: pip install imbalanced-learn"
            )
    
    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE to resample the data.
        
        Args:
            X: Features array.
            y: Target array.
            
        Returns:
            Tuple of (resampled X, resampled y).
        """
        if self._smote is None:
            self._init_smote()
        
        X_resampled, y_resampled = self._smote.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def get_name(self) -> str:
        return "SMOTE"


class ADASYNSampler(SamplerInterface):
    """
    ADASYN (Adaptive Synthetic Sampling) sampler.
    Similar to SMOTE but focuses on harder to learn examples.
    """
    
    def __init__(
        self,
        random_state: int = 42,
        n_neighbors: int = 5,
        sampling_strategy: str = "auto"
    ):
        """
        Initialize ADASYN sampler.
        
        Args:
            random_state: Random seed for reproducibility.
            n_neighbors: Number of nearest neighbors.
            sampling_strategy: Strategy for resampling.
        """
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.sampling_strategy = sampling_strategy
        self._adasyn = None
    
    def _init_adasyn(self) -> None:
        """Initialize the ADASYN object."""
        try:
            from imblearn.over_sampling import ADASYN
            self._adasyn = ADASYN(
                random_state=self.random_state,
                n_neighbors=self.n_neighbors,
                sampling_strategy=self.sampling_strategy
            )
        except ImportError:
            raise ImportError(
                "imbalanced-learn package is required for ADASYN. "
                "Install it with: pip install imbalanced-learn"
            )
    
    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply ADASYN to resample the data.
        
        Args:
            X: Features array.
            y: Target array.
            
        Returns:
            Tuple of (resampled X, resampled y).
        """
        if self._adasyn is None:
            self._init_adasyn()
        
        X_resampled, y_resampled = self._adasyn.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def get_name(self) -> str:
        return "ADASYN"


class SamplerFactory:
    """Factory for creating sampler instances."""
    
    _samplers = {
        "none": NoSampler,
        "smote": SMOTESampler,
        "adasyn": ADASYNSampler
    }
    
    @classmethod
    def create(
        cls,
        sampler_type: str,
        **kwargs
    ) -> SamplerInterface:
        """
        Create a sampler instance.
        
        Args:
            sampler_type: Type of sampler ('none', 'smote', 'adasyn').
            **kwargs: Additional arguments for the sampler.
            
        Returns:
            Sampler instance.
        """
        sampler_type = sampler_type.lower()
        
        if sampler_type not in cls._samplers:
            raise ValueError(
                f"Unknown sampler type: {sampler_type}. "
                f"Available types: {list(cls._samplers.keys())}"
            )
        
        return cls._samplers[sampler_type](**kwargs)
    
    @classmethod
    def available_samplers(cls) -> list:
        """Get list of available sampler types."""
        return list(cls._samplers.keys())


class SamplingManager:
    """
    Manages the sampling process for handling class imbalance.
    Provides comparison between different sampling strategies.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the sampling manager.
        
        Args:
            random_state: Random seed for reproducibility.
        """
        self.random_state = random_state
        self.sampling_results = {}
    
    def apply_sampling(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sampler_type: str = "smote",
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply a specific sampling strategy.
        
        Args:
            X: Features array.
            y: Target array.
            sampler_type: Type of sampler to use.
            **kwargs: Additional arguments for the sampler.
            
        Returns:
            Tuple of (resampled X, resampled y).
        """
        kwargs.setdefault("random_state", self.random_state)
        sampler = SamplerFactory.create(sampler_type, **kwargs)
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        # Store results for comparison
        self.sampling_results[sampler_type] = {
            "sampler_name": sampler.get_name(),
            "original_distribution": pd.Series(y).value_counts().to_dict(),
            "resampled_distribution": pd.Series(y_resampled).value_counts().to_dict(),
            "original_size": len(y),
            "resampled_size": len(y_resampled)
        }
        
        return X_resampled, y_resampled
    
    def compare_samplers(
        self,
        X: np.ndarray,
        y: np.ndarray,
        samplers: Optional[list] = None
    ) -> dict:
        """
        Compare different sampling strategies.
        
        Args:
            X: Features array.
            y: Target array.
            samplers: List of sampler types to compare.
            
        Returns:
            Dictionary with comparison results.
        """
        if samplers is None:
            samplers = SamplerFactory.available_samplers()
        
        results = {}
        for sampler_type in samplers:
            try:
                self.apply_sampling(X.copy(), y.copy(), sampler_type)
                results[sampler_type] = self.sampling_results[sampler_type]
            except Exception as e:
                results[sampler_type] = {"error": str(e)}
        
        return results
    
    def get_sampling_report(self) -> pd.DataFrame:
        """
        Get a summary report of all applied sampling strategies.
        
        Returns:
            DataFrame with sampling results.
        """
        if not self.sampling_results:
            return pd.DataFrame()
        
        report_data = []
        for sampler_type, result in self.sampling_results.items():
            if "error" not in result:
                report_data.append({
                    "Sampler": result["sampler_name"],
                    "Original Size": result["original_size"],
                    "Resampled Size": result["resampled_size"],
                    "Size Increase": result["resampled_size"] - result["original_size"],
                    "Original Distribution": str(result["original_distribution"]),
                    "Resampled Distribution": str(result["resampled_distribution"])
                })
        
        return pd.DataFrame(report_data)
