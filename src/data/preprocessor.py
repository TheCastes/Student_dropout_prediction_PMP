"""
Data Preprocessing Module.
Handles data transformation, encoding, and preparation for ML models.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


class PreprocessorInterface(ABC):
    """Abstract interface for data preprocessing."""
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'PreprocessorInterface':
        """Fit the preprocessor to the data."""
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        pass
    
    @abstractmethod
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform the data."""
        pass


class LabelEncoderWrapper:
    """Wrapper for sklearn LabelEncoder to handle target encoding."""
    
    def __init__(self):
        self.encoder = LabelEncoder()
        self.classes_ = None
        self.class_mapping = None
    
    def fit(self, y: pd.Series) -> 'LabelEncoderWrapper':
        """Fit the encoder to the target variable."""
        self.encoder.fit(y)
        self.classes_ = self.encoder.classes_
        self.class_mapping = {
            label: idx for idx, label in enumerate(self.classes_)
        }
        return self
    
    def transform(self, y: pd.Series) -> np.ndarray:
        """Transform the target variable."""
        return self.encoder.transform(y)
    
    def fit_transform(self, y: pd.Series) -> np.ndarray:
        """Fit and transform the target variable."""
        self.fit(y)
        return self.transform(y)
    
    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Inverse transform encoded labels."""
        return self.encoder.inverse_transform(y)


class FeaturePreprocessor(PreprocessorInterface):
    """
    Preprocessor for feature engineering and transformation.
    Handles scaling, encoding, and feature selection.
    """
    
    def __init__(
        self,
        scaling: str = "standard",
        handle_categorical: bool = True
    ):
        """
        Initialize the feature preprocessor.
        
        Args:
            scaling: Type of scaling ('standard', 'minmax', or None).
            handle_categorical: Whether to encode categorical variables.
        """
        self.scaling = scaling
        self.handle_categorical = handle_categorical
        
        # Scalers
        self.scaler = None
        
        # Column information
        self.numerical_columns = []
        self.categorical_columns = []
        self.feature_names = []
        
        # Fitted flag
        self._is_fitted = False
    
    def _identify_column_types(self, X: pd.DataFrame) -> None:
        """Identify numerical and categorical columns."""
        self.numerical_columns = X.select_dtypes(
            include=['int64', 'float64']
        ).columns.tolist()
        
        self.categorical_columns = X.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        
        self.feature_names = X.columns.tolist()
    
    def _init_scaler(self) -> None:
        """Initialize the scaler based on configuration."""
        if self.scaling == "standard":
            self.scaler = StandardScaler()
        elif self.scaling == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'FeaturePreprocessor':
        """
        Fit the preprocessor to the training data.
        
        Args:
            X: Features DataFrame.
            y: Target Series (optional, not used but kept for compatibility).
            
        Returns:
            Self for method chaining.
        """
        self._identify_column_types(X)
        self._init_scaler()
        
        if self.scaler is not None and len(self.numerical_columns) > 0:
            self.scaler.fit(X[self.numerical_columns])
        
        self._is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using fitted preprocessor.
        
        Args:
            X: Features DataFrame to transform.
            
        Returns:
            Transformed DataFrame.
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")
        
        X_transformed = X.copy()
        
        # Apply scaling to numerical columns
        if self.scaler is not None and len(self.numerical_columns) > 0:
            X_transformed[self.numerical_columns] = self.scaler.transform(
                X[self.numerical_columns]
            )
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Fit and transform the data.
        
        Args:
            X: Features DataFrame.
            y: Target Series (optional).
            
        Returns:
            Transformed DataFrame.
        """
        self.fit(X, y)
        return self.transform(X)
    
    def get_feature_names(self) -> List[str]:
        """Get the list of feature names."""
        return self.feature_names.copy()


class DataSplitter:
    """Handles train/test splitting with stratification."""
    
    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True
    ):
        """
        Initialize the data splitter.
        
        Args:
            test_size: Proportion of data for testing.
            random_state: Random seed for reproducibility.
            stratify: Whether to use stratified splitting.
        """
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify
    
    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and test sets.
        
        Args:
            X: Features DataFrame.
            y: Target Series.
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        stratify_param = y if self.stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )
        
        return X_train, X_test, y_train, y_test


class DataPipeline:
    """
    Complete data pipeline combining all preprocessing steps.
    Orchestrates the data preprocessing workflow.
    """
    
    def __init__(
        self,
        feature_preprocessor: Optional[FeaturePreprocessor] = None,
        label_encoder: Optional[LabelEncoderWrapper] = None,
        data_splitter: Optional[DataSplitter] = None
    ):
        """
        Initialize the data pipeline.
        
        Args:
            feature_preprocessor: Preprocessor for features.
            label_encoder: Encoder for target labels.
            data_splitter: Splitter for train/test split.
        """
        self.feature_preprocessor = feature_preprocessor or FeaturePreprocessor()
        self.label_encoder = label_encoder or LabelEncoderWrapper()
        self.data_splitter = data_splitter or DataSplitter()
        
        # Store split data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
    
    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Any]:
        """
        Run the complete data pipeline.
        
        Args:
            X: Features DataFrame.
            y: Target Series.
            
        Returns:
            Dictionary containing all processed data.
        """
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        y_encoded = pd.Series(y_encoded, index=y.index)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.data_splitter.split(X, y_encoded)
        
        # Fit preprocessor on training data and transform both sets
        self.X_train_scaled = self.feature_preprocessor.fit_transform(
            self.X_train, self.y_train
        )
        self.X_test_scaled = self.feature_preprocessor.transform(self.X_test)
        
        return {
            "X_train": self.X_train,
            "X_test": self.X_test,
            "y_train": self.y_train,
            "y_test": self.y_test,
            "X_train_scaled": self.X_train_scaled,
            "X_test_scaled": self.X_test_scaled,
            "feature_names": self.feature_preprocessor.get_feature_names(),
            "class_mapping": self.label_encoder.class_mapping,
            "classes": self.label_encoder.classes_
        }
