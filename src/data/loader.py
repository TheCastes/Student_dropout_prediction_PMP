"""
Data Loader Module.
Handles loading data from UCI repository and basic data operations.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from abc import ABC, abstractmethod


class DataLoaderInterface(ABC):
    """Abstract interface for data loading."""
    
    @abstractmethod
    def load(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and return features and target."""
        pass
    
    @abstractmethod
    def get_metadata(self) -> dict:
        """Return dataset metadata."""
        pass


class UCIDataLoader(DataLoaderInterface):
    """
    Loads data from UCI Machine Learning Repository.
    Implements the DataLoaderInterface for UCI datasets.
    Supports both online fetching and local CSV loading.
    """
    
    def __init__(self, dataset_id: int = 697, csv_path: Optional[str] = None):
        """
        Initialize the UCI data loader.
        
        Args:
            dataset_id: The UCI repository dataset ID.
            csv_path: Optional path to local CSV file.
        """
        self.dataset_id = dataset_id
        self.csv_path = csv_path
        self._dataset = None
        self._features = None
        self._targets = None
        self._metadata = None
        self._variables = None
    
    def load(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load the dataset from UCI repository or local file.
        
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if self._features is None:
            if self.csv_path:
                self._load_from_csv()
            else:
                try:
                    self._fetch_dataset()
                except Exception:
                    # Fallback to generating sample data for demo
                    print("Warning: Could not fetch from UCI. Using sample data.")
                    self._generate_sample_data()
        
        return self._features.copy(), self._targets.copy()
    
    def _fetch_dataset(self) -> None:
        """Fetch the dataset from UCI repository."""
        try:
            from ucimlrepo import fetch_ucirepo
            
            self._dataset = fetch_ucirepo(id=self.dataset_id)
            self._features = self._dataset.data.features
            self._targets = self._dataset.data.targets
            
            # Flatten targets if needed
            if isinstance(self._targets, pd.DataFrame):
                self._targets = self._targets.iloc[:, 0]
            
            self._metadata = self._dataset.metadata
            self._variables = self._dataset.variables
            
        except ImportError:
            raise ImportError(
                "ucimlrepo package is required. "
                "Install it with: pip install ucimlrepo"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to fetch dataset: {e}")
    
    def _load_from_csv(self) -> None:
        """Load dataset from local CSV file."""
        df = pd.read_csv(self.csv_path, delimiter=';')
        
        # The last column is typically the target
        if 'Target' in df.columns:
            self._targets = df['Target']
            self._features = df.drop('Target', axis=1)
        else:
            # Assume last column is target
            self._targets = df.iloc[:, -1]
            self._features = df.iloc[:, :-1]
        
        self._metadata = {
            "name": "Student Dropout Prediction",
            "source": "local CSV"
        }
    
    def _generate_sample_data(self) -> None:
        """Generate sample data for demonstration purposes."""
        np.random.seed(42)
        n_samples = 4424
        
        # Define feature names based on the actual dataset
        feature_names = [
            'Marital status', 'Application mode', 'Application order', 'Course',
            'Daytime/evening attendance', 'Previous qualification',
            'Previous qualification (grade)', 'Nacionality',
            'Mother\'s qualification', 'Father\'s qualification',
            'Mother\'s occupation', 'Father\'s occupation',
            'Admission grade', 'Displaced', 'Educational special needs',
            'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder',
            'Age at enrollment', 'International',
            'Curricular units 1st sem (credited)',
            'Curricular units 1st sem (enrolled)',
            'Curricular units 1st sem (evaluations)',
            'Curricular units 1st sem (approved)',
            'Curricular units 1st sem (grade)',
            'Curricular units 1st sem (without evaluations)',
            'Curricular units 2nd sem (credited)',
            'Curricular units 2nd sem (enrolled)',
            'Curricular units 2nd sem (evaluations)',
            'Curricular units 2nd sem (approved)',
            'Curricular units 2nd sem (grade)',
            'Curricular units 2nd sem (without evaluations)',
            'Unemployment rate', 'Inflation rate', 'GDP'
        ]
        
        # Generate random features
        data = {}
        for feat in feature_names:
            if 'grade' in feat.lower():
                data[feat] = np.random.uniform(0, 200, n_samples)
            elif 'rate' in feat.lower() or 'GDP' in feat:
                data[feat] = np.random.uniform(-5, 10, n_samples)
            elif 'age' in feat.lower():
                data[feat] = np.random.randint(17, 60, n_samples)
            else:
                data[feat] = np.random.randint(0, 10, n_samples)
        
        self._features = pd.DataFrame(data)
        
        # Generate targets with class imbalance similar to real data
        # Approximately: Dropout 32%, Enrolled 18%, Graduate 50%
        targets = np.random.choice(
            ['Dropout', 'Enrolled', 'Graduate'],
            size=n_samples,
            p=[0.32, 0.18, 0.50]
        )
        self._targets = pd.Series(targets, name='Target')
        
        self._metadata = {
            "name": "Student Dropout Prediction (Sample Data)",
            "source": "generated"
        }
    
    def get_metadata(self) -> dict:
        """
        Get dataset metadata.
        
        Returns:
            Dictionary containing dataset metadata.
        """
        if self._features is None:
            self.load()  # This will handle fallback to sample data
        
        metadata = self._metadata or {}
        
        return {
            "name": metadata.get("name", "Unknown"),
            "num_instances": len(self._features) if self._features is not None else 0,
            "num_features": len(self._features.columns) if self._features is not None else 0,
            "feature_names": list(self._features.columns) if self._features is not None else [],
            "target_classes": self._targets.unique().tolist() if self._targets is not None else [],
        }
    
    def get_variable_info(self) -> pd.DataFrame:
        """
        Get information about dataset variables.
        
        Returns:
            DataFrame with variable information.
        """
        if self._features is None:
            self.load()
        
        if self._variables is not None:
            return self._variables.copy()
        
        # Generate basic variable info if not available
        if self._features is not None:
            return pd.DataFrame({
                'name': self._features.columns,
                'type': self._features.dtypes.values
            })
        
        return pd.DataFrame()


class DataValidator:
    """Validates data quality and integrity."""
    
    @staticmethod
    def check_missing_values(df: pd.DataFrame) -> pd.Series:
        """
        Check for missing values in the dataset.
        
        Args:
            df: DataFrame to check.
            
        Returns:
            Series with missing value counts per column.
        """
        return df.isnull().sum()
    
    @staticmethod
    def check_duplicates(df: pd.DataFrame) -> int:
        """
        Check for duplicate rows.
        
        Args:
            df: DataFrame to check.
            
        Returns:
            Number of duplicate rows.
        """
        return df.duplicated().sum()
    
    @staticmethod
    def get_data_types(df: pd.DataFrame) -> pd.Series:
        """
        Get data types of all columns.
        
        Args:
            df: DataFrame to analyze.
            
        Returns:
            Series with data types.
        """
        return df.dtypes
    
    @staticmethod
    def get_basic_stats(df: pd.DataFrame) -> pd.DataFrame:
        """
        Get basic statistics for numerical columns.
        
        Args:
            df: DataFrame to analyze.
            
        Returns:
            DataFrame with basic statistics.
        """
        return df.describe()
    
    def validate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Perform complete validation of the dataset.
        
        Args:
            X: Features DataFrame.
            y: Target Series.
            
        Returns:
            Dictionary with validation results.
        """
        return {
            "missing_values": self.check_missing_values(X).to_dict(),
            "total_missing": self.check_missing_values(X).sum(),
            "duplicates": self.check_duplicates(X),
            "data_types": self.get_data_types(X).to_dict(),
            "num_samples": len(X),
            "num_features": len(X.columns),
            "target_distribution": y.value_counts().to_dict(),
            "target_classes": y.unique().tolist()
        }
