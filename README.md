# Student Performance Prediction System

A modular, object-oriented Python system for predicting student academic performance using machine learning techniques.

## Overview

This system implements the methodology described in the paper:
> "Early Prediction of student's Performance in Higher Education: A Case Study"
> by Martins et al. (2021)

The system predicts student outcomes (Dropout, Enrolled, Graduate) using data available at enrollment time.

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for data loading, preprocessing, modeling, and visualization
- **Multiple ML Algorithms**: 
  - Standard: Logistic Regression, SVM, Decision Tree, Random Forest
  - Boosting: Gradient Boosting, XGBoost, CatBoost, LogitBoost
- **Class Imbalance Handling**: SMOTE and ADASYN sampling techniques
- **Hyperparameter Tuning**: Grid Search and Randomized Search
- **Comprehensive Visualization**: Class distributions, confusion matrices, model comparisons
- **Interactive Web Interface**: Streamlit-based UI for easy interaction
- **Extensible Design**: Easy to add new models or preprocessing techniques

## Project Structure

```
student_prediction/
├── app.py                     # Streamlit web interface
├── main.py                    # Main pipeline orchestrator
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── outputs/                   # Generated outputs (plots, reports)
└── src/
    ├── __init__.py
    ├── config.py              # Configuration classes
    ├── data/
    │   ├── __init__.py
    │   ├── loader.py          # Data loading from UCI
    │   ├── preprocessor.py    # Data preprocessing
    │   └── sampling.py        # SMOTE/ADASYN sampling
    ├── models/
    │   ├── __init__.py
    │   ├── base.py            # Base model interfaces
    │   ├── standard.py        # Standard ML models
    │   ├── boosting.py        # Boosting models
    │   └── trainer.py         # Training & evaluation
    ├── visualization/
    │   ├── __init__.py
    │   └── visualizer.py      # All visualizations
    └── utils/
        ├── __init__.py
        └── helpers.py         # Utility functions
```

## Installation

```bash
# Clone or download the project
cd student_prediction

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Web Interface (Streamlit)

The easiest way to use the system is through the Streamlit web interface:

```bash
streamlit run app.py
```

This will open a browser with an interactive interface where you can:
1. **Load Data**: From UCI repository or upload your own CSV
2. **Preprocess**: Configure test split, scaling, and sampling
3. **Train Models**: Select and train multiple ML models
4. **View Results**: Interactive charts, confusion matrices, and comparisons
5. **Make Predictions**: Enter student data and get predictions

### Command Line (Python API)

```python
from main import StudentPerformancePipeline

# Create and run the pipeline
pipeline = StudentPerformancePipeline(output_dir="outputs")
results = pipeline.run_full_pipeline(
    sampler_type="smote",
    tune_hyperparameters=False,
    train_boosting=True
)
```

### Step-by-Step Usage

```python
from main import StudentPerformancePipeline

# Initialize pipeline
pipeline = StudentPerformancePipeline()

# 1. Load data from UCI repository
X, y = pipeline.load_data()

# 2. Validate data quality
validation = pipeline.validate_data()

# 3. Preprocess (split and scale)
processed = pipeline.preprocess_data(scaling="standard")

# 4. Apply SMOTE sampling
X_resampled, y_resampled = pipeline.apply_sampling("smote")

# 5. Train models
comparison, results = pipeline.train_all_models(tune_hyperparameters=False)

# 6. Generate visualizations
pipeline.visualize_data()
pipeline.visualize_results()

# 7. Print report
pipeline.generate_report()
```

### Using Individual Components

```python
# Data Loading
from src.data import UCIDataLoader
loader = UCIDataLoader(dataset_id=697)
X, y = loader.load()

# Preprocessing
from src.data import FeaturePreprocessor, DataSplitter
preprocessor = FeaturePreprocessor(scaling="standard")
splitter = DataSplitter(test_size=0.2)

# Sampling
from src.data import SamplerFactory
smote = SamplerFactory.create("smote", random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Models
from src.models import RandomForestModel, XGBoostModel
rf = RandomForestModel(random_state=42)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)

# Visualization
from src.visualization import DataVisualizer
viz = DataVisualizer()
viz.plot_class_distribution(y)
```

## Configuration

Customize the pipeline via `ProjectConfig`:

```python
from src.config import ProjectConfig, DataConfig, ModelConfig

config = ProjectConfig(
    data=DataConfig(
        test_size=0.2,
        random_state=42
    ),
    model=ModelConfig(
        cv_folds=5,
        scoring_metric="f1_macro",
        n_iter_random_search=100
    )
)

pipeline = StudentPerformancePipeline(config=config)
```

## Dataset

The system uses the "Predict Students' Dropout and Academic Success" dataset from UCI ML Repository:
- **ID**: 697
- **Samples**: 4424
- **Features**: 36
- **Classes**: Dropout, Enrolled, Graduate

## Results

Based on the paper's methodology, expected results:
- Boosting methods (XGBoost, Gradient Boosting) typically achieve the best F1 scores
- The "Enrolled" class (intermediate) is hardest to predict due to its minority status
- SMOTE sampling improves minority class prediction at the cost of overall accuracy

## Extending the System

### Adding a New Model

```python
from src.models.base import BaseModel, ModelRegistry

class MyNewModel(BaseModel):
    def __init__(self, random_state=42, **kwargs):
        super().__init__("My New Model", random_state)
        self._init_model()
    
    def _init_model(self):
        from sklearn.ensemble import SomeClassifier
        self.model = SomeClassifier(random_state=self.random_state)
    
    def get_hyperparameter_grid(self):
        return {'param1': [1, 2, 3], 'param2': ['a', 'b']}

# Register the model
ModelRegistry.register("my_new_model", MyNewModel)
```

### Adding a New Sampler

```python
from src.data.sampling import SamplerInterface, SamplerFactory

class MyNewSampler(SamplerInterface):
    def fit_resample(self, X, y):
        # Implementation
        return X_resampled, y_resampled
    
    def get_name(self):
        return "My New Sampler"

# Register in SamplerFactory._samplers
```

## License

This project is for educational purposes, based on publicly available research.

## References

1. Martins, M. V., et al. (2021). Early Prediction of student's Performance in Higher Education: A Case Study. WorldCIST 2021.
2. UCI ML Repository: https://archive.ics.uci.edu/dataset/697
