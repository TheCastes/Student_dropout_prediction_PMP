"""
Visualization Module.
Provides comprehensive visualization for data exploration and model results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import warnings


class VisualizationConfig:
    """Configuration for visualization settings."""
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 100,
        style: str = "seaborn-v0_8-whitegrid",
        palette: str = "Set2",
        save_path: Optional[str] = None
    ):
        self.figsize = figsize
        self.dpi = dpi
        self.style = style
        self.palette = palette
        self.save_path = Path(save_path) if save_path else None
        
        # Apply style
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-v0_8-whitegrid')
        
        sns.set_palette(palette)


class DataVisualizer:
    """Visualizations for data exploration."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
    
    def plot_class_distribution(
        self,
        y: pd.Series,
        class_names: Optional[Dict[int, str]] = None,
        title: str = "Class Distribution",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot the distribution of target classes.
        
        Args:
            y: Target series.
            class_names: Mapping of class indices to names.
            title: Plot title.
            save_name: Filename to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Count values
        if class_names:
            counts = y.value_counts().sort_index()
            labels = [class_names.get(i, str(i)) for i in counts.index]
        else:
            counts = y.value_counts()
            labels = counts.index.tolist()
        
        # Create bar plot
        colors = sns.color_palette(self.config.palette, len(counts))
        bars = ax.bar(labels, counts.values, color=colors, edgecolor='black')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                f'{count}\n({count/len(y)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10
            )
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_name and self.config.save_path:
            fig.savefig(self.config.save_path / save_name, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_feature_distributions(
        self,
        X: pd.DataFrame,
        n_cols: int = 4,
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot distributions of numerical features.
        
        Args:
            X: Features DataFrame.
            n_cols: Number of columns in subplot grid.
            save_name: Filename to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        n_features = len(numerical_cols)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(self.config.figsize[0] * 1.5, n_rows * 3)
        )
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, col in enumerate(numerical_cols):
            ax = axes[idx]
            X[col].hist(ax=ax, bins=30, edgecolor='black', alpha=0.7)
            ax.set_title(col[:30] + '...' if len(col) > 30 else col, fontsize=10)
            ax.set_xlabel('')
        
        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Feature Distributions', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_name and self.config.save_path:
            fig.savefig(self.config.save_path / save_name, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_correlation_matrix(
        self,
        X: pd.DataFrame,
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot correlation matrix heatmap.
        
        Args:
            X: Features DataFrame.
            save_name: Filename to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        corr_matrix = X[numerical_cols].corr()
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=False,
            cmap='RdBu_r',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_name and self.config.save_path:
            fig.savefig(self.config.save_path / save_name, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_sampling_comparison(
        self,
        original: pd.Series,
        resampled: pd.Series,
        class_names: Optional[Dict[int, str]] = None,
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare class distribution before and after sampling.
        
        Args:
            original: Original target series.
            resampled: Resampled target series.
            class_names: Mapping of class indices to names.
            save_name: Filename to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for ax, data, title in [
            (axes[0], original, "Before Sampling"),
            (axes[1], resampled, "After Sampling")
        ]:
            counts = pd.Series(data).value_counts().sort_index()
            
            if class_names:
                labels = [class_names.get(i, str(i)) for i in counts.index]
            else:
                labels = counts.index.tolist()
            
            colors = sns.color_palette(self.config.palette, len(counts))
            bars = ax.bar(labels, counts.values, color=colors, edgecolor='black')
            
            for bar, count in zip(bars, counts.values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(counts) * 0.01,
                    str(count),
                    ha='center', va='bottom', fontsize=10
                )
            
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            ax.set_title(title, fontweight='bold')
        
        plt.suptitle('Sampling Effect on Class Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_name and self.config.save_path:
            fig.savefig(self.config.save_path / save_name, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig


class ModelVisualizer:
    """Visualizations for model evaluation results."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        title: str = "Confusion Matrix",
        normalize: bool = False,
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix array.
            class_names: List of class names.
            title: Plot title.
            normalize: Whether to normalize the matrix.
            save_name: Filename to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar_kws={'shrink': 0.8}
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_name and self.config.save_path:
            fig.savefig(self.config.save_path / save_name, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        metric: str = 'F1 (Macro)',
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot bar chart comparing models.
        
        Args:
            comparison_df: DataFrame with model comparison results.
            metric: Metric to compare.
            save_name: Filename to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        df_sorted = comparison_df.sort_values(metric, ascending=True)
        colors = sns.color_palette(self.config.palette, len(df_sorted))
        
        bars = ax.barh(df_sorted['Model'], df_sorted[metric], color=colors, edgecolor='black')
        
        # Add value labels
        for bar, val in zip(bars, df_sorted[metric]):
            ax.text(
                val + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}',
                va='center', fontsize=10
            )
        
        ax.set_xlabel(metric, fontsize=12)
        ax.set_title(f'Model Comparison - {metric}', fontsize=14, fontweight='bold')
        ax.set_xlim(0, df_sorted[metric].max() * 1.15)
        
        plt.tight_layout()
        
        if save_name and self.config.save_path:
            fig.savefig(self.config.save_path / save_name, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_per_class_metrics(
        self,
        results: Dict[str, Dict[str, Any]],
        metric: str = 'f1_score',
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot per-class metrics for all models.
        
        Args:
            results: Dictionary of model results.
            metric: Metric to plot ('f1_score', 'precision', 'recall').
            save_name: Filename to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        # Prepare data
        data = []
        for model_name, result in results.items():
            for class_name, class_metrics in result['per_class_metrics'].items():
                data.append({
                    'Model': model_name,
                    'Class': class_name,
                    metric: class_metrics[metric]
                })
        
        df = pd.DataFrame(data)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Create grouped bar plot
        classes = df['Class'].unique()
        models = df['Model'].unique()
        x = np.arange(len(models))
        width = 0.25
        
        colors = sns.color_palette(self.config.palette, len(classes))
        
        for i, cls in enumerate(classes):
            cls_data = df[df['Class'] == cls]
            offset = (i - len(classes)/2 + 0.5) * width
            bars = ax.bar(
                x + offset,
                cls_data[metric],
                width,
                label=cls,
                color=colors[i],
                edgecolor='black'
            )
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'Per-Class {metric.replace("_", " ").title()} by Model',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(title='Class', loc='upper right')
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        if save_name and self.config.save_path:
            fig.savefig(self.config.save_path / save_name, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_feature_importance(
        self,
        importances: np.ndarray,
        feature_names: List[str],
        top_n: int = 20,
        title: str = "Feature Importance",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            importances: Feature importance array.
            feature_names: List of feature names.
            top_n: Number of top features to show.
            title: Plot title.
            save_name: Filename to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        colors = sns.color_palette('viridis', top_n)
        
        y_pos = np.arange(top_n)
        ax.barh(
            y_pos,
            importances[indices][::-1],
            color=colors[::-1],
            edgecolor='black'
        )
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in indices[::-1]])
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_name and self.config.save_path:
            fig.savefig(self.config.save_path / save_name, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig


class ReportGenerator:
    """Generates comprehensive visual reports."""
    
    def __init__(
        self,
        data_viz: Optional[DataVisualizer] = None,
        model_viz: Optional[ModelVisualizer] = None,
        save_path: Optional[str] = None
    ):
        config = VisualizationConfig(save_path=save_path)
        self.data_viz = data_viz or DataVisualizer(config)
        self.model_viz = model_viz or ModelVisualizer(config)
        self.save_path = Path(save_path) if save_path else None
    
    def generate_data_report(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        class_names: Optional[Dict[int, str]] = None
    ) -> List[plt.Figure]:
        """Generate all data exploration visualizations."""
        figures = []
        
        figures.append(self.data_viz.plot_class_distribution(
            y, class_names, save_name="class_distribution.png"
        ))
        
        figures.append(self.data_viz.plot_feature_distributions(
            X, save_name="feature_distributions.png"
        ))
        
        figures.append(self.data_viz.plot_correlation_matrix(
            X, save_name="correlation_matrix.png"
        ))
        
        return figures
    
    def generate_model_report(
        self,
        comparison_df: pd.DataFrame,
        results: Dict[str, Dict[str, Any]],
        class_names: List[str]
    ) -> List[plt.Figure]:
        """Generate all model evaluation visualizations."""
        figures = []
        
        figures.append(self.model_viz.plot_model_comparison(
            comparison_df, 'F1 (Macro)', save_name="model_comparison_f1.png"
        ))
        
        figures.append(self.model_viz.plot_model_comparison(
            comparison_df, 'Accuracy', save_name="model_comparison_accuracy.png"
        ))
        
        figures.append(self.model_viz.plot_per_class_metrics(
            results, 'f1_score', save_name="per_class_f1.png"
        ))
        
        # Plot confusion matrices for each model
        for model_name, result in results.items():
            safe_name = model_name.replace(' ', '_').lower()
            figures.append(self.model_viz.plot_confusion_matrix(
                result['confusion_matrix'],
                class_names,
                title=f'Confusion Matrix - {model_name}',
                save_name=f"confusion_matrix_{safe_name}.png"
            ))
        
        return figures
