"""
Model Evaluation Module - FIXED with proper plot generation
Comprehensive evaluation metrics and visualization for sentiment analysis models
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluation class with FIXED plot generation
    """
    
    def __init__(self, save_plots: bool = True, plot_dir: str = "plots"):
        """
        Initialize ModelEvaluator
        
        Args:
            save_plots (bool): Whether to save plots
            plot_dir (str): Directory to save plots
        """
        self.save_plots = save_plots
        self.plot_dir = Path(plot_dir)
        if self.save_plots:
            self.plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure matplotlib properly
        self._configure_matplotlib()
    
    def _configure_matplotlib(self):
        """Configure matplotlib for proper file saving"""
        matplotlib.use('Agg')  # Non-interactive backend
        plt.ioff()  # Turn off interactive mode
        
        # Set RC parameters for better output
        plt.rcParams.update({
            'figure.dpi': 100,
            'savefig.dpi': 150,
            'savefig.format': 'png',
            'savefig.bbox': 'tight',
            'savefig.facecolor': 'white',
            'savefig.edgecolor': 'none',
            'figure.figsize': [8, 6],
            'font.size': 10,
            'axes.labelsize': 10,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9
        })
    
    def evaluate_single_model(self, 
                            y_true: np.ndarray, 
                            y_pred: np.ndarray, 
                            y_proba: Optional[np.ndarray] = None,
                            model_name: str = "Model") -> Dict[str, Any]:
        """
        Evaluate a single model with comprehensive metrics
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_proba (np.ndarray, optional): Prediction probabilities
            model_name (str): Name of the model
            
        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        results = {'model_name': model_name}
        
        # Basic metrics
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['precision'] = precision_score(y_true, y_pred, average='weighted')
        results['recall'] = recall_score(y_true, y_pred, average='weighted')
        results['f1_score'] = f1_score(y_true, y_pred, average='weighted')
        
        # Per-class metrics
        results['precision_per_class'] = precision_score(y_true, y_pred, average=None)
        results['recall_per_class'] = recall_score(y_true, y_pred, average=None)
        results['f1_per_class'] = f1_score(y_true, y_pred, average=None)
        
        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Classification report
        results['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
        
        # ROC AUC if probabilities provided
        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                results['roc_auc'] = roc_auc_score(y_true, y_proba)
                results['roc_curve'] = roc_curve(y_true, y_proba)
                results['pr_curve'] = precision_recall_curve(y_true, y_proba)
            except ValueError as e:
                logger.warning(f"Could not calculate ROC metrics: {e}")
        
        # Class distribution analysis
        results['true_distribution'] = np.bincount(y_true) / len(y_true)
        results['pred_distribution'] = np.bincount(y_pred) / len(y_pred)
        
        return results
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            model_results (Dict[str, Dict[str, Any]]): Results from multiple models
            
        Returns:
            pd.DataFrame: Comparison dataframe
        """
        comparison_data = []
        
        for model_name, results in model_results.items():
            row = {
                'Model': model_name,
                'Accuracy': results.get('accuracy', 0),
                'Precision': results.get('precision', 0),
                'Recall': results.get('recall', 0),
                'F1-Score': results.get('f1_score', 0),
                'ROC-AUC': results.get('roc_auc', 0)
            }
            
            # Add per-class metrics if available
            if 'precision_per_class' in results:
                precision_per_class = results['precision_per_class']
                recall_per_class = results['recall_per_class']
                f1_per_class = results['f1_per_class']
                
                for i, (p, r, f) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
                    row[f'Precision_Class_{i}'] = p
                    row[f'Recall_Class_{i}'] = r
                    row[f'F1_Class_{i}'] = f
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df.round(4)
    
    def plot_confusion_matrix(self, 
                            confusion_matrix_data: np.ndarray, 
                            class_names: Optional[List[str]] = None,
                            model_name: str = "Model",
                            figsize: Tuple[int, int] = (8, 6)) -> None:
        """
        Plot confusion matrix with PROPER saving
        """
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(confusion_matrix_data))]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(confusion_matrix_data, interpolation='nearest', cmap='Blues')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Number of Predictions', rotation=270, labelpad=15)
        
        # Set ticks and labels
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        
        # Add text annotations
        thresh = confusion_matrix_data.max() / 2.0
        for i in range(confusion_matrix_data.shape[0]):
            for j in range(confusion_matrix_data.shape[1]):
                ax.text(j, i, format(confusion_matrix_data[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if confusion_matrix_data[i, j] > thresh else "black",
                       fontsize=16, weight='bold')
        
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title(f'Confusion Matrix - {model_name.replace("_", " ").title()}', 
                    fontsize=14, weight='bold', pad=20)
        
        if self.save_plots:
            filename = self.plot_dir / f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
            try:
                plt.savefig(filename, 
                           dpi=150, 
                           bbox_inches='tight', 
                           facecolor='white', 
                           edgecolor='none',
                           format='png')
                logger.info(f"✅ Saved confusion matrix: {filename}")
            except Exception as e:
                logger.error(f"❌ Error saving confusion matrix {filename}: {e}")
        
        plt.close(fig)  # Always close the figure
    
    def plot_roc_curves(self, model_results: Dict[str, Dict[str, Any]], 
                       figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot ROC curves for multiple models with PROPER saving
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        color_idx = 0
        
        for model_name, results in model_results.items():
            if 'roc_curve' in results:
                fpr, tpr, _ = results['roc_curve']
                auc_score = results.get('roc_auc', 0)
                
                ax.plot(fpr, tpr, 
                       color=colors[color_idx % len(colors)], 
                       linewidth=2,
                       label=f'{model_name.replace("_", " ").title()} (AUC = {auc_score:.3f})')
                color_idx += 1
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.8, linewidth=1, label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves Comparison', fontsize=14, weight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if self.save_plots:
            filename = self.plot_dir / 'roc_curves_comparison.png'
            try:
                plt.savefig(filename,
                           dpi=150,
                           bbox_inches='tight',
                           facecolor='white',
                           edgecolor='none',
                           format='png')
                logger.info(f"✅ Saved ROC curves: {filename}")
            except Exception as e:
                logger.error(f"❌ Error saving ROC curves {filename}: {e}")
        
        plt.close(fig)
    
    def plot_metrics_comparison(self, comparison_df: pd.DataFrame, 
                              figsize: Tuple[int, int] = (14, 6)) -> None:
        """
        Plot metrics comparison bar chart with PROPER saving
        """
        # Select main metrics
        main_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        available_metrics = [m for m in main_metrics if m in comparison_df.columns]
        
        if not available_metrics:
            logger.warning("No metrics available for plotting")
            return
        
        # Create subplots for each metric
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot accuracy and F1-score (most important)
        if 'Accuracy' in available_metrics:
            bars1 = ax1.bar(comparison_df['Model'], comparison_df['Accuracy'], 
                           color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(comparison_df)])
            ax1.set_title('Model Accuracy Comparison', fontsize=12, weight='bold')
            ax1.set_ylabel('Accuracy Score')
            ax1.set_ylim(0, 1)
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars1, comparison_df['Accuracy']):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom', fontsize=9, weight='bold')
        
        if 'F1-Score' in available_metrics:
            bars2 = ax2.bar(comparison_df['Model'], comparison_df['F1-Score'], 
                           color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(comparison_df)])
            ax2.set_title('Model F1-Score Comparison', fontsize=12, weight='bold')
            ax2.set_ylabel('F1-Score')
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars2, comparison_df['F1-Score']):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom', fontsize=9, weight='bold')
        
        plt.tight_layout()
        
        if self.save_plots:
            filename = self.plot_dir / 'metrics_comparison.png'
            try:
                plt.savefig(filename,
                           dpi=150,
                           bbox_inches='tight',
                           facecolor='white',
                           edgecolor='none',
                           format='png')
                logger.info(f"✅ Saved metrics comparison: {filename}")
            except Exception as e:
                logger.error(f"❌ Error saving metrics comparison {filename}: {e}")
        
        plt.close(fig)
    
    def plot_class_distribution(self, y_true: np.ndarray, y_pred: np.ndarray,
                              class_names: Optional[List[str]] = None,
                              model_name: str = "Model",
                              figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot true vs predicted class distributions with PROPER saving
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(np.unique(y_true)))]
        
        # True distribution
        true_counts = np.bincount(y_true)
        bars1 = ax1.bar(range(len(true_counts)), true_counts, 
                       color=['#ff7f7f', '#7fbf7f', '#7f7fff'][:len(true_counts)], alpha=0.8)
        ax1.set_title('True Class Distribution', fontsize=12, weight='bold')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.set_xticks(range(len(class_names)))
        ax1.set_xticklabels(class_names)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars1, true_counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(true_counts)*0.01, 
                    f'{value:,}', ha='center', va='bottom', fontsize=10, weight='bold')
        
        # Predicted distribution
        pred_counts = np.bincount(y_pred)
        bars2 = ax2.bar(range(len(pred_counts)), pred_counts, 
                       color=['#ff7f7f', '#7fbf7f', '#7f7fff'][:len(pred_counts)], alpha=0.8)
        ax2.set_title(f'Predicted Class Distribution\n({model_name.replace("_", " ").title()})', 
                     fontsize=12, weight='bold')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Count')
        ax2.set_xticks(range(len(class_names)))
        ax2.set_xticklabels(class_names)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars2, pred_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(pred_counts)*0.01, 
                    f'{value:,}', ha='center', va='bottom', fontsize=10, weight='bold')
        
        plt.tight_layout()
        
        if self.save_plots:
            filename = self.plot_dir / f'class_distributions_{model_name.lower().replace(" ", "_")}.png'
            try:
                plt.savefig(filename,
                           dpi=150,
                           bbox_inches='tight',
                           facecolor='white',
                           edgecolor='none',
                           format='png')
                logger.info(f"✅ Saved class distributions: {filename}")
            except Exception as e:
                logger.error(f"❌ Error saving class distributions {filename}: {e}")
        
        plt.close(fig)
    
    def generate_evaluation_report(self, model_results: Dict[str, Dict[str, Any]]) -> str:
        """
        Generate a comprehensive evaluation report
        """
        report = "SENTIMENT ANALYSIS MODEL EVALUATION REPORT\n"
        report += "=" * 60 + "\n\n"
        
        # Best model identification
        best_model = max(model_results.items(), key=lambda x: x[1].get('f1_score', 0))
        report += f"BEST MODEL: {best_model[0]} (F1-Score: {best_model[1].get('f1_score', 0):.4f})\n\n"
        
        # Model comparison
        comparison_df = self.compare_models(model_results)
        report += "MODEL COMPARISON:\n"
        report += "-" * 30 + "\n"
        report += comparison_df.to_string(index=False) + "\n\n"
        
        # Detailed results for each model
        for model_name, results in model_results.items():
            report += f"{model_name.upper()} DETAILED RESULTS:\n"
            report += "-" * 40 + "\n"
            report += f"Accuracy: {results.get('accuracy', 0):.4f}\n"
            report += f"Precision: {results.get('precision', 0):.4f}\n"
            report += f"Recall: {results.get('recall', 0):.4f}\n"
            report += f"F1-Score: {results.get('f1_score', 0):.4f}\n"
            
            if 'roc_auc' in results:
                report += f"ROC-AUC: {results['roc_auc']:.4f}\n"
            
            # Classification report
            if 'classification_report' in results:
                report += "\nClassification Report:\n"
                class_report = results['classification_report']
                for label, metrics in class_report.items():
                    if isinstance(metrics, dict):
                        report += f"  {label}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}\n"
            
            report += "\n"
        
        return report
    
    def save_evaluation_report(self, model_results: Dict[str, Dict[str, Any]], 
                             filename: str = "evaluation_report.txt") -> None:
        """
        Save evaluation report to file
        """
        report = self.generate_evaluation_report(model_results)
        
        filepath = self.plot_dir.parent / "reports" / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write(report)
        
        logger.info(f"✅ Evaluation report saved to {filepath}")
