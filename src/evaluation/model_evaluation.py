"""
Model Evaluation Module
Comprehensive evaluation metrics and visualization for sentiment analysis models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

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
    Comprehensive model evaluation class
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

        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")

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
                            confusion_matrix: np.ndarray, 
                            class_names: Optional[List[str]] = None,
                            model_name: str = "Model",
                            figsize: Tuple[int, int] = (8, 6)) -> None:
        """
        Plot confusion matrix

        Args:
            confusion_matrix (np.ndarray): Confusion matrix
            class_names (List[str], optional): Class names
            model_name (str): Model name for title
            figsize (Tuple[int, int]): Figure size
        """
        plt.figure(figsize=figsize)

        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(confusion_matrix))]

        # Plot heatmap
        sns.heatmap(confusion_matrix, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)

        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()

        if self.save_plots:
            plt.savefig(self.plot_dir / f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')

        plt.show()

    def plot_roc_curves(self, model_results: Dict[str, Dict[str, Any]], figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot ROC curves for multiple models

        Args:
            model_results (Dict[str, Dict[str, Any]]): Results from multiple models
            figsize (Tuple[int, int]): Figure size
        """
        plt.figure(figsize=figsize)

        colors = plt.cm.Set1(np.linspace(0, 1, len(model_results)))

        for i, (model_name, results) in enumerate(model_results.items()):
            if 'roc_curve' in results:
                fpr, tpr, _ = results['roc_curve']
                auc_score = results.get('roc_auc', 0)

                plt.plot(fpr, tpr, 
                        color=colors[i], 
                        label=f'{model_name} (AUC = {auc_score:.3f})')

        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)

        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if self.save_plots:
            plt.savefig(self.plot_dir / 'roc_curves_comparison.png')

        plt.show()

    def plot_precision_recall_curves(self, model_results: Dict[str, Dict[str, Any]], 
                                   figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot Precision-Recall curves for multiple models

        Args:
            model_results (Dict[str, Dict[str, Any]]): Results from multiple models
            figsize (Tuple[int, int]): Figure size
        """
        plt.figure(figsize=figsize)

        colors = plt.cm.Set1(np.linspace(0, 1, len(model_results)))

        for i, (model_name, results) in enumerate(model_results.items()):
            if 'pr_curve' in results:
                precision, recall, _ = results['pr_curve']

                plt.plot(recall, precision, 
                        color=colors[i], 
                        label=f'{model_name}')

        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if self.save_plots:
            plt.savefig(self.plot_dir / 'precision_recall_curves_comparison.png')

        plt.show()

    def plot_metrics_comparison(self, comparison_df: pd.DataFrame, 
                              figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot metrics comparison bar chart

        Args:
            comparison_df (pd.DataFrame): Model comparison dataframe
            figsize (Tuple[int, int]): Figure size
        """
        # Select main metrics
        main_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        available_metrics = [m for m in main_metrics if m in comparison_df.columns]

        if not available_metrics:
            logger.warning("No metrics available for plotting")
            return

        fig, axes = plt.subplots(1, len(available_metrics), figsize=figsize)
        if len(available_metrics) == 1:
            axes = [axes]

        for i, metric in enumerate(available_metrics):
            ax = axes[i]
            comparison_df.plot(x='Model', y=metric, kind='bar', ax=ax, 
                             color=plt.cm.Set3(np.linspace(0, 1, len(comparison_df))))
            ax.set_title(metric)
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if self.save_plots:
            plt.savefig(self.plot_dir / 'metrics_comparison.png')

        plt.show()

    def plot_class_distribution(self, y_true: np.ndarray, y_pred: np.ndarray,
                              class_names: Optional[List[str]] = None,
                              model_name: str = "Model",
                              figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot true vs predicted class distributions

        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            class_names (List[str], optional): Class names
            model_name (str): Model name
            figsize (Tuple[int, int]): Figure size
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(np.unique(y_true)))]

        # True distribution
        true_counts = np.bincount(y_true)
        axes[0].bar(range(len(true_counts)), true_counts, color='skyblue')
        axes[0].set_title('True Distribution')
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Count')
        axes[0].set_xticks(range(len(class_names)))
        axes[0].set_xticklabels(class_names)

        # Predicted distribution
        pred_counts = np.bincount(y_pred)
        axes[1].bar(range(len(pred_counts)), pred_counts, color='lightcoral')
        axes[1].set_title('Predicted Distribution')
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Count')
        axes[1].set_xticks(range(len(class_names)))
        axes[1].set_xticklabels(class_names)

        plt.suptitle(f'Class Distributions - {model_name}')
        plt.tight_layout()

        if self.save_plots:
            plt.savefig(self.plot_dir / f'class_distributions_{model_name.lower().replace(" ", "_")}.png')

        plt.show()

    def generate_evaluation_report(self, model_results: Dict[str, Dict[str, Any]]) -> str:
        """
        Generate a comprehensive evaluation report

        Args:
            model_results (Dict[str, Dict[str, Any]]): Results from multiple models

        Returns:
            str: Formatted evaluation report
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

        Args:
            model_results (Dict[str, Dict[str, Any]]): Model results
            filename (str): Output filename
        """
        report = self.generate_evaluation_report(model_results)

        filepath = self.plot_dir / filename
        with open(filepath, 'w') as f:
            f.write(report)

        logger.info(f"Evaluation report saved to {filepath}")

# Utility functions
def quick_evaluate_models(model_results: Dict[str, Dict[str, Any]], 
                        save_plots: bool = True) -> ModelEvaluator:
    """
    Quick evaluation of multiple models

    Args:
        model_results (Dict[str, Dict[str, Any]]): Model results
        save_plots (bool): Whether to save plots

    Returns:
        ModelEvaluator: Evaluator instance
    """
    evaluator = ModelEvaluator(save_plots=save_plots)

    # Generate comparison
    comparison_df = evaluator.compare_models(model_results)
    print(comparison_df)

    # Generate plots if data available
    evaluator.plot_metrics_comparison(comparison_df)

    # ROC curves if available
    roc_available = any('roc_curve' in results for results in model_results.values())
    if roc_available:
        evaluator.plot_roc_curves(model_results)

    # PR curves if available
    pr_available = any('pr_curve' in results for results in model_results.values())
    if pr_available:
        evaluator.plot_precision_recall_curves(model_results)

    # Generate and save report
    evaluator.save_evaluation_report(model_results)

    return evaluator

def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray, 
                              class_names: Optional[List[str]] = None) -> None:
    """
    Print formatted classification report

    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        class_names (List[str], optional): Class names
    """
    if class_names:
        target_names = class_names
    else:
        target_names = None

    report = classification_report(y_true, y_pred, target_names=target_names)
    print("CLASSIFICATION REPORT")
    print("=" * 40)
    print(report)
