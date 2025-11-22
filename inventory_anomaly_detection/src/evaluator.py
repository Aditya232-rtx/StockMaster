"""
Evaluation module for the Inventory Anomaly Detection system.
Provides functionality to evaluate model performance and generate reports.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score,
    roc_curve, roc_auc_score, precision_recall_curve
)
from typing import Dict, Tuple, List, Optional, Union
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Handles evaluation of the anomaly detection model.
    """
    
    def __init__(self, y_true: Optional[np.ndarray] = None, 
                 y_pred: Optional[np.ndarray] = None,
                 y_scores: Optional[np.ndarray] = None):
        """
        Initialize the ModelEvaluator with true labels, predictions, and scores.
        
        Args:
            y_true: True labels (1 for normal, -1 for anomaly)
            y_pred: Predicted labels (1 for normal, -1 for anomaly)
            y_scores: Anomaly scores (higher = more anomalous)
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_scores = y_scores
        self.metrics = {}
    
    def set_data(self, y_true: np.ndarray, y_pred: np.ndarray, 
                y_scores: np.ndarray) -> None:
        """
        Set the evaluation data.
        
        Args:
            y_true: True labels (1 for normal, -1 for anomaly)
            y_pred: Predicted labels (1 for normal, -1 for anomaly)
            y_scores: Anomaly scores (higher = more anomalous)
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_scores = y_scores
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Returns:
            Dictionary of evaluation metrics
        """
        if self.y_true is None or self.y_pred is None or self.y_scores is None:
            raise ValueError("Labels, predictions, and scores must be provided.")
        
        # Convert to binary (0 for normal, 1 for anomaly) for metrics calculation
        y_true_binary = (self.y_true == -1).astype(int)
        y_pred_binary = (self.y_pred == -1).astype(int)
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
        
        self.metrics = {
            'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            'true_negative_rate': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'auc_roc': roc_auc_score(y_true_binary, self.y_scores),
            'average_precision': average_precision_score(y_true_binary, self.y_scores)
        }
        
        return self.metrics
    
    def get_classification_report(self) -> str:
        """
        Generate a classification report.
        
        Returns:
            Formatted classification report
        """
        if self.y_true is None or self.y_pred is None:
            raise ValueError("Labels and predictions must be provided.")
        
        # Convert to binary (0 for normal, 1 for anomaly) for classification report
        y_true_binary = (self.y_true == -1).astype(int)
        y_pred_binary = (self.y_pred == -1).astype(int)
        
        # Generate classification report
        report = classification_report(
            y_true_binary, 
            y_pred_binary,
            target_names=['Normal', 'Anomaly'],
            output_dict=False
        )
        
        return report
    
    def plot_confusion_matrix(self, save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Plot a confusion matrix.
        
        Args:
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        if self.y_true is None or self.y_pred is None:
            raise ValueError("Labels and predictions must be provided.")
        
        # Convert to binary (0 for normal, 1 for anomaly)
        y_true_binary = (self.y_true == -1).astype(int)
        y_pred_binary = (self.y_pred == -1).astype(int)
        
        # Create confusion matrix
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        cm_df = pd.DataFrame(
            cm,
            index=['Normal', 'Anomaly'],
            columns=['Predicted Normal', 'Predicted Anomaly']
        )
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        # Save if path is provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_roc_curve(self, save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Plot the ROC curve.
        
        Args:
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        if self.y_true is None or self.y_scores is None:
            raise ValueError("Labels and scores must be provided.")
        
        # Convert to binary (0 for normal, 1 for anomaly)
        y_true_binary = (self.y_true == -1).astype(int)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true_binary, self.y_scores)
        roc_auc = roc_auc_score(y_true_binary, self.y_scores)
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        
        # Save if path is provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_precision_recall_curve(self, save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Plot the precision-recall curve.
        
        Args:
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        if self.y_true is None or self.y_scores is None:
            raise ValueError("Labels and scores must be provided.")
        
        # Convert to binary (0 for normal, 1 for anomaly)
        y_true_binary = (self.y_true == -1).astype(int)
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true_binary, self.y_scores)
        avg_precision = average_precision_score(y_true_binary, self.y_scores)
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f'Precision-Recall Curve (AP = {avg_precision:.2f})')
        
        # Save if path is provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_score_distribution(self, save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Plot the distribution of anomaly scores.
        
        Args:
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        if self.y_scores is None:
            raise ValueError("Scores must be provided.")
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.histplot(self.y_scores, kde=True, bins=50)
        plt.title('Distribution of Anomaly Scores')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Count')
        
        # Add vertical line at threshold (if available)
        if hasattr(self, 'threshold'):
            plt.axvline(x=self.threshold, color='r', linestyle='--', 
                       label=f'Threshold: {self.threshold:.2f}')
            plt.legend()
        
        # Save if path is provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def generate_report(self, output_dir: Union[str, Path]) -> None:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            output_dir: Directory to save the report and plots
        """
        if not self.metrics:
            self.calculate_metrics()
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate plots
        self.plot_confusion_matrix(output_dir / 'confusion_matrix.png')
        self.plot_roc_curve(output_dir / 'roc_curve.png')
        self.plot_precision_recall_curve(output_dir / 'precision_recall_curve.png')
        self.plot_score_distribution(output_dir / 'score_distribution.png')
        
        # Generate text report
        report = f"""# Anomaly Detection Model Evaluation Report

## Metrics Summary
"""
        # Add metrics to report
        for metric, value in self.metrics.items():
            report += f"- **{metric.replace('_', ' ').title()}**: {value:.4f}\n"
        
        # Add classification report
        report += "\n## Classification Report\n\n"
        report += self.get_classification_report()
        
        # Save report
        with open(output_dir / 'evaluation_report.md', 'w') as f:
            f.write(report)
        
        logger.info(f"Evaluation report saved to {output_dir}")


def main():
    """Example usage of the ModelEvaluator class."""
    import numpy as np
    from pathlib import Path
    
    # Example data
    np.random.seed(42)
    n_samples = 1000
    y_true = np.random.choice([-1, 1], size=n_samples, p=[0.05, 0.95])  # 5% anomalies
    y_scores = np.random.rand(n_samples)  # Random scores for demonstration
    
    # Create predictions based on a threshold
    threshold = np.percentile(y_scores, 95)  # 5% contamination
    y_pred = np.where(y_scores >= threshold, -1, 1)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(y_true, y_pred, y_scores)
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics()
    print("\nMetrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Generate classification report
    print("\nClassification Report:")
    print(evaluator.get_classification_report())
    
    # Generate full report
    output_dir = Path(__file__).parent.parent.parent / 'reports'
    evaluator.generate_report(output_dir)
    
    print(f"\nEvaluation report generated in {output_dir}")


if __name__ == "__main__":
    main()
