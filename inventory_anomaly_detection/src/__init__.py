"""
Inventory Anomaly Detection System

This package provides functionality to detect anomalies in inventory transactions
using machine learning techniques, specifically Isolation Forest with ensemble methods.
"""

__version__ = "0.1.0"

# Import key components for easier access
from .data_processor import DataProcessor
from .feature_engineer import FeatureEngineer
from .anomaly_detector import AnomalyDetector
from .evaluator import ModelEvaluator

__all__ = [
    'DataProcessor',
    'FeatureEngineer',
    'AnomalyDetector',
    'ModelEvaluator'
]
