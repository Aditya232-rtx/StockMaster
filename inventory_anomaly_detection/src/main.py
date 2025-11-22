"""
Main script for the Inventory Anomaly Detection system.
This script ties together all components to detect anomalies in inventory transactions.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processor import DataProcessor
from src.feature_engineer import FeatureEngineer
from src.anomaly_detector import AnomalyDetector
from src.evaluator import ModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('inventory_anomaly_detection.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Inventory Anomaly Detection System')
    
    # Data paths
    parser.add_argument('--data-dir', type=str, default='.',
                       help='Directory containing the input data files')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Directory to save outputs')
    
    # Model parameters
    parser.add_argument('--contamination', type=float, default=0.05,
                       help='Expected proportion of anomalies in the data (0-0.5)')
    parser.add_argument('--n-estimators', type=int, default=100,
                       help='Number of base estimators in the ensemble')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Operation mode
    parser.add_argument('--train', action='store_true',
                       help='Train a new model')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate model performance')
    parser.add_argument('--predict', action='store_true',
                       help='Make predictions on new data')
    parser.add_argument('--model-path', type=str, default='models/inventory_anomaly_detector.joblib',
                       help='Path to save/load the model')
    
    return parser.parse_args()

def setup_directories(output_dir: str) -> dict:
    """Create necessary directories if they don't exist."""
    paths = {
        'output': Path(output_dir),
        'models': Path(output_dir) / 'models',
        'reports': Path(output_dir) / 'reports',
        'data': Path(output_dir) / 'data'
    }
    
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    return paths

def train_model(data_dir: str, output_path: str, contamination: float, 
               n_estimators: int, random_state: int) -> AnomalyDetector:
    """Train a new anomaly detection model."""
    logger.info("Starting model training...")
    
    # Load and process data
    logger.info("Loading and processing data...")
    processor = DataProcessor(data_dir)
    processor.load_data()
    processor.clean_data()
    products, transactions, _ = processor.get_processed_data()
    
    # Create features
    logger.info("Creating features...")
    feature_engineer = FeatureEngineer(transactions, products)
    features = feature_engineer.create_all_features()
    feature_cols = feature_engineer.get_feature_columns()
    X = features[feature_cols].fillna(0)
    
    # Train the model
    logger.info("Training anomaly detection model...")
    detector = AnomalyDetector(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state
    )
    
    detector.fit(X)
    
    # Save the model
    detector.save_model(output_path)
    logger.info(f"Model trained and saved to {output_path}")
    
    return detector, X

def evaluate_model(data_dir: str, model_path: str, output_dir: str) -> None:
    """Evaluate the performance of a trained model."""
    logger.info("Starting model evaluation...")
    
    # Load the model
    detector = AnomalyDetector.load_model(model_path)
    
    # Load and process data
    processor = DataProcessor(data_dir)
    processor.load_data()
    processor.clean_data()
    products, transactions, _ = processor.get_processed_data()
    
    # Create features
    feature_engineer = FeatureEngineer(transactions, products)
    features = feature_engineer.create_all_features()
    feature_cols = feature_engineer.get_feature_columns()
    X = features[feature_cols].fillna(0)
    
    # Make predictions
    logger.info("Making predictions...")
    scores = detector.score_samples(X)
    predictions = detector.predict(X)
    
    # For evaluation, we need some ground truth (simulated for this example)
    # In a real scenario, you would have labeled anomalies
    y_true = np.random.choice([-1, 1], size=len(X), p=[0.05, 0.95])  # Simulated ground truth
    
    # Evaluate
    logger.info("Evaluating model performance...")
    evaluator = ModelEvaluator(y_true, predictions, scores)
    metrics = evaluator.calculate_metrics()
    
    # Print metrics
    print("\nModel Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Generate and save report
    evaluator.generate_report(output_dir)
    logger.info(f"Evaluation report saved to {output_dir}")

def predict_anomalies(data_dir: str, model_path: str, output_path: str) -> None:
    """Make predictions on new data using a trained model."""
    logger.info("Starting prediction on new data...")
    
    # Load the model
    detector = AnomalyDetector.load_model(model_path)
    
    # Load and process new data
    # Note: This assumes the new data has the same structure as the training data
    processor = DataProcessor(data_dir)
    processor.load_data()
    processor.clean_data()
    products, transactions, _ = processor.get_processed_data()
    
    # Create features
    feature_engineer = FeatureEngineer(transactions, products)
    features = feature_engineer.create_all_features()
    feature_cols = feature_engineer.get_feature_columns()
    X = features[feature_cols].fillna(0)
    
    # Make predictions
    logger.info("Making predictions...")
    scores = detector.score_samples(X)
    predictions = detector.predict(X)
    risk_levels = detector.get_risk_levels(scores)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'transaction_id': features.index,
        'date': features['date'],
        'product_sku': features['product_sku'],
        'transaction_type': features['transaction_type'],
        'quantity_change': features['quantity_change'],
        'location': features['location'],
        'anomaly_score': scores,
        'prediction': predictions,
        'risk_level': risk_levels
    })
    
    # Save results
    results.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
    
    # Print summary
    print("\nPrediction Summary:")
    print(f"Total transactions: {len(results)}")
    print(f"Anomalies detected: {(results['prediction'] == -1).sum()} ({(results['prediction'] == -1).mean()*100:.2f}%)")
    print("\nRisk level distribution:")
    print(results['risk_level'].value_counts())

def main():
    """Main function to run the inventory anomaly detection system."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up directories
    paths = setup_directories(args.output_dir)
    
    # Set model path
    model_path = Path(args.model_path)
    
    try:
        # Train mode
        if args.train:
            detector, _ = train_model(
                data_dir=args.data_dir,
                output_path=model_path,
                contamination=args.contamination,
                n_estimators=args.n_estimators,
                random_state=args.random_state
            )
        
        # Evaluate mode
        if args.evaluate:
            evaluate_model(
                data_dir=args.data_dir,
                model_path=model_path,
                output_dir=str(paths['reports'])
            )
        
        # Predict mode (default if no mode specified)
        if args.predict or not (args.train or args.evaluate):
            output_path = paths['output'] / 'anomaly_predictions.csv'
            predict_anomalies(
                data_dir=args.data_dir,
                model_path=model_path,
                output_path=str(output_path)
            )
    
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
