"""
Run the complete inventory anomaly detection pipeline.
This script runs the entire workflow from data processing to model evaluation.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

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
    parser = argparse.ArgumentParser(description='Run Inventory Anomaly Detection Pipeline')
    
    # Data paths
    parser.add_argument('--data-dir', type=str, default='data',
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

def run_pipeline(args):
    """Run the complete anomaly detection pipeline."""
    logger.info("Starting Inventory Anomaly Detection Pipeline")
    
    # Set up directories
    paths = setup_directories(args.output_dir)
    
    try:
        # 1. Load and process data
        logger.info("Step 1/5: Loading and processing data...")
        processor = DataProcessor(args.data_dir)
        processor.load_data()
        processor.clean_data()
        products, transactions, vendor_prices = processor.get_processed_data()
        
        # Save processed data
        products.to_csv(paths['data'] / 'processed_products.csv', index=False)
        transactions.to_csv(paths['data'] / 'processed_transactions.csv', index=False)
        vendor_prices.to_csv(paths['data'] / 'processed_vendor_prices.csv', index=False)
        
        # 2. Feature engineering
        logger.info("Step 2/5: Creating features...")
        feature_engineer = FeatureEngineer(transactions, products)
        features = feature_engineer.create_all_features()
        feature_cols = feature_engineer.get_feature_columns()
        X = features[feature_cols].fillna(0)
        
        # Save features
        features.to_csv(paths['data'] / 'all_features.csv', index=False)
        
        # 3. Train the model
        logger.info("Step 3/5: Training the anomaly detection model...")
        detector = AnomalyDetector(
            contamination=args.contamination,
            n_estimators=args.n_estimators,
            random_state=args.random_state
        )
        
        # For demonstration, we'll use all data for training
        # In a real scenario, you would split into train/test sets
        detector.fit(X)
        
        # Save the model
        model_path = paths['models'] / 'inventory_anomaly_detector.joblib'
        detector.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # 4. Make predictions
        logger.info("Step 4/5: Making predictions...")
        scores = detector.score_samples(X)
        predictions = detector.predict(X)
        risk_levels = detector.get_risk_levels(scores)
        
        # Create results DataFrame
        results = features.copy()
        results['anomaly_score'] = scores
        results['prediction'] = predictions
        results['risk_level'] = risk_levels
        
        # Save results
        results_path = paths['output'] / 'anomaly_predictions.csv'
        results.to_csv(results_path, index=False)
        logger.info(f"Predictions saved to {results_path}")
        
        # 5. Evaluate the model
        logger.info("Step 5/5: Evaluating the model...")
        
        # For demonstration, we'll simulate some ground truth labels
        # In a real scenario, you would have actual labeled anomalies
        np.random.seed(args.random_state)
        y_true = np.random.choice([-1, 1], size=len(X), p=[args.contamination, 1-args.contamination])
        
        # Initialize evaluator
        evaluator = ModelEvaluator(y_true, predictions, scores)
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics()
        
        # Print metrics
        print("\nModel Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Generate and save report
        evaluator.generate_report(paths['reports'])
        logger.info(f"Evaluation report saved to {paths['reports']}")
        
        # Print summary
        print("\nPrediction Summary:")
        print(f"Total transactions: {len(results)}")
        print(f"Anomalies detected: {(results['prediction'] == -1).sum()} ({(results['prediction'] == -1).mean()*100:.2f}%)")
        print("\nRisk level distribution:")
        print(results['risk_level'].value_counts())
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred during pipeline execution: {e}", exc_info=True)
        raise

def main():
    """Main function to run the pipeline."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Run the pipeline
    run_pipeline(args)

if __name__ == "__main__":
    main()
