"""
Anomaly detection module for the Inventory Anomaly Detection system.
Implements Isolation Forest and ensemble methods for detecting anomalies.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import Dict, List, Tuple, Optional, Union
import joblib
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    Implements anomaly detection using Isolation Forest and ensemble methods.
    """
    
    def __init__(self, 
                 contamination: float = 0.05,
                 random_state: int = 42,
                 n_estimators: int = 100,
                 max_samples: Union[float, int] = 'auto',
                 n_jobs: int = -1):
        """
        Initialize the AnomalyDetector.
        
        Args:
            contamination: Expected proportion of outliers in the data (0-0.5)
            random_state: Random seed for reproducibility
            n_estimators: Number of base estimators in the ensemble
            max_samples: Number of samples to draw to train each base estimator
            n_jobs: Number of jobs to run in parallel (-1 uses all available cores)
        """
        self.contamination = contamination
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        
        # Initialize models
        self.models = {
            'temporal': self._init_model(),
            'quantity': self._init_model(),
            'value': self._init_model(),
            'ensemble': self._init_model()
        }
        
        # Feature groups for different models
        self.feature_groups = {
            'temporal': [
                'hour', 'day_of_week', 'day_of_month', 'week_of_year', 
                'month', 'quarter', 'year', 'is_weekend', 'is_business_hours',
                'is_month_end', 'is_quarter_end', 'is_year_end', 'time_since_last_txn',
                'txn_count_7d', 'txn_count_30d', 'txn_count_90d'
            ],
            'quantity': [
                'abs_quantity', 'avg_txn_size_30d', 'std_txn_size_30d',
                'z_score_txn_size', 'is_large_txn', 'is_very_large_txn',
                'days_of_inventory', 'reorder_point_proximity'
            ],
            'value': [
                'transaction_value', 'avg_txn_value_30d', 'value_ratio_to_avg',
                'days_of_inventory', 'reorder_point_proximity'
            ]
        }
        
        # Initialize scalers per model to keep feature dimensions aligned
        self.scalers = {
            model_type: StandardScaler()
            for model_type in self.models
            if model_type != 'ensemble'
        }
        self.score_scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Store feature names and other metadata
        self.feature_names_ = None
        self.fitted_ = False
    
    def _init_model(self) -> IsolationForest:
        """Initialize an Isolation Forest model with the specified parameters."""
        return IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=0
        )
    
    def _prepare_features(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Prepare feature sets for different models.
        
        Args:
            X: Input DataFrame with all features
            
        Returns:
            Dictionary with feature sets for each model
        """
        # Store feature names if not already set
        if self.feature_names_ is None:
            self.feature_names_ = X.columns.tolist()
        
        # Select features for each model
        prepared = {}
        for model_type, features in self.feature_groups.items():
            # Get only the features that exist in X
            available_features = [f for f in features if f in X.columns]
            prepared[model_type] = X[available_features].values
            
            # Log any missing features
            missing = set(features) - set(available_features)
            if missing:
                logger.warning(f"Missing features for {model_type} model: {missing}")
        
        # Also prepare the full feature set for the ensemble model
        prepared['ensemble'] = X.values
        
        return prepared
    
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'AnomalyDetector':
        """
        Fit the anomaly detection models.
        
        Args:
            X: Input features
            y: Not used, present for scikit-learn compatibility
            
        Returns:
            self: Returns an instance of self
        """
        logger.info("Fitting anomaly detection models...")
        
        # Prepare the features
        feature_sets = self._prepare_features(X)
        
        # Fit each model on its feature set
        for model_type, model in self.models.items():
            logger.info(f"Fitting {model_type} model...")

            if model_type == 'ensemble':
                continue

            if model_type not in feature_sets:
                continue

            scaler = self.scalers.setdefault(model_type, StandardScaler())
            X_scaled = scaler.fit_transform(feature_sets[model_type])
            model.fit(X_scaled)

            # Get anomaly scores from this model
            scores = -model.score_samples(X_scaled)  # Convert to positive (higher = more anomalous)

            # Store the scores as features for the ensemble model
            if 'ensemble_scores' not in locals():
                ensemble_scores = scores.reshape(-1, 1)
            else:
                ensemble_scores = np.hstack([ensemble_scores, scores.reshape(-1, 1)])
        
        # Fit the ensemble model on the scores from individual models
        if 'ensemble_scores' in locals():
            logger.info("Fitting ensemble model...")
            self.models['ensemble'].fit(ensemble_scores)
        
        # Fit the score scaler on the ensemble scores
        if 'ensemble_scores' in locals():
            self.score_scaler.fit(ensemble_scores)
        
        self.fitted_ = True
        logger.info("All models fitted successfully.")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies using the ensemble model.
        
        Args:
            X: Input features
            
        Returns:
            Array of predictions (-1 for anomalies, 1 for normal)
        """
        if not self.fitted_:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Get anomaly scores
        scores = self.score_samples(X)
        
        # Convert scores to predictions (-1 for anomalies, 1 for normal)
        threshold = np.percentile(scores, 100 * (1 - self.contamination))
        return np.where(scores >= threshold, -1, 1)
    
    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute the anomaly scores for each sample.
        
        Args:
            X: Input features
            
        Returns:
            Array of anomaly scores (higher = more anomalous)
        """
        if not self.fitted_:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Prepare the features
        feature_sets = self._prepare_features(X)
        
        # Get scores from each model
        all_scores = []
        for model_type, model in self.models.items():
            if model_type == 'ensemble':
                continue

            if model_type not in feature_sets:
                continue

            scaler = self.scalers.get(model_type)
            if scaler is None:
                logger.warning(f"No scaler found for {model_type} model; skipping score computation.")
                continue

            # Scale the features
            X_scaled = scaler.transform(feature_sets[model_type])

            # Get anomaly scores (convert to positive, higher = more anomalous)
            scores = -model.score_samples(X_scaled)
            all_scores.append(scores)
        
        # Stack scores for the ensemble model
        if all_scores:
            ensemble_features = np.column_stack(all_scores)
            
            # Scale the scores to [0, 1]
            ensemble_features_scaled = self.score_scaler.transform(ensemble_features)
            
            # Get final scores from the ensemble model
            final_scores = -self.models['ensemble'].score_samples(ensemble_features_scaled)
            
            # Scale final scores to [0, 1]
            final_scores = (final_scores - final_scores.min()) / (final_scores.max() - final_scores.min())
            
            return final_scores
        
        return np.zeros(len(X))
    
    def get_risk_levels(self, scores: np.ndarray) -> np.ndarray:
        """
        Convert anomaly scores to risk levels.
        
        Args:
            scores: Array of anomaly scores
            
        Returns:
            Array of risk levels ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')
        """
        # Define thresholds for different risk levels
        thresholds = {
            'CRITICAL': 0.9,   # Top 10% of anomalies
            'HIGH': 0.75,      # Next 15%
            'MEDIUM': 0.5,     # Next 25%
            'LOW': 0.0        # Bottom 50%
        }
        
        # Initialize with 'LOW' risk
        risk_levels = np.full_like(scores, 'LOW', dtype='<U8')
        
        # Set risk levels based on thresholds
        risk_levels[scores >= np.percentile(scores, 100 * thresholds['CRITICAL'])] = 'CRITICAL'
        risk_levels[(scores >= np.percentile(scores, 100 * thresholds['HIGH'])) & 
                   (scores < np.percentile(scores, 100 * thresholds['CRITICAL']))] = 'HIGH'
        risk_levels[(scores >= np.percentile(scores, 100 * thresholds['MEDIUM'])) & 
                   (scores < np.percentile(scores, 100 * thresholds['HIGH']))] = 'MEDIUM'
        
        return risk_levels
    
    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.fitted_:
            raise RuntimeError("Model not fitted. Cannot save an unfitted model.")
        
        # Create directory if it doesn't exist
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        joblib.dump({
            'models': self.models,
            'feature_names': self.feature_names_,
            'scalers': self.scalers,
            'score_scaler': self.score_scaler,
            'contamination': self.contamination,
            'random_state': self.random_state,
            'feature_groups': self.feature_groups
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> 'AnomalyDetector':
        """
        Load a trained model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded AnomalyDetector instance
        """
        # Load the model data
        model_data = joblib.load(filepath)
        
        # Create a new instance
        detector = cls(
            contamination=model_data['contamination'],
            random_state=model_data['random_state']
        )
        
        # Set the model attributes
        detector.models = model_data['models']
        detector.feature_names_ = model_data['feature_names']
        detector.scalers = model_data.get('scalers')
        if detector.scalers is None:
            legacy_scaler = model_data.get('scaler')
            if legacy_scaler is not None:
                detector.scalers = {
                    model_type: legacy_scaler
                    for model_type in detector.models
                    if model_type != 'ensemble'
                }
                logger.warning(
                    "Loaded legacy model with shared scaler. Consider retraining to update scalers."
                )
            else:
                detector.scalers = {
                    model_type: StandardScaler()
                    for model_type in detector.models
                    if model_type != 'ensemble'
                }
        detector.score_scaler = model_data['score_scaler']
        detector.feature_groups = model_data['feature_groups']
        detector.fitted_ = True
        
        logger.info(f"Model loaded from {filepath}")
        return detector


def main():
    """Example usage of the AnomalyDetector class."""
    import pandas as pd
    from pathlib import Path
    from sklearn.model_selection import train_test_split
    
    # Example usage
    data_dir = Path(__file__).parent.parent.parent  # Adjust based on your directory structure
    output_dir = data_dir / 'models'
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Load and process the data
        from data_processor import DataProcessor
        from feature_engineer import FeatureEngineer
        
        processor = DataProcessor(data_dir)
        processor.load_data()
        processor.clean_data()
        products, transactions, _ = processor.get_processed_data()
        
        # Create features
        feature_engineer = FeatureEngineer(transactions, products)
        features = feature_engineer.create_all_features()
        
        # Get feature columns
        feature_cols = feature_engineer.get_feature_columns()
        X = features[feature_cols].fillna(0)  # Handle any remaining NaNs
        
        # Split into train and test sets
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
        
        # Initialize and train the anomaly detector
        detector = AnomalyDetector(contamination=0.05, random_state=42)
        detector.fit(X_train)
        
        # Make predictions on test data
        scores = detector.score_samples(X_test)
        predictions = detector.predict(X_test)
        risk_levels = detector.get_risk_levels(scores)
        
        # Print some results
        results = pd.DataFrame({
            'score': scores,
            'prediction': predictions,
            'risk_level': risk_levels
        })
        
        print("\nAnomaly detection results:")
        print(results.head())
        print("\nRisk level distribution:")
        print(results['risk_level'].value_counts())
        
        # Save the model
        model_path = output_dir / 'inventory_anomaly_detector.joblib'
        detector.save_model(model_path)
        
        # Example of loading the model
        # loaded_detector = AnomalyDetector.load_model(model_path)
        
        print("\nAnomaly detection completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()
