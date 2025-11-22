import pandas as pd
import numpy as np
import joblib
from datetime import datetime

class ModelTester:
    def __init__(self, model_dir='ml_models'):
        """Initialize the model tester with paths to saved models."""
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.feature_schema = {}
        self.load_models()
    
    def load_models(self):
        """Load the trained models and scalers."""
        try:
            # Load demand forecasting model and scaler
            self.models['demand'] = joblib.load(f"{self.model_dir}/demand_model.pkl")
            self.scalers['demand'] = joblib.load(f"{self.model_dir}/demand_scaler.pkl")
            
            # Load vendor price model and scaler
            self.models['vendor_price'] = joblib.load(f"{self.model_dir}/vendor_price_model.pkl")
            self.scalers['vendor_price'] = joblib.load(f"{self.model_dir}/vendor_price_scaler.pkl")

            # Load reorder optimizer if available
            try:
                self.models['reorder_optimizer'] = joblib.load(f"{self.model_dir}/reorder_optimizer.pkl")
            except FileNotFoundError:
                print("Note: Reorder optimizer not found. Using individual models.")
                
            print("All models and scalers loaded successfully!")

            self._register_feature_schema('demand')
            self._register_feature_schema('vendor_price')

        except Exception as e:
            print(f"Error loading models: {e}")
            raise

    def _register_feature_schema(self, key):
        """Store the expected feature ordering for a model/scaler pair."""
        columns = None

        scaler = self.scalers.get(key)
        if scaler is not None and hasattr(scaler, "feature_names_in_"):
            columns = list(scaler.feature_names_in_)

        if columns is None:
            model = self.models.get(key)
            if model is not None and hasattr(model, "feature_names_in_"):
                columns = list(model.feature_names_in_)

        if columns:
            self.feature_schema[key] = columns

    def _ensure_dataframe(self, data, key):
        """Convert feature payload to DataFrame and align column order."""
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            raise ValueError("Features must be provided as a dict or DataFrame.")

        expected = self.feature_schema.get(key)
        if expected:
            missing = [col for col in expected if col not in df.columns]
            if missing:
                raise ValueError(
                    f"Missing required {key} features: {', '.join(missing)}"
                )
            df = df[expected]

        try:
            df = df.astype(float)
        except ValueError as exc:
            raise ValueError(f"Non-numeric values detected in {key} features.") from exc

        return df

    def prepare_test_data(self, product_data, vendor_data):
        """
        Prepare test data for prediction.

        Args:
            product_data: Dictionary of product features
            vendor_data: Dictionary of vendor features
            
        Returns:
            Dictionary with prepared feature arrays
        """
        # Convert to DataFrame for scaling
        product_features = self._ensure_dataframe(product_data, 'demand')
        vendor_features = self._ensure_dataframe(vendor_data, 'vendor_price')

        # Scale features
        product_scaled = self.scalers['demand'].transform(product_features)
        vendor_scaled = self.scalers['vendor_price'].transform(vendor_features)

        
        return {
            'product_features': product_features,
            'vendor_features': vendor_features,
            'product_scaled': product_scaled,
            'vendor_scaled': vendor_scaled
        }
    
    def predict_demand(self, product_features):
        """Predict demand for a product."""
        if 'demand' not in self.models:
            raise ValueError("Demand forecasting model not found")

        # Scale features if provided as raw dict/DataFrame
        if isinstance(product_features, (dict, pd.DataFrame)):
            df = self._ensure_dataframe(product_features, 'demand')
            product_features = self.scalers['demand'].transform(df)

        return self.models['demand'].predict(product_features)[0]

    def predict_price(self, vendor_features):
        """Predict vendor price."""
        if 'vendor_price' not in self.models:
            raise ValueError("Vendor price prediction model not found")

        # Scale features if provided as raw dict/DataFrame
        if isinstance(vendor_features, (dict, pd.DataFrame)):
            df = self._ensure_dataframe(vendor_features, 'vendor_price')
            vendor_features = self.scalers['vendor_price'].transform(df)

        return self.models['vendor_price'].predict(vendor_features)[0]
    
    def get_inventory_recommendation(self, product_id, product_features, vendor_features):
        """
        Get inventory recommendation using the reorder optimizer or individual models.
        
        Args:
            product_id: Product ID
            product_features: Dictionary of product features
            vendor_features: Dictionary of vendor features
            
        Returns:
            Dictionary with recommendation
        """
        try:
            if 'reorder_optimizer' in self.models:
                # Use the reorder optimizer if available
                return self.models['reorder_optimizer'].get_recommendation(
                    product_features=product_features,
                    vendor_features=vendor_features
                )
            else:
                # Fallback to individual models
                demand = self.predict_demand(product_features)
                price = self.predict_price(vendor_features)
                
                return {
                    'status': 'success',
                    'product_id': product_id,
                    'predicted_demand': round(demand, 2),
                    'predicted_price': round(price, 2),
                    'message': 'Using basic prediction (reorder optimizer not available)'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error generating recommendation: {str(e)}'
            }

def create_sample_data():
    """Create sample test data."""
    # Sample product features (should match training features exactly)
    # These should match the exact feature names used during training
    product_features = {
        'num_transactions': 5,
        'avg_price': 100.0,
        'day_of_week': 2,
        'day_of_month': 15,
        'month': 6,
        'quarter': 2,
        'qty_lag_1': 50.0,
        'qty_lag_3': 45.0,
        'qty_lag_7': 55.0,
        'qty_lag_14': 60.0,
        'rolling_avg_3': 50.0,
        'rolling_std_3': 5.0,
        'rolling_avg_7': 52.0,
        'rolling_std_7': 6.0,
        'rolling_avg_14': 55.0,
        'rolling_std_14': 7.0
    }
    
    # Sample vendor features (should match training features)
    # These should match the exact feature names used during training
    vendor_features = {
        'purchase_price_std': 5.0,
        'purchase_price_min': 80.0,
        'purchase_price_max': 100.0,
        'unit_cost_mean': 70.0,
        'unit_cost_std': 3.0,
        'reliability_score_mean': 0.9,
        'reliability_score_min': 0.8,
        'reliability_score_max': 1.0
    }
    
    return product_features, vendor_features

if __name__ == "__main__":
    print("=== Model Testing ===\n")
    
    try:
        # Initialize tester
        tester = ModelTester()
        
        # Create sample data
        print("Creating sample test data...")
        product_features, vendor_features = create_sample_data()
        
        # Test demand prediction
        print("\n=== Testing Demand Prediction ===")
        demand = tester.predict_demand(product_features)
        print(f"Predicted Demand: {demand:.2f}")
        
        # Test price prediction
        print("\n=== Testing Price Prediction ===")
        price = tester.predict_price(vendor_features)
        print(f"Predicted Price: ${price:.2f}")
        
        # Test inventory recommendation
        print("\n=== Testing Inventory Recommendation ===")
        recommendation = tester.get_inventory_recommendation(
            product_id="TEST-001",
            product_features=product_features,
            vendor_features=vendor_features
        )
        
        print("\nRecommendation:")
        for key, value in recommendation.items():
            print(f"- {key}: {value}")
            
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        import traceback
        traceback.print_exc()
