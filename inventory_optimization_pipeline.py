import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class ReorderOptimizer:
    """Class to handle reorder point optimization."""
    
    def __init__(self, demand_model, price_model, service_level=1.96):
        """
        Initialize the reorder optimizer.
        
        Args:
            demand_model: Trained demand forecasting model
            price_model: Trained vendor price prediction model
            service_level: Z-score for desired service level (default: 1.96 for 95%)
        """
        self.demand_model = demand_model
        self.price_model = price_model
        self.service_level = service_level
        
    def calculate_eoq(self, demand_rate, ordering_cost, holding_cost):
        """Calculate Economic Order Quantity (EOQ)."""
        if demand_rate <= 0 or ordering_cost <= 0 or holding_cost <= 0:
            return 0
        return np.sqrt((2 * demand_rate * ordering_cost) / holding_cost)
        
    def calculate_safety_stock(self, lead_time_demand_std, lead_time=1):
        """Calculate safety stock based on lead time demand standard deviation."""
        if lead_time_demand_std <= 0 or lead_time <= 0:
            return 0
        return self.service_level * lead_time_demand_std * np.sqrt(lead_time)
        
    def calculate_reorder_point(self, lead_time_demand, safety_stock):
        """Calculate reorder point."""
        return lead_time_demand + safety_stock
        
    def get_recommendation(self, product_features, vendor_features, 
                         current_inventory=0, lead_time=7, 
                         ordering_cost=50, holding_cost_rate=0.2):
        """
        Get inventory recommendation for a product.
        
        Args:
            product_features: Dictionary of product features for demand forecasting
            vendor_features: Dictionary of vendor features for price prediction
            current_inventory: Current inventory level
            lead_time: Lead time in days
            ordering_cost: Cost per order
            holding_cost_rate: Annual holding cost as a percentage of unit cost
            
        Returns:
            Dictionary with recommendation details
        """
        try:
            # Predict demand
            demand_features = pd.DataFrame([product_features])
            predicted_demand = max(0, self.demand_model.predict(demand_features)[0])
            
            # Predict price
            price_features = pd.DataFrame([vendor_features])
            predicted_price = max(0, self.price_model.predict(price_features)[0])
            
            # Calculate holding cost per unit (assuming price is per unit)
            holding_cost = predicted_price * holding_cost_rate / 365  # Daily holding cost
            
            # Calculate EOQ
            eoq = self.calculate_eoq(predicted_demand, ordering_cost, holding_cost)
            
            # Calculate safety stock (using rolling_std_7 as an estimate of demand variability)
            lead_time_demand_std = product_features.get('rolling_std_7', predicted_demand * 0.3)  # 30% of demand as fallback
            safety_stock = self.calculate_safety_stock(lead_time_demand_std, lead_time)
            
            # Calculate reorder point
            lead_time_demand = predicted_demand * (lead_time / 7)  # Weekly to daily conversion if needed
            reorder_point = self.calculate_reorder_point(lead_time_demand, safety_stock)
            
            # Determine if an order should be placed
            order_quantity = max(0, eoq) if current_inventory < reorder_point else 0
            
            return {
                'predicted_demand': round(predicted_demand, 2),
                'predicted_price': round(predicted_price, 2),
                'economic_order_quantity': round(eoq, 2),
                'safety_stock': round(safety_stock, 2),
                'reorder_point': round(reorder_point, 2),
                'current_inventory': current_inventory,
                'recommended_order_quantity': round(order_quantity, 2),
                'lead_time_days': lead_time,
                'order_total': round(order_quantity * predicted_price, 2)
            }
            
        except Exception as e:
            print(f"Error generating recommendation: {str(e)}")
            return {
                'error': str(e),
                'predicted_demand': 0,
                'predicted_price': 0,
                'economic_order_quantity': 0,
                'safety_stock': 0,
                'reorder_point': 0,
                'current_inventory': current_inventory,
                'recommended_order_quantity': 0,
                'lead_time_days': lead_time,
                'order_total': 0
            }

class InferencePipeline:
    def __init__(self, models, scalers):
        self.models = models
        self.scalers = scalers
        
    def predict_demand(self, product_features):
        """Predict demand for a product."""
        try:
            if 'demand_enhanced' in self.models:
                model = self.models['demand_enhanced']
                scaler = self.scalers.get('demand_enhanced')
            else:
                model = self.models.get('demand')
                scaler = self.scalers.get('demand')
            
            if model is None or scaler is None:
                return None
                
            # Scale features
            features_scaled = scaler.transform([product_features])
            return max(model.predict(features_scaled)[0], 0)
        except Exception as e:
            print(f"Error in demand prediction: {e}")
            return None
            
    def predict_price(self, vendor_features):
        """Predict vendor price."""
        try:
            model = self.models.get('vendor_price')
            scaler = self.scalers.get('vendor_price')
            
            if model is None or scaler is None:
                return None
                
            features_scaled = scaler.transform([vendor_features])
            return max(model.predict(features_scaled)[0], 0.01)
        except Exception as e:
            print(f"Error in price prediction: {e}")
            return 100.0  # Default price
            
    def get_recommendation(self, product_id, product_features, vendor_features):
        """Get complete inventory recommendation."""
        try:
            if 'reorder_optimizer' in self.models:
                # Use the reorder optimizer if available
                optimizer = self.models['reorder_optimizer']
                return optimizer.get_recommendation(
                    product_features=product_features,
                    vendor_features=vendor_features
                )
            
            # Fallback to basic prediction if no optimizer
            demand = self.predict_demand(product_features)
            price = self.predict_price(vendor_features)
            
            if demand is None or price is None:
                return {'status': 'error', 'message': 'Failed to generate prediction'}
                
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
                'message': f'Error in recommendation: {str(e)}'
            }

class InventoryOptimization:
    def __init__(self, data_dir='.'):
        """Initialize the inventory optimization pipeline."""
        self.data_dir = data_dir
        self.models = {}
        self.scalers = {}
        self.features = {}
        self.results = {}
        
    def load_data(self):
        """Load and preprocess the inventory data."""
        print("Loading data...")
        # Load transaction data
        transactions = pd.read_csv(f'{self.data_dir}/ml_data_transactions.csv', parse_dates=['date'])
        transactions.rename(columns={'date': 'transaction_date', 'quantity_change': 'quantity', 'product_sku': 'product_id'}, inplace=True)
        
        # Load vendor data
        vendor_prices = pd.read_csv(f'{self.data_dir}/ml_data_vendor_prices.csv')
        
        # Load product data
        products = pd.read_csv(f'{self.data_dir}/ml_data_products.csv')
        
        return transactions, vendor_prices, products
    
    def prepare_demand_forecasting_data(self, transactions):
        """Prepare data for demand forecasting."""
        print("Preparing demand forecasting data...")
        print(f"Initial transactions shape: {transactions.shape}")
        print(f"Columns in transactions: {transactions.columns.tolist()}")
        print(f"Sample data:\n{transactions.head()}")
        
        # Basic feature engineering
        transactions = transactions.sort_values(['product_id', 'transaction_date'])
        
        # Group by product and date
        daily_sales = transactions.groupby(['product_id', 'transaction_date']).agg(
            quantity_sold=('quantity', 'sum'),
            num_transactions=('transaction_type', 'count')
        ).reset_index()
        
        print(f"\nAfter initial grouping:")
        print(f"Daily sales shape: {daily_sales.shape}")
        print(f"Sample daily sales:\n{daily_sales.head()}")
        
        # Add a dummy price column since it's not in the data
        daily_sales['avg_price'] = 100  # Placeholder price
        
        # Time-based features
        daily_sales['day_of_week'] = daily_sales['transaction_date'].dt.dayofweek
        daily_sales['day_of_month'] = daily_sales['transaction_date'].dt.day
        daily_sales['month'] = daily_sales['transaction_date'].dt.month
        daily_sales['quarter'] = daily_sales['transaction_date'].dt.quarter
        
        # Sort by product and date
        daily_sales = daily_sales.sort_values(['product_id', 'transaction_date'])
        
        # Lag features - use smaller lags to keep more data
        for lag in [1, 3, 7, 14]:  # Reduced from [1, 7, 14, 30]
            daily_sales[f'qty_lag_{lag}'] = daily_sales.groupby('product_id')['quantity_sold'].shift(lag)
        
        # Calculate rolling statistics within each product group
        for window in [3, 7, 14]:  # Reduced window sizes
            daily_sales[f'rolling_avg_{window}'] = daily_sales.groupby('product_id')['quantity_sold'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            daily_sales[f'rolling_std_{window}'] = daily_sales.groupby('product_id')['quantity_sold'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
        
        # Only drop rows where the target is NaN
        df = daily_sales.dropna(subset=['quantity_sold'])
        
        # Fill remaining NaN values with appropriate defaults
        for col in df.columns:
            if col.startswith('qty_lag_'):
                df[col] = df[col].fillna(0)  # Fill lag NaNs with 0 (no previous sales)
            elif col.startswith('rolling_std_'):
                df[col] = df[col].fillna(0)  # Fill std NaNs with 0 (no variation)
            elif col.startswith('rolling_avg_'):
                # Fill avg NaNs with the first non-NaN value in the group
                df[col] = df.groupby('product_id')[col].transform(
                    lambda x: x.fillna(method='bfill').fillna(0)
                )
        
        # Debug print
        print(f"\nAfter feature engineering:")
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Number of unique products: {df['product_id'].nunique()}")
        print(f"Date range: {df['transaction_date'].min()} to {df['transaction_date'].max()}")
        
        # Define features and target
        target = 'quantity_sold'
        features = [col for col in df.columns if col not in [target, 'product_id', 'transaction_date']]
        
        print(f"\nFeatures to use: {features}")
        print(f"Target: {target}")
        
        return df, features, target
    
    def prepare_vendor_price_data(self, transactions, vendor_prices):
        """Prepare data for vendor price prediction."""
        print("Preparing vendor price data...")
        # Merge transactions with vendor prices
        # Prepare vendor data
        vendor_data = vendor_prices.copy()
        vendor_data.rename(columns={
            'product_sku': 'product_id',
            'price': 'purchase_price',
            'vendor_reliability': 'reliability_score',
            'product_unit_cost': 'unit_cost'
        }, inplace=True)
        
        # Feature engineering
        vendor_agg = vendor_data.groupby('product_id').agg({
            'purchase_price': ['mean', 'std', 'min', 'max'],
            'unit_cost': ['mean', 'std'],
            'reliability_score': ['mean', 'min', 'max']
        }).reset_index()
        
        # Flatten multi-index columns
        vendor_agg.columns = ['_'.join(col).strip('_') for col in vendor_agg.columns.values]
        
        # Reliability score is already included in the data as reliability_score_mean
        # from the groupby operation above
        
        # Define features and target
        target = 'purchase_price_mean'
        features = [col for col in vendor_agg.columns if col not in [target, 'product_id']]
        # Filter out any non-numeric columns that might have been created
        features = [f for f in features if not any(x in f for x in ['vendor_', 'product_'])]
        
        return vendor_agg, features, target
    
    def train_demand_forecasting_model(self, X, y):
        """Train the demand forecasting model."""
        print("Training demand forecasting model...")
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['demand'] = scaler
        
        # Initialize and train model
        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Calculate MAE and RMSE scores
        mae_scores = []
        rmse_scores = []
        
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mae_scores.append(mean_absolute_error(y_test, y_pred))
            rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        
        # Convert to numpy arrays for easier calculations
        mae_scores = np.array(mae_scores)
        rmse_scores = np.array(rmse_scores)
        
        # Train final model on all data
        model.fit(X_scaled, y)
        self.models['demand'] = model
        
        # Store results
        self.results['demand_forecasting'] = {
            'mae': mae_scores.mean(),
            'mae_std': mae_scores.std(),
            'rmse': rmse_scores.mean(),
            'rmse_std': rmse_scores.std(),
            'features': X.columns.tolist()
        }
        
        return model, mae_scores.mean()
    
    def train_vendor_price_model(self, X, y):
        """Train the vendor price prediction model."""
        print("Training vendor price prediction model...")
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['vendor_price'] = scaler
        
        # Initialize and train model
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Cross-validation
        r2_scores = []
        rmse_scores = []
        
        for train_idx, test_idx in TimeSeriesSplit(n_splits=5).split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            r2_scores.append(r2_score(y_test, y_pred))
            rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        
        # Convert to numpy arrays for easier calculations
        r2_scores = np.array(r2_scores)
        rmse_scores = np.array(rmse_scores)
        
        # Train final model on all data
        model.fit(X_scaled, y)
        self.models['vendor_price'] = model
        
        # Calculate MAPE on full dataset
        y_pred = model.predict(X_scaled)
        mape = mean_absolute_percentage_error(y, y_pred) * 100
        
        # Store results
        self.results['vendor_price_prediction'] = {
            'r2': r2_scores.mean(),
            'r2_std': r2_scores.std(),
            'rmse': rmse_scores.mean(),
            'rmse_std': rmse_scores.std(),
            'mape': mape,
            'features': X.columns.tolist()
        }
        
        return model, r2_scores.mean(), mape
    
    def train_reorder_optimizer(self):
        """Train the reorder point optimizer."""
        print("Training reorder point optimizer...")
        # This would combine the demand forecasting and vendor price models
        # to calculate optimal reorder points, safety stock, and vendor selection
        # Implementation depends on specific business rules and constraints
        pass
    
    def evaluate_models(self):
        """Evaluate all trained models and print results."""
        print("\n=== Model Evaluation Results ===")
        
        # Demand Forecasting Results
        if 'demand_forecasting' in self.results:
            res = self.results['demand_forecasting']
            print("\nDemand Forecasting (Gradient Boosting):")
            print(f"- MAE: {res['mae']:.4f} ± {res['mae_std']:.4f}")
            print(f"- RMSE: {res['rmse']:.4f} ± {res['rmse_std']:.4f}")
            print(f"- Features used: {len(res['features'])}")
        
        # Vendor Price Prediction Results
        if 'vendor_price_prediction' in self.results:
            res = self.results['vendor_price_prediction']
            print("\nVendor Price Prediction (Random Forest):")
            print(f"- R² Score: {res['r2']:.4f} ± {res['r2_std']:.4f}")
            print(f"- RMSE: {res['rmse']:.4f} ± {res['rmse_std']:.4f}")
            print(f"- MAPE: {res['mape']:.2f}%")
            print(f"- Features used: {len(res['features'])}")
    
    def save_models(self, output_dir='ml_models'):
        """Save trained models and scalers."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, f"{output_dir}/{name}_model.pkl")
        
        # Save scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f"{output_dir}/{name}_scaler.pkl")
        
        # Save feature lists
        for model_name, res in self.results.items():
            with open(f"{output_dir}/{model_name}_features.txt", 'w') as f:
                if 'features' in res:  # Only save if features exist
                    f.write('\n'.join(str(f) for f in res['features']))
        
        print(f"\nModels and artifacts saved to {output_dir}/")
        
    def enhance_demand_forecasting(self, X, y):
        """Enhance the demand forecasting model with additional features and tuning."""
        print("\nEnhancing demand forecasting model...")
        
        # Add more sophisticated features
        if isinstance(X, pd.DataFrame):
            X_enhanced = X.copy()
            # Add interaction terms
            if 'avg_price' in X_enhanced.columns and 'qty_lag_1' in X_enhanced.columns:
                X_enhanced['price_quantity_interaction'] = X_enhanced['avg_price'] * X_enhanced['qty_lag_1']
            # Add day of year seasonality
            if 'day_of_week' in X_enhanced.columns:
                X_enhanced['day_sin'] = np.sin(2 * np.pi * X_enhanced['day_of_week']/7)
                X_enhanced['day_cos'] = np.cos(2 * np.pi * X_enhanced['day_of_week']/7)
        else:
            X_enhanced = X
    
        # Hyperparameter tuning with GridSearchCV
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.8, 1.0]
        }
        
        model = GradientBoostingRegressor(random_state=42)
        tscv = TimeSeriesSplit(n_splits=3)  # Reduced for faster execution
        
        print("Performing grid search for best hyperparameters...")
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_enhanced, y)
        
        # Train final model with best parameters
        best_model = grid_search.best_estimator_
        best_model.fit(X_enhanced, y)
        
        # Cross-validation with best model
        mae_scores = -cross_val_score(
            best_model, X_enhanced, y,
            cv=tscv,
            scoring='neg_mean_absolute_error'
        )
        
        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_enhanced.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Feature Importance:")
            print(feature_importance.head(10).to_string())
        
        # Update model and results
        self.models['demand_enhanced'] = best_model
        self.scalers['demand_enhanced'] = self.scalers.get('demand', StandardScaler())
        
        self.results['demand_forecasting_enhanced'] = {
            'mae': mae_scores.mean(),
            'mae_std': mae_scores.std(),
            'features': list(X_enhanced.columns),
            'best_params': grid_search.best_params_
        }
        
        return best_model, mae_scores.mean()
        
    def implement_reorder_optimizer(self, demand_model, price_model):
        """
        Implement the reorder point optimizer that combines demand and price models.
        
        Args:
            demand_model: Trained demand forecasting model
            price_model: Trained vendor price prediction model
            
        Returns:
            ReorderOptimizer instance
        """
        print("\nImplementing reorder point optimizer...")
        
        # Create and store the optimizer
        self.reorder_optimizer = ReorderOptimizer(demand_model, price_model)
        self.models['reorder_optimizer'] = self.reorder_optimizer
        print("Reorder point optimizer implemented successfully.")
        return self.reorder_optimizer
    
    def create_inference_pipeline(self):
        """Create an inference pipeline for making predictions with trained models."""
        print("\nCreating inference pipeline...")
        
        class InferencePipeline:
            def __init__(self, models, scalers):
                self.models = models
                self.scalers = scalers
                
            def predict_demand(self, product_features):
                """Predict demand for a product."""
                try:
                    if 'demand_enhanced' in self.models:
                        model = self.models['demand_enhanced']
                        scaler = self.scalers.get('demand_enhanced')
                    else:
                        model = self.models.get('demand')
                        scaler = self.scalers.get('demand')
                    
                    if model is None or scaler is None:
                        return None
                        
                    # Scale features
                    features_scaled = scaler.transform([product_features])
                    return max(model.predict(features_scaled)[0], 0)
                except Exception as e:
                    print(f"Error in demand prediction: {e}")
                    return None
                    
            def predict_price(self, vendor_features):
                """Predict vendor price."""
                try:
                    model = self.models.get('vendor_price')
                    scaler = self.scalers.get('vendor_price')
                    
                    if model is None or scaler is None:
                        return None
                        
                    features_scaled = scaler.transform([vendor_features])
                    return max(model.predict(features_scaled)[0], 0.01)
                except Exception as e:
                    print(f"Error in price prediction: {e}")
                    return 100.0  # Default price
                    
            def get_recommendation(self, product_id, product_features, vendor_features):
                """Get complete inventory recommendation."""
                try:
                    # Get demand forecast
                    demand = self.predict_demand(product_features)
                    if demand is None:
                        return {'status': 'error', 'message': 'Failed to predict demand'}
                    
                    # Get price prediction
                    price = self.predict_price(vendor_features)
                    
                    # Get reorder recommendation if available
                    if 'reorder_optimizer' in self.models:
                        return self.models['reorder_optimizer'].optimize_reorder(
                            product_features, 
                            vendor_features
                        )
                    
                    # Fallback basic recommendation
                    return {
                        'status': 'success',
                        'product_id': product_id,
                        'predicted_demand': round(demand, 2),
                        'predicted_price': round(price, 2),
                        'message': 'Reorder optimizer not available, using basic prediction'
                    }
                except Exception as e:
                    return {
                        'status': 'error',
                        'message': f'Error in recommendation: {str(e)}'
                    }
        
        # Initialize and store the pipeline
        self.inference_pipeline = InferencePipeline(self.models, self.scalers)
        print("Inference pipeline created successfully.")
        return self.inference_pipeline

def main():
    # Initialize pipeline
    pipeline = InventoryOptimization()
    
    try:
        print("=== Starting Inventory Optimization Pipeline ===\n")
        
        # Load and prepare data
        print("1. Loading and preparing data...")
        transactions, vendor_prices, products = pipeline.load_data()
        
        # 1. Demand Forecasting
        print("\n2. Training demand forecasting models...")
        df_demand, demand_features, demand_target = pipeline.prepare_demand_forecasting_data(transactions)
        X_demand = df_demand[demand_features]
        y_demand = df_demand[demand_target]
        
        # 2. Vendor Price Prediction
        print("\n3. Training vendor price prediction model...")
        df_vendor, vendor_features, vendor_target = pipeline.prepare_vendor_price_data(transactions, vendor_prices)
        X_vendor = df_vendor[vendor_features]
        y_vendor = df_vendor[vendor_target]
        
        # Train base models
        print("\n4. Training base models...")
        demand_model, _ = pipeline.train_demand_forecasting_model(X_demand, y_demand)
        price_model, _, _ = pipeline.train_vendor_price_model(X_vendor, y_vendor)
        
        # Step 1: Enhance demand forecasting
        print("\n5. Enhancing demand forecasting...")
        enhanced_model, enhanced_mae = pipeline.enhance_demand_forecasting(X_demand, y_demand)
        
        # Step 2: Implement reorder point optimizer
        print("\n6. Implementing reorder point optimizer...")
        reorder_optimizer = pipeline.implement_reorder_optimizer(enhanced_model, price_model)
        
        # Step 3: Create inference pipeline
        print("\n7. Creating inference pipeline...")
        inference_pipeline = pipeline.create_inference_pipeline()
        
        # Save all models and artifacts
        print("\n8. Saving models and artifacts...")
        pipeline.save_models()
        
        # Evaluate and print results
        print("\n=== Model Evaluation ===")
        pipeline.evaluate_models()
        
        # Example usage with sample data
        if len(X_demand) > 0 and len(X_vendor) > 0:
            print("\n=== Sample Recommendation ===")
            # Get sample product and vendor features
            sample_product = X_demand.iloc[0].to_dict()
            sample_vendor = {k: 0 for k in X_vendor.columns}  # Initialize with zeros
            # Set some default values for required features
            for col in ['purchase_price_mean', 'unit_cost_mean', 'lead_time_days_mean']:
                if col in X_vendor.columns:
                    sample_vendor[col] = X_vendor[col].median()
            
            # Get recommendation
            recommendation = inference_pipeline.get_recommendation(
                product_id=df_demand['product_id'].iloc[0],
                product_features=sample_product,
                vendor_features=sample_vendor
            )
            
            print("\nSample Inventory Recommendation:")
            for key, value in recommendation.items():
                print(f"- {key}: {value}")
        
        print("\n=== Pipeline Execution Complete ===")
        print("All models trained and saved successfully!")
        
    except Exception as e:
        print(f"\n=== Error in Pipeline ===")
        print(f"Error: {str(e)}")
        print("\nStack trace:")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
