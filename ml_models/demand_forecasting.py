import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings

warnings.filterwarnings('ignore')

def train_demand_model():
    print("Loading sales data...")
    df = pd.read_csv('augmented_sales_history.csv') # Path relative to root
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['product_id', 'date'])

    print("Feature Engineering...")
    
    # 1. Temporal Features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # 2. Lag Features (Extensive)
    lags = [1, 2, 3, 7, 14, 21, 28, 30, 60, 90]
    for lag in lags:
        df[f'lag_{lag}d'] = df.groupby('product_id')['quantity_sold'].shift(lag)
        
    # 3. Rolling Statistics
    windows = [7, 14, 30, 60]
    for w in windows:
        df[f'rolling_mean_{w}d'] = df.groupby('product_id')['quantity_sold'].transform(lambda x: x.shift(1).rolling(window=w).mean())
        df[f'rolling_std_{w}d'] = df.groupby('product_id')['quantity_sold'].transform(lambda x: x.shift(1).rolling(window=w).std())
        df[f'rolling_max_{w}d'] = df.groupby('product_id')['quantity_sold'].transform(lambda x: x.shift(1).rolling(window=w).max())

    # 4. Price & Promo Features
    # Lagged price to avoid data leakage if price changes daily based on demand (though usually price is set)
    # But here price is known for the day we are predicting (usually). 
    # If we are forecasting future, we assume we know the planned price.
    df['price_lag_1'] = df.groupby('product_id')['price'].shift(1)
    
    # Interaction
    df['promo_x_weekend'] = df['is_promo'] * df['is_weekend']

    # Drop NaNs from lags
    df = df.dropna()
    
    # Encoding Product ID
    le = LabelEncoder()
    df['product_encoded'] = le.fit_transform(df['product_id'])
    
    features = [
        'day_of_week', 'day_of_month', 'month', 'quarter', 'is_weekend', 'day_of_year',
        'price', 'is_promo', 'promo_discount', 'promo_x_weekend', 'product_encoded'
    ] + [f'lag_{l}d' for l in lags] + \
      [f'rolling_mean_{w}d' for w in windows] + \
      [f'rolling_std_{w}d' for w in windows] + \
      [f'rolling_max_{w}d' for w in windows]

    X = df[features]
    y = df['quantity_sold']
    
    print(f"Training Gradient Boosting Regressor on {len(X)} samples with {len(features)} features...")
    
    # Time Series Cross-Validation
    tscv = TimeSeriesSplit(n_splits=5)
    mae_scores = []
    rmse_scores = []
    
    model = GradientBoostingRegressor(
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=6, 
        subsample=0.8,
        random_state=42
    )

    fold = 1
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        print(f"Fold {fold} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        fold += 1
        
    avg_mae = np.mean(mae_scores)
    print(f"\nAverage MAE: {avg_mae:.4f}")
    print(f"Average RMSE: {np.mean(rmse_scores):.4f}")
    
    if avg_mae < 1.88:
        print("SUCCESS: MAE target (< 1.88) met!")
    else:
        print("WARNING: MAE target not met. Consider tuning hyperparameters.")

    # Retrain on full dataset for final model
    model.fit(X, y)
    joblib.dump(model, 'ml_models/demand_forecasting_model.pkl')
    joblib.dump(le, 'ml_models/product_encoder.pkl')
    print("Final model saved.")

if __name__ == "__main__":
    train_demand_model()
