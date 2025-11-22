import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
import joblib

def train_vendor_model():
    print("Loading vendor data...")
    df = pd.read_csv('augmented_vendor_history.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['vendor_id', 'product_id', 'date'])

    print("Feature Engineering...")
    
    # 1. Date Features
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    
    # 2. Lags (Price history)
    # Group by Vendor AND Product
    g = df.groupby(['vendor_id', 'product_id'])
    df['price_lag_1m'] = g['purchase_price'].shift(1)
    df['price_lag_3m'] = g['purchase_price'].shift(3)
    df['reliability_lag_1m'] = g['reliability_score'].shift(1)
    
    # 3. Rolling Stats
    df['rolling_mean_price_3m'] = g['purchase_price'].transform(lambda x: x.shift(1).rolling(3).mean())
    df['rolling_std_price_3m'] = g['purchase_price'].transform(lambda x: x.shift(1).rolling(3).std())
    
    # 4. Market Features
    # Market index is already in dataset, let's add lags for it too
    df['market_index_lag_1m'] = g['market_index'].shift(1)
    
    # 5. Trends
    df['price_trend_3m'] = (df['price_lag_1m'] - df['price_lag_3m']) / (df['price_lag_3m'] + 1e-6)
    
    # Drop NaNs
    df = df.dropna()
    
    # Encoders
    le_vendor = LabelEncoder()
    df['vendor_encoded'] = le_vendor.fit_transform(df['vendor_id'])
    
    le_product = LabelEncoder()
    df['product_encoded'] = le_product.fit_transform(df['product_id'])
    
    features = [
        'month', 'year', 'quarter',
        'price_lag_1m', 'price_lag_3m',
        'reliability_lag_1m', 'reliability_score', # Current reliability might be known or estimated? Let's use lag if predicting future. 
        # But prompt says "Features: vendor_reliability". If we are predicting price for a PO, we know current vendor reliability score.
        'rolling_mean_price_3m', 'rolling_std_price_3m',
        'market_index', 'market_index_lag_1m',
        'price_trend_3m',
        'vendor_encoded', 'product_encoded',
        'lead_time_days' # Known contract term usually
    ]
    
    X = df[features]
    y = df['purchase_price']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training Random Forest on {len(X_train)} samples...")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    print("Evaluating...")
    preds = model.predict(X_test)
    
    r2 = r2_score(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds)
    
    print(f"R²: {r2:.4f}")
    print(f"MAPE: {mape:.4%}")
    
    if r2 > 0.75 and mape < 0.0286:
        print("SUCCESS: Targets met (R² > 0.75, MAPE < 2.86%)")
    else:
        print("WARNING: Targets not met.")

    joblib.dump(model, 'ml_models/vendor_price_model.pkl')
    joblib.dump(le_vendor, 'ml_models/vendor_encoder.pkl')
    joblib.dump(le_product, 'ml_models/product_encoder_vendor.pkl')
    print("Model saved.")

if __name__ == "__main__":
    train_vendor_model()
