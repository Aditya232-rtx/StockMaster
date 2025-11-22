import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def augment_data():
    print("Loading transaction data...")
    try:
        transactions = pd.read_csv('ml_data_transactions.csv')
        products = pd.read_csv('ml_data_products.csv')
        vendor_prices = pd.read_csv('ml_data_vendor_prices.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # --- 1. Enhanced Sales History ---
    print("Generating enhanced sales history (with Price & Promos)...")
    
    # Analyze base demand
    delivery_data = transactions[transactions['transaction_type'] == 'DELIVERY']
    product_stats = {}
    for sku in products['sku'].unique():
        sku_data = delivery_data[delivery_data['product_sku'] == sku]
        if not sku_data.empty:
            qty = sku_data['quantity_change'].abs()
            mean_qty = qty.mean()
            std_qty = qty.std() if len(qty) > 1 else 0
            if pd.isna(std_qty): std_qty = 0
        else:
            mean_qty = 10
            std_qty = 3
        product_stats[sku] = {'mean': mean_qty, 'std': std_qty}

    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    sales_data = []
    
    for sku in products['sku'].unique():
        stats = product_stats.get(sku)
        base_price = products[products['sku'] == sku]['unit_cost'].values[0] * 1.5 # Margin
        
        for date in date_range:
            # Seasonality
            seasonality = 1.0
            if date.month in [10, 11, 12]: seasonality = 1.3
            elif date.month in [1, 2]: seasonality = 0.8
            
            # Weekend bump
            if date.weekday() >= 5: seasonality *= 1.1

            # Promotions
            is_promo = 0
            promo_discount = 0.0
            # Random 5% chance of promo, higher chance in Q4
            promo_prob = 0.1 if date.month in [11, 12] else 0.05
            if np.random.random() < promo_prob:
                is_promo = 1
                promo_discount = np.random.choice([0.1, 0.2, 0.3])
            
            current_price = base_price * (1 - promo_discount)
            
            # Elasticity: Lower price -> Higher demand
            elasticity_factor = 1.0 + (promo_discount * 2.0) # Simple elasticity model
            
            mu = stats['mean'] * seasonality * elasticity_factor
            sigma = stats['std']
            
            demand = np.random.normal(mu, sigma)
            demand = max(0, int(round(demand)))
            
            sales_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'product_id': sku,
                'quantity_sold': demand,
                'price': round(current_price, 2),
                'is_promo': is_promo,
                'promo_discount': promo_discount
            })

    df_sales = pd.DataFrame(sales_data)
    df_sales.to_csv('augmented_sales_history.csv', index=False)
    print(f"Saved augmented_sales_history.csv. Rows: {len(df_sales)}")

    # --- 2. Enhanced Vendor History ---
    print("Generating enhanced vendor history...")
    vendor_data = []
    
    # Get unique vendor-product pairs
    vp_pairs = vendor_prices[['vendor_code', 'product_sku', 'price', 'vendor_reliability']].drop_duplicates()
    
    # Generate monthly data points for the last 2 years
    # We want to predict purchase_price
    
    months = pd.date_range(start=start_date, end=end_date, freq='MS') # Month Start
    
    for _, row in vp_pairs.iterrows():
        vendor = row['vendor_code']
        sku = row['product_sku']
        base_cost = row['price']
        base_reliability = row['vendor_reliability']
        
        # Market trend for this product material
        market_trend = np.random.uniform(0.95, 1.05) # Initial random market condition
        
        for date in months:
            # Slowly drifting market trend
            market_trend += np.random.normal(0, 0.01)
            
            # Vendor specific fluctuation
            vendor_noise = np.random.normal(0, base_cost * 0.02)
            
            purchase_price = base_cost * market_trend + vendor_noise
            
            # Reliability fluctuates slightly
            reliability = min(100, max(0, base_reliability + np.random.normal(0, 2)))
            
            vendor_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'vendor_id': vendor,
                'product_id': sku,
                'purchase_price': round(purchase_price, 2),
                'reliability_score': round(reliability, 1),
                'lead_time_days': np.random.randint(3, 14), # Simulated variation
                'market_index': round(market_trend * 100, 2) # Feature for prediction
            })
            
    df_vendor = pd.DataFrame(vendor_data)
    df_vendor.to_csv('augmented_vendor_history.csv', index=False)
    print(f"Saved augmented_vendor_history.csv. Rows: {len(df_vendor)}")

if __name__ == "__main__":
    augment_data()
