"""
Feature engineering module for the Inventory Anomaly Detection system.
Creates relevant features for anomaly detection from the processed data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Handles feature engineering for the anomaly detection system.
    Creates temporal, behavioral, and quantitative features from the data.
    """
    
    def __init__(self, transactions_df: pd.DataFrame, products_df: pd.DataFrame):
        """
        Initialize the FeatureEngineer with transactions and products data.
        
        Args:
            transactions_df: DataFrame containing transaction data
            products_df: DataFrame containing product information
        """
        self.transactions = transactions_df.copy()
        self.products = products_df.copy()
        self.features_df = None
        
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features from the transaction data.
        
        Args:
            df: Input DataFrame with transaction data
            
        Returns:
            DataFrame with added temporal features
        """
        logger.info("Creating temporal features...")
        
        # Basic datetime features
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        
        # Time-based features
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        df['is_month_end'] = (df['date'].dt.is_month_end).astype(int)
        df['is_quarter_end'] = (df['date'].dt.is_quarter_end).astype(int)
        df['is_year_end'] = ((df['month'] == 12) & (df['day_of_month'] == 31)).astype(int)
        
        # Time since last transaction by product
        df = df.sort_values(['product_sku', 'date'])
        df['time_since_last_txn'] = df.groupby('product_sku')['date'].diff().dt.total_seconds() / 3600  # in hours
        
        # Rolling time-based features
        grouped_dates = df.groupby('product_sku')
        for window in [7, 30, 90]:  # 1 week, 1 month, 3 months
            window_counts = (
                grouped_dates
                .rolling(window=f'{window}D', on='date', closed='left', min_periods=1)['date']
                .count()
                .reset_index(level=0, drop=True)
            )
            df[f'txn_count_{window}d'] = window_counts.to_numpy()
        
        return df
    
    def create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create behavioral features from the transaction data.
        
        Args:
            df: Input DataFrame with transaction data
            
        Returns:
            DataFrame with added behavioral features
        """
        logger.info("Creating behavioral features...")
        
        # Transaction type counts
        txn_type_dummies = pd.get_dummies(df['transaction_type'], prefix='txn_type')
        df = pd.concat([df, txn_type_dummies], axis=1)
        
        # Location-based features
        location_dummies = pd.get_dummies(df['location'], prefix='loc')
        df = pd.concat([df, location_dummies], axis=1)
        
        # Transaction frequency by product
        df = df.sort_values(['product_sku', 'date'])
        grouped = df.groupby('product_sku')
        rolling_txn_counts = (
            grouped
            .rolling(window='7D', on='date', closed='left', min_periods=1)['date']
            .count()
            .reset_index(level=0, drop=True)
        )
        df['txn_count_7d'] = rolling_txn_counts.to_numpy()
        
        # Average transaction size by product (absolute quantity)
        df['abs_quantity'] = df['quantity_change'].abs()
        avg_txn_size = (
            grouped
            .rolling(window='30D', on='date', closed='left', min_periods=1)['abs_quantity']
            .mean()
            .reset_index(level=0, drop=True)
        )
        df['avg_txn_size_30d'] = avg_txn_size.to_numpy()
        
        # Standard deviation of transaction sizes
        std_txn_size = (
            grouped
            .rolling(window='30D', on='date', closed='left', min_periods=1)['abs_quantity']
            .std()
            .reset_index(level=0, drop=True)
        )
        df['std_txn_size_30d'] = std_txn_size.to_numpy()
        
        # Z-score of current transaction size compared to history
        df['z_score_txn_size'] = (
            (df['quantity_change'].abs() - df['avg_txn_size_30d']) / 
            df['std_txn_size_30d'].replace(0, np.nan)
        ).fillna(0)
        
        return df
    
    def create_quantitative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create quantitative features from the transaction data.
        
        Args:
            df: Input DataFrame with transaction data
            
        Returns:
            DataFrame with added quantitative features
        """
        logger.info("Creating quantitative features...")
        
        # Absolute quantity and value
        df['abs_quantity'] = df['quantity_change'].abs()
        df['transaction_value'] = df['abs_quantity'] * df['unit_cost']
        
        # Ratio to average transaction value for this product
        grouped = df.groupby('product_sku')
        avg_value = (
            grouped
            .rolling(window='30D', on='date', closed='left', min_periods=1)['transaction_value']
            .mean()
            .reset_index(level=0, drop=True)
        )
        df['avg_txn_value_30d'] = avg_value.to_numpy()
        df['value_ratio_to_avg'] = df['transaction_value'] / df['avg_txn_value_30d'].replace(0, np.nan)
        
        # Days of inventory based on average daily usage
        daily_usage = (
            grouped
            .rolling(window='30D', on='date', closed='left', min_periods=1)['abs_quantity']
            .mean()
            .reset_index(level=0, drop=True)
        )
        df['daily_usage_30d'] = daily_usage.to_numpy()
        df['days_of_inventory'] = df['inventory_after_txn'] / df['daily_usage_30d'].replace(0, np.nan)
        
        # Reorder point proximity (how close to reorder point after this transaction)
        df['reorder_point_proximity'] = (
            (df['inventory_after_txn'] - df['low_stock_threshold']) / 
            df['reorder_quantity'].replace(0, np.nan)
        )
        
        # Large transaction flags
        df['is_large_txn'] = (df['abs_quantity'] > df['reorder_quantity'] * 0.5).astype(int)
        df['is_very_large_txn'] = (df['abs_quantity'] > df['reorder_quantity']).astype(int)
        
        return df
    
    def calculate_inventory_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate inventory levels after each transaction.
        
        Args:
            df: Input DataFrame with transaction data
            
        Returns:
            DataFrame with inventory levels calculated
        """
        logger.info("Calculating inventory levels...")
        
        # Sort by product and date
        df = df.sort_values(['product_sku', 'date'])
        
        # Calculate running inventory
        df['inventory_after_txn'] = df.groupby('product_sku')['quantity_change'].cumsum()
        
        # Shift to get inventory before this transaction
        df['inventory_before_txn'] = df.groupby('product_sku')['inventory_after_txn'].shift(1).fillna(0)
        
        # Calculate inventory change
        df['inventory_change'] = df['inventory_after_txn'] - df['inventory_before_txn']
        
        return df
    
    def create_all_features(self) -> pd.DataFrame:
        """
        Create all features for the anomaly detection model.
        
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting feature engineering...")
        
        # Start with a copy of the transactions data
        features_df = self.transactions.copy()
        
        # Calculate inventory levels first
        features_df = self.calculate_inventory_levels(features_df)
        
        # Add product information
        features_df = pd.merge(
            features_df,
            self.products,
            left_on='product_sku',
            right_on='sku',
            how='left'
        )
        
        # Create different types of features
        features_df = self.create_temporal_features(features_df)
        features_df = self.create_behavioral_features(features_df)
        features_df = self.create_quantitative_features(features_df)
        
        # Drop any remaining NaN values that might cause issues
        features_df = features_df.dropna()
        
        # Store the features DataFrame
        self.features_df = features_df
        
        logger.info(f"Feature engineering complete. Created {len(features_df.columns)} features.")
        
        return features_df
    
    def get_feature_columns(self) -> List[str]:
        """
        Get the list of feature column names.
        
        Returns:
            List of feature column names
        """
        if self.features_df is None:
            raise ValueError("Features not created yet. Call create_all_features() first.")
        
        # Exclude non-feature columns
        non_feature_cols = [
            'date', 'product_sku', 'product_name', 'category', 'transaction_type',
            'quantity_change', 'location', 'sku', 'name', 'unit_cost',
            'low_stock_threshold', 'reorder_quantity', 'lead_time_days'
        ]
        
        feature_cols = [col for col in self.features_df.columns if col not in non_feature_cols]
        numeric_feature_cols = [
            col for col in feature_cols if pd.api.types.is_numeric_dtype(self.features_df[col])
        ]
        return numeric_feature_cols


def main():
    """Example usage of the FeatureEngineer class."""
    # Example usage
    from data_processor import DataProcessor
    
    data_dir = Path(__file__).parent.parent.parent  # Adjust based on your directory structure
    
    try:
        # Load and process the data
        processor = DataProcessor(data_dir)
        processor.load_data()
        processor.clean_data()
        products, transactions, _ = processor.get_processed_data()
        
        # Create features
        feature_engineer = FeatureEngineer(transactions, products)
        features = feature_engineer.create_all_features()
        
        # Print some info
        print("\nSample of engineered features:")
        print(features[['product_sku', 'date', 'transaction_type', 'abs_quantity', 
                       'is_weekend', 'is_business_hours', 'z_score_txn_size', 
                       'days_of_inventory']].head())
        
        print("\nFeature engineering completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()
