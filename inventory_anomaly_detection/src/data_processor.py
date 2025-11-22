"""
Data processing module for the Inventory Anomaly Detection system.
Handles loading, cleaning, and preparing the inventory data for analysis.
"""

import os
import pandas as pd
from typing import Tuple, Dict, Optional
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Handles data loading, cleaning, and preparation for the anomaly detection system.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the DataProcessor with the directory containing the data files.
        
        Args:
            data_dir: Path to the directory containing the data files
        """
        self.data_dir = Path(data_dir)
        self.products_df = None
        self.transactions_df = None
        self.vendor_prices_df = None
        
    def load_data(self) -> None:
        """Load all data files into DataFrames."""
        try:
            # Load products data
            products_path = self.data_dir / 'ml_data_products.csv'
            self.products_df = pd.read_csv(products_path)
            logger.info(f"Loaded products data with {len(self.products_df)} records")
            
            # Load transactions data
            transactions_path = self.data_dir / 'ml_data_transactions.csv'
            self.transactions_df = pd.read_csv(transactions_path, parse_dates=['date'])
            logger.info(f"Loaded transactions data with {len(self.transactions_df)} records")
            
            # Load vendor prices data
            vendor_prices_path = self.data_dir / 'ml_data_vendor_prices.csv'
            self.vendor_prices_df = pd.read_csv(vendor_prices_path, parse_dates=['effective_date'])
            logger.info(f"Loaded vendor prices data with {len(self.vendor_prices_df)} records")
            
        except FileNotFoundError as e:
            logger.error(f"Error loading data files: {e}")
            raise
    
    def clean_data(self) -> None:
        """Clean and preprocess the loaded data."""
        if self.transactions_df is None or self.products_df is None or self.vendor_prices_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Clean transactions data
        self._clean_transactions()
        
        # Clean products data
        self._clean_products()
        
        # Clean vendor prices data
        self._clean_vendor_prices()
        
        logger.info("Data cleaning completed successfully")
    
    def _clean_transactions(self) -> None:
        """Clean the transactions DataFrame."""
        # Convert date to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(self.transactions_df['date']):
            self.transactions_df['date'] = pd.to_datetime(self.transactions_df['date'])
        
        # Ensure quantity_change is numeric
        self.transactions_df['quantity_change'] = pd.to_numeric(
            self.transactions_df['quantity_change'], errors='coerce'
        )
        
        # Drop rows with missing critical values
        initial_count = len(self.transactions_df)
        self.transactions_df.dropna(
            subset=['date', 'product_sku', 'transaction_type', 'quantity_change'],
            inplace=True
        )
        
        if len(self.transactions_df) < initial_count:
            logger.warning(
                f"Dropped {initial_count - len(self.transactions_df)} rows with missing values "
                "from transactions data"
            )
    
    def _clean_products(self) -> None:
        """Clean the products DataFrame."""
        # Ensure numeric columns are properly typed
        numeric_cols = ['unit_cost', 'low_stock_threshold', 'reorder_quantity', 'lead_time_days']
        for col in numeric_cols:
            if col in self.products_df.columns:
                self.products_df[col] = pd.to_numeric(
                    self.products_df[col], errors='coerce'
                )
        
        # Drop duplicates based on SKU
        initial_count = len(self.products_df)
        self.products_df.drop_duplicates(subset=['sku'], keep='first', inplace=True)
        
        if len(self.products_df) < initial_count:
            logger.warning(
                f"Dropped {initial_count - len(self.products_df)} duplicate product SKUs"
            )
    
    def _clean_vendor_prices(self) -> None:
        """Clean the vendor prices DataFrame."""
        # Ensure numeric columns are properly typed
        numeric_cols = ['price', 'vendor_reliability', 'product_unit_cost']
        for col in numeric_cols:
            if col in self.vendor_prices_df.columns:
                self.vendor_prices_df[col] = pd.to_numeric(
                    self.vendor_prices_df[col], errors='coerce'
                )
        
        # Ensure date is datetime
        if 'effective_date' in self.vendor_prices_df.columns:
            self.vendor_prices_df['effective_date'] = pd.to_datetime(
                self.vendor_prices_df['effective_date'], errors='coerce'
            )
    
    def get_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get the processed DataFrames.
        
        Returns:
            A tuple containing (products_df, transactions_df, vendor_prices_df)
        """
        if self.products_df is None or self.transactions_df is None or self.vendor_prices_df is None:
            raise ValueError("Data not loaded and processed. Call load_data() and clean_data() first.")
        
        return self.products_df, self.transactions_df, self.vendor_prices_df
    
    def get_merged_data(self) -> pd.DataFrame:
        """
        Merge all data sources into a single DataFrame for analysis.
        
        Returns:
            A merged DataFrame with all relevant information
        """
        if self.products_df is None or self.transactions_df is None or self.vendor_prices_df is None:
            raise ValueError("Data not loaded and processed. Call load_data() and clean_data() first.")
        
        # Merge transactions with product information
        merged_df = pd.merge(
            self.transactions_df,
            self.products_df,
            left_on='product_sku',
            right_on='sku',
            how='left'
        )
        
        # Get the most recent vendor price for each product at the time of transaction
        # This is a simplified approach - in a production system, you'd want to use the 
        # price that was effective at the time of the transaction
        latest_prices = self.vendor_prices_df.sort_values('effective_date').groupby('product_sku').last().reset_index()
        
        # Merge with vendor prices
        merged_df = pd.merge(
            merged_df,
            latest_prices[['product_sku', 'price', 'vendor_reliability']],
            left_on='product_sku',
            right_on='product_sku',
            how='left'
        )
        
        # Calculate transaction value
        merged_df['transaction_value'] = merged_df['quantity_change'].abs() * merged_df['unit_cost']
        
        return merged_df


def main():
    """Example usage of the DataProcessor class."""
    # Example usage
    data_dir = Path(__file__).parent.parent.parent  # Adjust based on your directory structure
    processor = DataProcessor(data_dir)
    
    try:
        # Load and process the data
        processor.load_data()
        processor.clean_data()
        
        # Get the processed data
        products, transactions, vendor_prices = processor.get_processed_data()
        
        # Get the merged dataset
        merged_data = processor.get_merged_data()
        
        # Print some basic info
        print("\nSample of merged data:")
        print(merged_data.head())
        
        print("\nData processing completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()
