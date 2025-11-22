# Inventory Optimization ML System

## Overview
This project implements machine learning models for inventory optimization, including demand forecasting and vendor price prediction. The system helps optimize inventory levels, reduce costs, and improve supply chain efficiency.

## Features

### 1. Demand Forecasting
- Predicts future product demand using Gradient Boosting Regressor
- Considers temporal patterns, trends, and seasonality
- Current Performance (MAE: 52.04 ± 5.40, RMSE: 87.64 ± 8.87)

### 2. Vendor Price Prediction
- Predicts optimal vendor prices using Random Forest Regressor
- Considers vendor reliability and historical pricing
- Current Performance (R²: 0.31, RMSE: 87.09 ± 129.88, MAPE: 0.65%)

### 3. Inventory Optimization
- Calculates Economic Order Quantity (EOQ)
- Determines optimal reorder points
- Suggests safety stock levels
- Provides order recommendations

## Project Structure

```
usp_ml_stockmaster/
├── data/                           # Data directory (not in version control)
│   ├── ml_data_transactions.csv    # Transaction history
│   ├── ml_data_products.csv        # Product catalog
│   └── ml_data_vendor_prices.csv   # Vendor pricing data
├── ml_models/                      # Trained models and scalers
│   ├── demand_model.pkl
│   ├── demand_scaler.pkl
│   ├── vendor_price_model.pkl
│   └── vendor_price_scaler.pkl
├── inventory_optimization_pipeline.py  # Main training pipeline
├── test_model.py                   # Model testing script
├── inventory_ml_integration.py     # Integration module for production
└── data_augmentation.py            # Data generation and augmentation
```

## Setup

### Prerequisites
- Python 3.8+
- Required packages (install via `pip install -r requirements.txt`):
  ```
  pandas>=1.3.0
  numpy>=1.21.0
  scikit-learn>=1.0.0
  joblib>=1.1.0
  matplotlib>=3.4.0
  seaborn>=0.11.0
  ```

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd usp_ml_stockmaster
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Training the Models
Run the training pipeline:
```bash
python inventory_optimization_pipeline.py
```

### 2. Testing the Models
Test the trained models with sample data:
```bash
python test_model.py
```

### 3. Integration with Inventory System
Import and use the `InventoryMLIntegration` class in your application:

```python
from inventory_ml_integration import InventoryMLIntegration

# Initialize the ML integration
ml_integration = InventoryMLIntegration(model_dir='ml_models')

# Get inventory recommendation
recommendation = ml_integration.get_inventory_recommendation(
    product_id="PROD-001",
    product_data={
        'unit_cost': 50.0,
        'unit_price': 100.0,
        'holding_cost_rate': 0.2
    },
    vendor_data={
        'reliability_score': 0.9,
        'min_price': 45.0,
        'max_price': 55.0,
        'price_std': 2.5,
        'ordering_cost': 75.0
    },
    current_inventory=25,
    historical_sales=your_sales_dataframe,  # Optional
    lead_time_days=7
)

# Process the recommendation
if recommendation['status'] == 'success':
    print(f"Recommended order quantity: {recommendation['recommended_order_quantity']}")
    print(f"Predicted demand: {recommendation['predicted_demand']}")
    print(f"Suggested reorder point: {recommendation['reorder_point']}")
```

## Model Performance

### Demand Forecasting
- **Model**: Gradient Boosting Regressor
- **MAE**: 52.04 ± 5.40
- **RMSE**: 87.64 ± 8.87
- **Features Used**: 16

### Vendor Price Prediction
- **Model**: Random Forest Regressor
- **R² Score**: 0.31 ± 0.93
- **RMSE**: 87.09 ± 129.88
- **MAPE**: 0.65%
- **Features Used**: 8

## Data Requirements

### Transaction Data (`ml_data_transactions.csv`)
- `transaction_date`: Date of transaction
- `product_sku`: Product identifier
- `quantity`: Quantity sold/returned
- `transaction_type`: Type of transaction (e.g., 'SALE', 'RETURN')
- `unit_price`: Price per unit

### Product Data (`ml_data_products.csv`)
- `product_sku`: Product identifier
- `product_name`: Name of the product
- `category`: Product category
- `unit_cost`: Cost per unit
- `lead_time_days`: Lead time for restocking

### Vendor Price Data (`ml_data_vendor_prices.csv`)
- `vendor_id`: Vendor identifier
- `product_sku`: Product identifier
- `purchase_price`: Price per unit from vendor
- `reliability_score`: Vendor reliability score (0-1)
- `lead_time_days`: Vendor lead time

## API Reference

### `InventoryMLIntegration` Class

#### `__init__(self, model_dir='ml_models')`
Initialize the ML integration with trained models.

#### `get_inventory_recommendation(self, product_id, product_data, vendor_data, current_inventory=0, historical_sales=None, lead_time_days=7, service_level=1.96)`
Get inventory recommendation for a product.

**Parameters:**
- `product_id`: Unique identifier for the product
- `product_data`: Dictionary containing product information
- `vendor_data`: Dictionary containing vendor information
- `current_inventory`: Current inventory level (default: 0)
- `historical_sales`: Optional DataFrame of historical sales data
- `lead_time_days`: Lead time in days for restocking (default: 7)
- `service_level`: Z-score for desired service level (default: 1.96 for 95%)

**Returns:**
Dictionary containing inventory recommendation with the following keys:
- `status`: 'success' or 'error'
- `product_id`: Product identifier
- `current_inventory`: Current inventory level
- `predicted_demand`: Forecasted demand
- `predicted_price`: Expected purchase price
- `safety_stock`: Recommended safety stock
- `reorder_point`: Recommended reorder point
- `recommended_order_quantity`: Suggested order quantity
- `next_review_date`: Suggested next review date

## Maintenance

### Updating Models
To retrain the models with new data:
1. Update the data files in the `data/` directory
2. Run the training pipeline:
   ```bash
   python inventory_optimization_pipeline.py
   ```

### Monitoring Performance
Monitor model performance using the evaluation metrics printed during training. Consider retraining when:
- Demand patterns change significantly
- New products or vendors are introduced
- Model performance degrades below acceptable thresholds

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
