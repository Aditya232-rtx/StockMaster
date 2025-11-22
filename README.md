# 📦 StockMaster - Inventory Management System

A comprehensive, modern inventory management system built with Django REST Framework and vanilla JavaScript. StockMaster provides real-time inventory tracking, warehouse operations management, and advanced analytics for efficient stock control.

![StockMaster Dashboard](https://img.shields.io/badge/Status-Active-success)
![Django](https://img.shields.io/badge/Django-4.2+-green)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🌟 Features

### Core Functionality
- **📊 Real-time Dashboard** - Comprehensive overview of inventory metrics and KPIs
- **📦 Stock Management** - Track products across multiple locations and warehouses
- **🔄 Operations Management** - Handle receipts, deliveries, adjustments, and internal transfers
- **📈 Move History** - Complete audit trail of all inventory movements
- **🏢 Multi-location Support** - Manage inventory across multiple warehouses and locations
- **👥 Vendor Management** - Track vendors, purchase orders, and price history

### Advanced Features
- **🔐 Secure Authentication** - User authentication with Django OTP support
- **🌐 RESTful API** - Complete REST API with Django REST Framework
- **📱 Responsive Design** - Mobile-friendly interface with modern UI/UX
- **🔍 Advanced Filtering** - Search and filter across all inventory data
- **📄 API Documentation** - Auto-generated API docs with drf-spectacular
- **🔗 Blockchain Integration** - Provenance tracking with blockchain hash support

### 🤖 Machine Learning Features
- **📊 Demand Forecasting** - AI-powered demand prediction using Gradient Boosting
- **💰 Vendor Price Prediction** - Intelligent price forecasting with Random Forest
- **📈 Inventory Optimization** - Automated EOQ, reorder points, and safety stock calculations
- **🎯 Smart Recommendations** - Data-driven order quantity suggestions
- **⚡ Real-time Analytics** - ML-powered insights for inventory decisions

## 🤖 Machine Learning System

StockMaster includes a sophisticated ML-powered inventory optimization system that helps reduce costs, prevent stockouts, and improve supply chain efficiency.

### ML Capabilities

#### 1. Demand Forecasting
Predicts future product demand using **Gradient Boosting Regressor** with advanced temporal pattern recognition.

**Performance Metrics:**
- MAE (Mean Absolute Error): 52.04 ± 5.40
- RMSE (Root Mean Square Error): 87.64 ± 8.87
- Features Used: 16 temporal and product-specific features

**Key Features:**
- Considers seasonal patterns and trends
- Accounts for historical sales data
- Adjusts for product lifecycle stages
- Handles multiple time horizons

#### 2. Vendor Price Prediction
Optimizes vendor selection and pricing using **Random Forest Regressor**.

**Performance Metrics:**
- R² Score: 0.31 ± 0.93
- RMSE: 87.09 ± 129.88
- MAPE (Mean Absolute Percentage Error): 0.65%
- Features Used: 8 vendor and product features

**Key Features:**
- Vendor reliability scoring
- Historical price trend analysis
- Multi-vendor comparison
- Price volatility assessment

#### 3. Inventory Optimization Engine
Calculates optimal inventory parameters using proven operations research methods.

**Capabilities:**
- **Economic Order Quantity (EOQ)** - Minimizes total inventory costs
- **Reorder Point Calculation** - Prevents stockouts with configurable service levels
- **Safety Stock Optimization** - Balances holding costs vs. stockout risk
- **Order Recommendations** - Actionable suggestions with timing and quantities

### ML Project Structure

```
usp_ml_stockmaster/
├── data/                              # Training data (gitignored)
│   ├── ml_data_transactions.csv       # Historical transaction data
│   ├── ml_data_products.csv           # Product catalog
│   └── ml_data_vendor_prices.csv      # Vendor pricing history
├── ml_models/                         # Trained models and scalers
│   ├── demand_model.pkl               # Demand forecasting model
│   ├── demand_scaler.pkl              # Feature scaler for demand
│   ├── vendor_price_model.pkl         # Price prediction model
│   └── vendor_price_scaler.pkl        # Feature scaler for prices
├── inventory_optimization_pipeline.py # Model training pipeline
├── test_model.py                      # Model testing and validation
├── inventory_ml_integration.py        # Production integration module
└── data_augmentation.py               # Data generation utilities
```

### Using the ML System

#### Training Models

Train or retrain the ML models with your data:

```bash
cd usp_ml_stockmaster
python inventory_optimization_pipeline.py
```

This will:
1. Load historical data from CSV files
2. Engineer features for both models
3. Train demand forecasting and price prediction models
4. Evaluate performance with cross-validation
5. Save trained models to `ml_models/` directory

#### Testing Models

Validate model performance:

```bash
python test_model.py
```

#### Integration in Your Application

Use the ML system in your Django application:

```python
from inventory_ml_integration import InventoryMLIntegration

# Initialize ML integration
ml_integration = InventoryMLIntegration(model_dir='ml_models')

# Get comprehensive inventory recommendation
recommendation = ml_integration.get_inventory_recommendation(
    product_id="PROD-001",
    product_data={
        'unit_cost': 50.0,
        'unit_price': 100.0,
        'holding_cost_rate': 0.2  # 20% annual holding cost
    },
    vendor_data={
        'reliability_score': 0.9,      # 90% reliability
        'min_price': 45.0,
        'max_price': 55.0,
        'price_std': 2.5,
        'ordering_cost': 75.0          # Fixed cost per order
    },
    current_inventory=25,
    historical_sales=sales_dataframe,  # Optional pandas DataFrame
    lead_time_days=7,
    service_level=1.96                 # 95% service level (Z-score)
)

# Process recommendation
if recommendation['status'] == 'success':
    print(f"📦 Recommended Order Quantity: {recommendation['recommended_order_quantity']}")
    print(f"📊 Predicted Demand: {recommendation['predicted_demand']}")
    print(f"🎯 Reorder Point: {recommendation['reorder_point']}")
    print(f"🛡️ Safety Stock: {recommendation['safety_stock']}")
    print(f"💰 Predicted Price: ${recommendation['predicted_price']:.2f}")
    print(f"📅 Next Review: {recommendation['next_review_date']}")
```

### ML Data Requirements

#### Transaction Data (`ml_data_transactions.csv`)
```csv
transaction_date,product_sku,quantity,transaction_type,unit_price
2024-01-15,PROD-001,50,SALE,100.00
2024-01-16,PROD-001,-5,RETURN,100.00
```

**Required Columns:**
- `transaction_date`: Date of transaction (YYYY-MM-DD)
- `product_sku`: Unique product identifier
- `quantity`: Quantity sold (positive) or returned (negative)
- `transaction_type`: Type (SALE, RETURN, etc.)
- `unit_price`: Price per unit

#### Product Data (`ml_data_products.csv`)
```csv
product_sku,product_name,category,unit_cost,lead_time_days
PROD-001,Widget A,Electronics,50.00,7
```

**Required Columns:**
- `product_sku`: Unique product identifier
- `product_name`: Product name
- `category`: Product category
- `unit_cost`: Cost per unit
- `lead_time_days`: Restocking lead time

#### Vendor Price Data (`ml_data_vendor_prices.csv`)
```csv
vendor_id,product_sku,purchase_price,reliability_score,lead_time_days
VEND-001,PROD-001,48.50,0.95,5
```

**Required Columns:**
- `vendor_id`: Unique vendor identifier
- `product_sku`: Product identifier
- `purchase_price`: Vendor's price per unit
- `reliability_score`: Reliability (0-1 scale)
- `lead_time_days`: Vendor lead time

### ML API Reference

#### `InventoryMLIntegration` Class

##### Constructor
```python
InventoryMLIntegration(model_dir='ml_models')
```
Initialize with path to trained models directory.

##### `get_inventory_recommendation()`
```python
get_inventory_recommendation(
    product_id: str,
    product_data: dict,
    vendor_data: dict,
    current_inventory: float = 0,
    historical_sales: pd.DataFrame = None,
    lead_time_days: int = 7,
    service_level: float = 1.96
) -> dict
```

**Parameters:**
- `product_id`: Unique product identifier
- `product_data`: Dict with `unit_cost`, `unit_price`, `holding_cost_rate`
- `vendor_data`: Dict with `reliability_score`, `min_price`, `max_price`, `price_std`, `ordering_cost`
- `current_inventory`: Current stock level (default: 0)
- `historical_sales`: Optional DataFrame with sales history
- `lead_time_days`: Restocking lead time (default: 7)
- `service_level`: Z-score for service level (default: 1.96 = 95%)

**Returns:**
```python
{
    'status': 'success',
    'product_id': 'PROD-001',
    'current_inventory': 25,
    'predicted_demand': 150.5,
    'predicted_price': 48.75,
    'safety_stock': 30,
    'reorder_point': 80,
    'recommended_order_quantity': 200,
    'next_review_date': '2024-02-01',
    'confidence_interval': {
        'demand_lower': 120,
        'demand_upper': 180
    }
}
```

### ML Model Maintenance

#### When to Retrain Models

Retrain your models when:
- ✅ New products are added to the catalog
- ✅ Significant changes in demand patterns occur
- ✅ New vendors are onboarded
- ✅ Seasonal patterns shift
- ✅ Model performance degrades (monitor MAE/RMSE)
- ✅ Every 3-6 months as a best practice

#### Monitoring Performance

Track these metrics to ensure model health:
- **Demand Forecasting**: MAE, RMSE, forecast bias
- **Price Prediction**: R², MAPE, prediction intervals
- **Business Metrics**: Stockout rate, inventory turnover, holding costs

#### Updating Training Data

1. Export new transaction data from your database
2. Update CSV files in the `data/` directory
3. Run the training pipeline
4. Validate new model performance
5. Deploy updated models to production

### ML Dependencies

Additional Python packages required for ML features:

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
joblib>=1.1.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

Install ML dependencies:
```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn
```


## 🏗️ Architecture

### Backend (Django)
```
StockMaster/
├── config/                 # Django project configuration
│   ├── settings.py        # Project settings
│   ├── urls.py            # URL routing
│   └── wsgi.py            # WSGI configuration
├── inventory_core/        # Main inventory application
│   ├── models.py          # Database models
│   ├── serializers.py     # DRF serializers
│   ├── views.py           # API views
│   ├── admin.py           # Django admin configuration
│   └── migrations/        # Database migrations
├── manage.py              # Django management script
└── requirements.txt       # Python dependencies
```

### Frontend (Vanilla JavaScript)
```
frontend/
├── index.html             # Landing/Login page
├── dashboard.html         # Main dashboard
├── operation.html         # Operations management
├── new_delivery.html      # New delivery order form
├── stock.html             # Stock overview
├── history.html           # Movement history
├── signup.html            # User registration
├── api.js                 # API client utilities
└── *.css                  # Styling files
```

## 🗄️ Database Models

### Core Models
- **Product** - Product information with SKU, category, UOM, and ML-related fields
- **Location** - Warehouse and location hierarchy
- **Vendor** - Vendor management with reliability scores
- **StockByLocation** - Real-time stock levels per location

### Movement Models
- **Receipt** - Incoming goods from vendors
- **DeliveryOrder** - Outgoing goods to customers
- **InternalTransfer** - Stock transfers between locations
- **StockAdjustment** - Inventory adjustments with reasons

### Supporting Models
- **PurchaseOrder** - Purchase order management
- **StockTransaction** - Complete transaction history
- **PriceHistory** - Vendor price tracking over time
- **UnitOfMeasure** - Product measurement units

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- PostgreSQL database
- Node.js (for development tools, optional)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Aditya232-rtx/StockMaster.git
   cd StockMaster
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   Create a `.env` file in the project root:
   ```env
   SECRET_KEY=your-secret-key-here
   DEBUG=True
   DATABASE_NAME=stockmaster_db
   DATABASE_USER=your_db_user
   DATABASE_PASSWORD=your_db_password
   DATABASE_HOST=localhost
   DATABASE_PORT=5432
   ```

5. **Set up the database**
   ```bash
   # Create PostgreSQL database
   createdb stockmaster_db

   # Run migrations
   python manage.py migrate

   # Create superuser
   python manage.py createsuperuser
   ```

6. **Load initial data (optional)**
   ```bash
   python manage.py loaddata initial_data
   ```

7. **Run the development server**
   ```bash
   python manage.py runserver
   ```

8. **Access the application**
   - Frontend: `http://localhost:8000/frontend/index.html`
   - Admin Panel: `http://localhost:8000/admin/`
   - API Documentation: `http://localhost:8000/api/schema/swagger-ui/`

## 📱 Usage

### Dashboard
Access the main dashboard to view:
- Total inventory value
- Stock levels across locations
- Recent movements
- Low stock alerts
- Quick action buttons

### Operations Management
Navigate to the Operations page to manage:

1. **Receipts** - Record incoming inventory from vendors
   - Click "NEW" to create a new receipt
   - Enter vendor information and products
   - Validate and confirm receipt

2. **Deliveries** - Process outgoing orders
   - Click "NEW" in the Delivery tab
   - Fill in customer and product details
   - Schedule delivery date
   - Track delivery status (Draft → Ready → Done)

3. **Adjustments** - Adjust stock levels
   - Record discrepancies
   - Add adjustment reasons
   - Update inventory counts

4. **Internal Transfers** - Move stock between locations
   - Select source and destination locations
   - Choose products and quantities
   - Track transfer status

### Stock Management
- View all products and their stock levels
- Filter by location, category, or product
- Check low stock alerts
- View product details and history

## 🔌 API Endpoints

### Authentication
```
POST   /api/auth/login/          # User login
POST   /api/auth/logout/         # User logout
POST   /api/auth/register/       # User registration
```

### Products
```
GET    /api/products/            # List all products
POST   /api/products/            # Create product
GET    /api/products/{id}/       # Get product details
PUT    /api/products/{id}/       # Update product
DELETE /api/products/{id}/       # Delete product
```

### Inventory Operations
```
GET    /api/receipts/            # List receipts
POST   /api/receipts/            # Create receipt
GET    /api/deliveries/          # List deliveries
POST   /api/deliveries/          # Create delivery
GET    /api/adjustments/         # List adjustments
POST   /api/adjustments/         # Create adjustment
GET    /api/transfers/           # List transfers
POST   /api/transfers/           # Create transfer
```

### Stock Levels
```
GET    /api/stock/               # Get stock levels
GET    /api/stock/by-location/   # Stock by location
GET    /api/stock/low-stock/     # Low stock alerts
```

For complete API documentation, visit `/api/schema/swagger-ui/` when the server is running.

## 🎨 Design System

### Color Palette
- **Main Background**: `#EDF6F7` (Light Cyan)
- **Container Background**: `#F8F7FE` (Light Lavender)
- **Input Background**: `#FFFFFD` (Off-White)
- **Accent Color**: `#e53935` (Red)
- **Text Primary**: `#333333` (Dark Gray)
- **Text Secondary**: `#666666` (Medium Gray)

### Typography
- **Font Family**: Outfit (Google Fonts)
- **Icon Set**: Material Icons Round

### Components
- Modern card-based layouts
- Smooth transitions and hover effects
- Responsive tables with sorting and filtering
- Form validation and error handling
- Toast notifications for user feedback

## 🧪 Testing

Run the test suite:
```bash
python manage.py test inventory_core
```

Run specific test cases:
```bash
python manage.py test inventory_core.tests.TestProductModel
```

## 📦 Dependencies

### Backend
- **Django** (4.2+) - Web framework
- **Django REST Framework** - API development
- **psycopg2-binary** - PostgreSQL adapter
- **python-dotenv** - Environment variable management
- **django-otp** - Two-factor authentication
- **python-decouple** - Configuration management
- **drf-spectacular** - API documentation
- **django-filter** - Advanced filtering

### Frontend
- **Vanilla JavaScript** - No framework dependencies
- **Material Icons** - Icon library
- **Google Fonts (Outfit)** - Typography

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Coding Standards
- Follow PEP 8 for Python code
- Use ESLint for JavaScript code
- Write meaningful commit messages
- Add tests for new features
- Update documentation as needed

## 🗺️ Roadmap

- [ ] Machine Learning integration for demand forecasting
- [ ] Barcode/QR code scanning support
- [ ] Advanced reporting and analytics
- [ ] Mobile application (React Native)
- [ ] Multi-currency support
- [ ] Automated reorder suggestions
- [ ] Integration with popular e-commerce platforms
- [ ] Real-time notifications with WebSockets
- [ ] Export functionality (PDF, Excel, CSV)
- [ ] Multi-language support

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Aditya** - *Initial work* - [@Aditya232-rtx](https://github.com/Aditya232-rtx)

## 🙏 Acknowledgments

- Django and DRF communities for excellent documentation
- Material Design for icon library
- Google Fonts for typography
- All contributors who help improve this project

## 📞 Support

For support, email your-email@example.com or open an issue in the GitHub repository.

## 🔗 Links

- [GitHub Repository](https://github.com/Aditya232-rtx/StockMaster)
- [Issue Tracker](https://github.com/Aditya232-rtx/StockMaster/issues)
- [Documentation](https://github.com/Aditya232-rtx/StockMaster/wiki)

---

**Built with ❤️ using Django and JavaScript**
