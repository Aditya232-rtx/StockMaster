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
Demo video Link-https://drive.google.com/drive/folders/12Ju7ln4EUEQcRGThfMNOmLMeYBRbNHqn?usp=sharing
- [GitHub Repository](https://github.com/Aditya232-rtx/StockMaster)
- [Issue Tracker](https://github.com/Aditya232-rtx/StockMaster/issues)
- [Documentation](https://github.com/Aditya232-rtx/StockMaster/wiki)

---

**Built with ❤️ using Django and JavaScript**
