"""
URL configuration for config project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView, SpectacularRedocView
from inventory_core.views import (
    ReceiptViewSet,
    DashboardKPIView,
    DeliveryOrderViewSet,
    InternalTransferViewSet,
    StockAdjustmentViewSet,
    ProductViewSet,
    LocationViewSet,
    ProductCategoryViewSet,
    UnitOfMeasureViewSet,
    StockByLocationViewSet,
    MLDataExportView,
    MLForecastUpdateView,
    BlockchainAuditView,
    VendorViewSet,
    PurchaseOrderViewSet,
    PriceHistoryViewSet,
    BestVendorPriceView,
)

router = DefaultRouter()
router.register(r'products', ProductViewSet)
router.register(r'receipts', ReceiptViewSet)
router.register(r'deliveries', DeliveryOrderViewSet)
router.register(r'transfers', InternalTransferViewSet)
router.register(r'adjustments', StockAdjustmentViewSet)
router.register(r'locations', LocationViewSet)
router.register(r'categories', ProductCategoryViewSet)
router.register(r'uom', UnitOfMeasureViewSet)
router.register(r'inventory-state', StockByLocationViewSet)
router.register(r'vendors', VendorViewSet)
router.register(r'purchase-orders', PurchaseOrderViewSet)
router.register(r'price-history', PriceHistoryViewSet)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('django.contrib.auth.urls')),
    
    # API Documentation
    path('api/schema/', SpectacularAPIView.as_view(), name='schema'),
    path('api/docs/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
    path('api/redoc/', SpectacularRedocView.as_view(url_name='schema'), name='redoc'),
    
    # API Endpoints
    path('api/kpis/', DashboardKPIView.as_view(), name='dashboard-kpis'),
    
    # --- INTEGRATION ENDPOINTS ---
    # ML Data Exports
    path('api/ml/export/transactions/', MLDataExportView.as_view(), name='ml-data-export'),
    # ML Feedback/Update
    path('api/ml/update/forecast/', MLForecastUpdateView.as_view(), name='ml-forecast-update'),
    # Blockchain Callback
    path('api/blockchain/audit/', BlockchainAuditView.as_view(), name='blockchain-audit-callback'),
    # Best Vendor Price
    path('api/best-vendor-price/', BestVendorPriceView.as_view(), name='best-vendor-price'),
    
    path('api/', include(router.urls)),
]
