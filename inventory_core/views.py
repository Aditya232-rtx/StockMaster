from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework.views import APIView
from django.db import models
from django.db.models import Sum, F
from .models import Receipt, StockByLocation, DeliveryOrder, InternalTransfer, StockAdjustment, Location, ProductCategory, UnitOfMeasure, Product
from .serializers import ReceiptSerializer, DeliveryOrderSerializer, InternalTransferSerializer, StockAdjustmentSerializer, LocationSerializer, ProductCategorySerializer, UnitOfMeasureSerializer, StockByLocationSerializer, ProductSerializer

class ReceiptViewSet(viewsets.ModelViewSet):
    """API endpoint for managing incoming stock Receipts."""
    queryset = Receipt.objects.prefetch_related('items__product', 'items__location').all()
    serializer_class = ReceiptSerializer
    filterset_fields = ['status', 'vendor', 'date', 'items__location']

class DashboardKPIView(APIView):
    """API endpoint to calculate and return real-time Key Performance Indicators."""
    
    def get(self, request, *args, **kwargs):
        # IV.3.1. Calculating Total Stock
        total_stock = StockByLocation.objects.aggregate(
            total_quantity=Sum('quantity')
        )['total_quantity'] or 0
        
        # IV.3.2. Identifying Low Stock Items
        low_stock_items = StockByLocation.objects.filter(
            # Using F() expression to compare two fields at the database level
            quantity__lte=F('product__low_stock_threshold'),
            quantity__gt=0 # Not strictly necessary but useful to exclude 0 stock from "Low Stock" status
        ).count()
        
        # IV.3.3. Pending Movements (Draft/Waiting)
        pending_receipts = Receipt.objects.filter(status__in=['DRAFT', 'WAITING']).count()
        pending_deliveries = DeliveryOrder.objects.filter(status__in=['DRAFT', 'WAITING']).count()
        pending_transfers = InternalTransfer.objects.filter(status__in=['DRAFT', 'WAITING']).count()
        
        kpis = {
            "total_products_in_stock": StockByLocation.objects.filter(quantity__gt=0).count(),
            "total_stock_quantity": total_stock,
            "low_stock_out_of_stock_items": low_stock_items,
            "pending_receipts": pending_receipts,
            "pending_deliveries": pending_deliveries,
            "internal_transfers_scheduled": pending_transfers,
        }
        
        return Response(kpis)

class DeliveryOrderViewSet(viewsets.ModelViewSet):
    """API endpoint for managing outgoing stock Delivery Orders."""
    queryset = DeliveryOrder.objects.prefetch_related('items__product', 'items__location').all()
    serializer_class = DeliveryOrderSerializer
    filterset_fields = ['status', 'customer', 'date', 'items__location']

class InternalTransferViewSet(viewsets.ModelViewSet):
    """API endpoint for managing internal stock transfers."""
    queryset = InternalTransfer.objects.prefetch_related('items__product').all()
    serializer_class = InternalTransferSerializer
    filterset_fields = ['status', 'source_location', 'destination_location']

class StockAdjustmentViewSet(viewsets.ModelViewSet):
    """API endpoint for managing stock adjustments."""
    queryset = StockAdjustment.objects.prefetch_related('items__product', 'items__location').all()
    serializer_class = StockAdjustmentSerializer
    filterset_fields = ['status', 'created_at', 'reason']

class ProductViewSet(viewsets.ModelViewSet):
    """API endpoint for managing Products."""
    queryset = Product.objects.select_related('category', 'uom').all()
    serializer_class = ProductSerializer
    filterset_fields = ['category', 'sku']

class LocationViewSet(viewsets.ModelViewSet):
    """Endpoint for managing Locations (Warehouses, Racks)."""
    queryset = Location.objects.all()
    serializer_class = LocationSerializer

class ProductCategoryViewSet(viewsets.ModelViewSet):
    """Endpoint for managing Product Categories."""
    queryset = ProductCategory.objects.all()
    serializer_class = ProductCategorySerializer

class UnitOfMeasureViewSet(viewsets.ModelViewSet):
    """Endpoint for managing Units of Measure."""
    queryset = UnitOfMeasure.objects.all()
    serializer_class = UnitOfMeasureSerializer

class StockByLocationViewSet(viewsets.ReadOnlyModelViewSet):
    """Read-only endpoint for viewing the current inventory state."""
    queryset = StockByLocation.objects.filter(quantity__gt=0).select_related('product', 'location') # Show only items with quantity > 0
    serializer_class = StockByLocationSerializer
    filterset_fields = ['product', 'location']
