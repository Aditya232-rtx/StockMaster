from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from rest_framework.permissions import IsAdminUser, IsAuthenticated
from django.db import models
from django.db.models import Sum, F
from django.shortcuts import get_object_or_404
from .models import Receipt, StockByLocation, DeliveryOrder, InternalTransfer, StockAdjustment, Location, ProductCategory, UnitOfMeasure, Product, StockTransaction, Vendor, PurchaseOrder, PurchaseOrderLineItem, PriceHistory
from .serializers import ReceiptSerializer, DeliveryOrderSerializer, InternalTransferSerializer, StockAdjustmentSerializer, LocationSerializer, ProductCategorySerializer, UnitOfMeasureSerializer, StockByLocationSerializer, ProductSerializer, VendorSerializer, PurchaseOrderSerializer, PriceHistorySerializer

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

    filter_backends = ['django_filters.rest_framework.DjangoFilterBackend',
        'rest_framework.filters.SearchFilter',
        'rest_framework.filters.OrderingFilter']
    search_fields = ['document_number', 'status']
    ordering_fields = ['date', 'document_number']

class VendorViewSet(viewsets.ReadOnlyModelViewSet):
    """Read-only viewset for Vendor"""
    queryset = Vendor.objects.all()
    serializer_class = VendorSerializer
    filter_backends = ['django_filters.rest_framework.DjangoFilterBackend',
        'rest_framework.filters.SearchFilter',
        'rest_framework.filters.OrderingFilter']
    search_fields = ['code', 'name']
    ordering_fields = ['name']

class PurchaseOrderViewSet(viewsets.ModelViewSet):
    """ViewSet for PurchaseOrder with nested line items"""
    queryset = PurchaseOrder.objects.select_related('vendor').prefetch_related('items__product').all()
    serializer_class = PurchaseOrderSerializer
    filter_backends = ['django_filters.rest_framework.DjangoFilterBackend',
        'rest_framework.filters.SearchFilter',
        'rest_framework.filters.OrderingFilter']
    filterset_fields = ['status', 'order_date', 'expected_delivery_date', 'vendor']
    search_fields = ['po_number', 'vendor__name']
    ordering_fields = ['order_date', 'expected_delivery_date']

class PriceHistoryViewSet(viewsets.ReadOnlyModelViewSet):
    """Read-only viewset for price history"""
    queryset = PriceHistory.objects.select_related('product', 'vendor').all()
    serializer_class = PriceHistorySerializer
    filter_backends = ['django_filters.rest_framework.DjangoFilterBackend',
        'rest_framework.filters.SearchFilter',
        'rest_framework.filters.OrderingFilter']
    filterset_fields = ['product', 'vendor', 'effective_date']
    search_fields = ['product__sku', 'vendor__code']
    ordering_fields = ['effective_date']

class BestVendorPriceView(APIView):
    """Return best vendor price for a given product SKU"""
    permission_classes = [IsAuthenticated]
    def get(self, request, *args, **kwargs):
        sku = request.query_params.get('sku')
        if not sku:
            return Response({"error": "sku query param required"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            product = Product.objects.get(sku=sku)
        except Product.DoesNotExist:
            return Response({"error": f"Product {sku} not found"}, status=status.HTTP_404_NOT_FOUND)
        # Find latest price per vendor and pick lowest
        latest_prices = PriceHistory.objects.filter(product=product).order_by('vendor', '-effective_date')
        best = None
        for ph in latest_prices:
            if best is None or ph.price < best.price:
                best = ph
        if not best:
            return Response({"error": "No price history for product"}, status=status.HTTP_404_NOT_FOUND)
        data = {
            "product": product.id,
            "product_sku": product.sku,
            "vendor": best.vendor.id,
            "vendor_code": best.vendor.code,
            "price": best.price,
            "currency": best.currency,
            "effective_date": best.effective_date,
        }
        return Response(data, status=status.HTTP_200_OK)

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

# --- ML & Blockchain Integration Views ---

from rest_framework import generics
from rest_framework.permissions import IsAdminUser, IsAuthenticated
from .serializers import MLDataExportSerializer

class MLDataExportView(generics.ListAPIView):
    """
    Export endpoint for the ML Engine. Provides historical transaction data
    for demand forecasting and anomaly detection training.
    Filtering by date range is highly recommended for performance.
    """
    queryset = StockTransaction.objects.all().order_by('timestamp')
    serializer_class = MLDataExportSerializer
    permission_classes = [IsAdminUser]  # Restrict access to administrators/ML service accounts
    
    # Enable filtering by date range (e.g., ?timestamp__gte=2024-01-01)
    filterset_fields = {
        'timestamp': ['gte', 'lte'], 
        'document_type': ['exact'],
        'product__sku': ['exact']
    }

class MLForecastUpdateView(APIView):
    """
    API for the ML Engine to update product forecast data or reorder points.
    This fulfills the requirement for the SmartReorderEngine to push data.
    """
    permission_classes = [IsAuthenticated]  # Require authentication for ML callbacks
    
    def post(self, request, *args, **kwargs):
        sku = request.data.get('product_sku')
        new_reorder_point = request.data.get('reorder_point')
        
        if not sku or new_reorder_point is None:
            return Response({"error": "Missing product_sku or reorder_point"}, 
                            status=status.HTTP_400_BAD_REQUEST)
        
        product = get_object_or_404(Product, sku=sku)
        
        # Update the product model with the ML-calculated value
        product.low_stock_threshold = new_reorder_point
        product.save()
        
        return Response({"message": f"Updated reorder point for {sku} to {new_reorder_point}"}, 
                        status=status.HTTP_200_OK)

class BlockchainAuditView(APIView):
    """
    API for the Blockchain Layer to record the final transaction hash 
    and provenance data back into the operational DB for quick lookup.
    """
    permission_classes = [IsAuthenticated]  # Require authentication for blockchain callbacks
    
    def post(self, request, *args, **kwargs):
        tx_id = request.data.get('tx_id')
        blockchain_hash = request.data.get('blockchain_hash')
        document_number = request.data.get('document_number')

        if not document_number or not blockchain_hash:
            return Response({"error": "Missing document_number or blockchain_hash"}, status=status.HTTP_400_BAD_REQUEST)

        # Try to locate the movement document across possible models
        from .models import Receipt, DeliveryOrder, InternalTransfer, StockAdjustment
        document = None
        for model in (Receipt, DeliveryOrder, InternalTransfer, StockAdjustment):
            try:
                document = model.objects.get(document_number=document_number)
                break
            except model.DoesNotExist:
                continue
        if not document:
            return Response({"error": f"Document {document_number} not found"}, status=status.HTTP_404_NOT_FOUND)

        # Persist the blockchain hash
        document.blockchain_hash = blockchain_hash
        document.save()

        return Response({"message": f"Recorded Blockchain Hash {blockchain_hash} for document {document_number}"},
                        status=status.HTTP_201_CREATED)
