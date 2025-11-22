from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient
from rest_framework import status
from .models import Product, Location, UnitOfMeasure, ProductCategory, StockByLocation, Receipt, DeliveryOrder, InternalTransfer, StockAdjustment

class InventoryFlowTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        
        # Setup Basic Data
        self.uom = UnitOfMeasure.objects.create(name='Units', symbol='pcs')
        self.category = ProductCategory.objects.create(name='General')
        self.wh = Location.objects.create(name='Warehouse', is_warehouse=True)
        self.shop = Location.objects.create(name='Shop', parent=self.wh)
        
        self.product = Product.objects.create(
            sku='TEST-PROD',
            name='Test Product',
            category=self.category,
            uom=self.uom,
            low_stock_threshold=10
        )

    def test_1_receipt_flow(self):
        """Test receiving stock increases inventory."""
        url = reverse('receipt-list')
        data = {
            'document_number': 'REC-001',
            'vendor': 'Test Vendor',
            'items': [
                {'product_sku': self.product.sku, 'location_id': self.wh.id, 'quantity': 100}
            ]
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        
        # Verify Stock
        stock = StockByLocation.objects.get(product=self.product, location=self.wh)
        self.assertEqual(stock.quantity, 100)

    def test_2_delivery_flow_success(self):
        """Test delivering stock decreases inventory."""
        # Seed stock first
        StockByLocation.objects.create(product=self.product, location=self.wh, quantity=100)
        
        url = reverse('deliveryorder-list')
        data = {
            'document_number': 'DEL-001',
            'customer': 'Test Customer',
            'items': [
                {'product_sku': self.product.sku, 'location_id': self.wh.id, 'quantity': 30}
            ]
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        
        # Verify Stock
        stock = StockByLocation.objects.get(product=self.product, location=self.wh)
        self.assertEqual(stock.quantity, 70)

    def test_3_delivery_flow_insufficient_stock(self):
        """Test delivery fails when stock is insufficient."""
        # Seed stock
        StockByLocation.objects.create(product=self.product, location=self.wh, quantity=10)
        
        url = reverse('deliveryorder-list')
        data = {
            'document_number': 'DEL-FAIL',
            'customer': 'Test Customer',
            'items': [
                {'product_sku': self.product.sku, 'location_id': self.wh.id, 'quantity': 50}
            ]
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        
        # Verify Stock Unchanged
        stock = StockByLocation.objects.get(product=self.product, location=self.wh)
        self.assertEqual(stock.quantity, 10)

    def test_4_internal_transfer(self):
        """Test moving stock between locations."""
        # Seed stock at source
        StockByLocation.objects.create(product=self.product, location=self.wh, quantity=100)
        
        url = reverse('internaltransfer-list')
        data = {
            'document_number': 'TRF-001',
            'source_location': self.wh.id,
            'destination_location': self.shop.id,
            'items': [
                {'product_sku': self.product.sku, 'quantity': 20}
            ]
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        
        # Verify Source Decreased
        source_stock = StockByLocation.objects.get(product=self.product, location=self.wh)
        self.assertEqual(source_stock.quantity, 80)
        
        # Verify Destination Increased
        dest_stock = StockByLocation.objects.get(product=self.product, location=self.shop)
        self.assertEqual(dest_stock.quantity, 20)

    def test_5_stock_adjustment_loss(self):
        """Test negative adjustment (loss)."""
        # Seed stock
        StockByLocation.objects.create(product=self.product, location=self.wh, quantity=50)
        
        url = reverse('stockadjustment-list')
        data = {
            'document_number': 'ADJ-LOSS',
            'reason': 'Damaged',
            'items': [
                {'product_sku': self.product.sku, 'location_id': self.wh.id, 'quantity_change': -5}
            ]
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        
        # Verify Stock
        stock = StockByLocation.objects.get(product=self.product, location=self.wh)
        self.assertEqual(stock.quantity, 45)

    def test_6_stock_adjustment_excessive_loss(self):
        """Test negative adjustment fails if exceeds stock."""
        # Seed stock
        StockByLocation.objects.create(product=self.product, location=self.wh, quantity=5)
        
        url = reverse('stockadjustment-list')
        data = {
            'document_number': 'ADJ-FAIL',
            'reason': 'Lost',
            'items': [
                {'product_sku': self.product.sku, 'location_id': self.wh.id, 'quantity_change': -10}
            ]
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
