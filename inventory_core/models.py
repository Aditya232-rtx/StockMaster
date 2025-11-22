from django.db import models
from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType
from django.core.validators import MinValueValidator
from django.db.models import UniqueConstraint, CheckConstraint, Q

# --- Foundational Models ---

class UnitOfMeasure(models.Model):
    name = models.CharField(max_length=50, unique=True)
    symbol = models.CharField(max_length=10, unique=True)

    def __str__(self):
        return f"{self.name} ({self.symbol})"

class ProductCategory(models.Model):
    name = models.CharField(max_length=100)
    parent = models.ForeignKey('self', on_delete=models.CASCADE, null=True, blank=True, related_name='children')
    description = models.TextField(blank=True)

    class Meta:
        verbose_name_plural = "Product Categories"

    def __str__(self):
        return self.name

class Product(models.Model):
    sku = models.CharField(max_length=50, unique=True)
    name = models.CharField(max_length=200)
    category = models.ForeignKey(ProductCategory, on_delete=models.SET_NULL, null=True, related_name='products')
    uom = models.ForeignKey(UnitOfMeasure, on_delete=models.PROTECT, related_name='products')
    description = models.TextField(blank=True)
    low_stock_threshold = models.PositiveIntegerField(default=10)
    
    # ML-related fields
    unit_cost = models.DecimalField(max_digits=10, decimal_places=2, default=0, help_text="Average unit cost")
    reorder_quantity = models.PositiveIntegerField(default=100, help_text="Default reorder quantity")
    lead_time_days = models.PositiveIntegerField(default=7, help_text="Average lead time in days")
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} ({self.sku})"

class Location(models.Model):
    name = models.CharField(max_length=100)
    parent = models.ForeignKey('self', on_delete=models.CASCADE, null=True, blank=True, related_name='children')
    is_warehouse = models.BooleanField(default=False)
    address = models.TextField(blank=True)

    def __str__(self):
        return self.name

# --- Vendor & Purchase Order Models ---

class Vendor(models.Model):
    code = models.CharField(max_length=50, unique=True)
    name = models.CharField(max_length=200)
    email = models.EmailField(blank=True)
    phone = models.CharField(max_length=20, blank=True)
    address = models.TextField(blank=True)
    reliability_score = models.PositiveIntegerField(default=85, validators=[MinValueValidator(0)], help_text="0-100 score")
    payment_terms = models.PositiveIntegerField(default=30, help_text="Payment terms in days")
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} ({self.code})"

class PurchaseOrder(models.Model):
    STATUS_CHOICES = [
        ('DRAFT', 'Draft'),
        ('SENT', 'Sent to Vendor'),
        ('CONFIRMED', 'Confirmed by Vendor'),
        ('RECEIVED', 'Goods Received'),
        ('CANCELLED', 'Cancelled'),
    ]
    
    po_number = models.CharField(max_length=50, unique=True)
    vendor = models.ForeignKey(Vendor, on_delete=models.PROTECT, related_name='purchase_orders')
    order_date = models.DateTimeField(auto_now_add=True)
    expected_delivery_date = models.DateField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='DRAFT', db_index=True)
    total_amount = models.DecimalField(max_digits=15, decimal_places=2, default=0)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.po_number} - {self.vendor.name}"

class PurchaseOrderLineItem(models.Model):
    purchase_order = models.ForeignKey(PurchaseOrder, on_delete=models.CASCADE, related_name='items')
    product = models.ForeignKey(Product, on_delete=models.PROTECT)
    quantity = models.DecimalField(max_digits=15, decimal_places=4, validators=[MinValueValidator(0)])
    unit_price = models.DecimalField(max_digits=10, decimal_places=2)
    received_quantity = models.DecimalField(max_digits=15, decimal_places=4, default=0)
    notes = models.TextField(blank=True)

    def __str__(self):
        return f"{self.quantity} of {self.product.sku} in {self.purchase_order.po_number}"

class PriceHistory(models.Model):
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='price_history')
    vendor = models.ForeignKey(Vendor, on_delete=models.CASCADE, related_name='price_history')
    price = models.DecimalField(max_digits=10, decimal_places=2)
    effective_date = models.DateField(db_index=True)
    currency = models.CharField(max_length=3, default='USD')
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name_plural = "Price Histories"
        ordering = ['-effective_date']
        indexes = [
            models.Index(fields=['product', 'vendor', '-effective_date']),
        ]

    def __str__(self):
        return f"{self.product.sku} from {self.vendor.code}: ${self.price} ({self.effective_date})"

# --- Inventory State & Transaction Models ---

class StockByLocation(models.Model):
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='stock_levels')
    location = models.ForeignKey(Location, on_delete=models.CASCADE, related_name='stock_levels')
    quantity = models.DecimalField(max_digits=15, decimal_places=4, default=0, validators=[MinValueValidator(0)])
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            UniqueConstraint(fields=['product', 'location'], name='unique_product_location_stock'),
            CheckConstraint(check=Q(quantity__gte=0), name='quantity_gte_0'),
        ]

    def __str__(self):
        return f"{self.product.sku} @ {self.location.name}: {self.quantity}"

class MovementBase(models.Model):
    STATUS_CHOICES = [
        ('DRAFT', 'Draft'),
        ('WAITING', 'Waiting'),
        ('READY', 'Ready'),
        ('DONE', 'Done'),
        ('CANCELED', 'Canceled'),
    ]

    document_number = models.CharField(max_length=50, unique=True)
    date = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='DRAFT', db_index=True)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True

    def __str__(self):
        return self.document_number

class Receipt(MovementBase):
    vendor = models.CharField(max_length=100, blank=True)
    purchase_order = models.ForeignKey(PurchaseOrder, on_delete=models.SET_NULL, null=True, blank=True, related_name='receipts')
    received_date = models.DateField(null=True, blank=True)
    # Additional fields specific to receipts can go here

class ReceiptLineItem(models.Model):
    receipt = models.ForeignKey(Receipt, on_delete=models.CASCADE, related_name='items')
    product = models.ForeignKey(Product, on_delete=models.PROTECT)
    location = models.ForeignKey(Location, on_delete=models.PROTECT)
    quantity = models.DecimalField(max_digits=15, decimal_places=4, validators=[MinValueValidator(0)])

    def __str__(self):
        return f"{self.quantity} of {self.product.sku} in {self.receipt.document_number}"

class DeliveryOrder(MovementBase):
    customer = models.CharField(max_length=100, blank=True)

class DeliveryLineItem(models.Model):
    delivery = models.ForeignKey(DeliveryOrder, on_delete=models.CASCADE, related_name='items')
    product = models.ForeignKey(Product, on_delete=models.PROTECT)
    location = models.ForeignKey(Location, on_delete=models.PROTECT)
    quantity = models.DecimalField(max_digits=15, decimal_places=4, validators=[MinValueValidator(0)])

    def __str__(self):
        return f"{self.quantity} of {self.product.sku} in {self.delivery.document_number}"

class InternalTransfer(MovementBase):
    source_location = models.ForeignKey(Location, on_delete=models.PROTECT, related_name='transfers_out')
    destination_location = models.ForeignKey(Location, on_delete=models.PROTECT, related_name='transfers_in')

class TransferLineItem(models.Model):
    transfer = models.ForeignKey(InternalTransfer, on_delete=models.CASCADE, related_name='items')
    product = models.ForeignKey(Product, on_delete=models.PROTECT)
    quantity = models.DecimalField(max_digits=15, decimal_places=4, validators=[MinValueValidator(0)])

    def __str__(self):
        return f"{self.quantity} of {self.product.sku} in {self.transfer.document_number}"

class StockAdjustment(MovementBase):
    reason = models.CharField(max_length=200)

class AdjustmentLineItem(models.Model):
    adjustment = models.ForeignKey(StockAdjustment, on_delete=models.CASCADE, related_name='items')
    product = models.ForeignKey(Product, on_delete=models.PROTECT)
    location = models.ForeignKey(Location, on_delete=models.PROTECT)
    quantity_change = models.DecimalField(max_digits=15, decimal_places=4) # Can be negative

    def __str__(self):
        return f"{self.quantity_change} of {self.product.sku} in {self.adjustment.document_number}"

class StockTransaction(models.Model):
    DOCUMENT_TYPES = [
        ('RECEIPT', 'Receipt'),
        ('DELIVERY', 'Delivery Order'),
        ('TRANSFER', 'Internal Transfer'),
        ('ADJUSTMENT', 'Stock Adjustment'),
    ]

    product = models.ForeignKey(Product, on_delete=models.PROTECT, related_name='transactions')
    location = models.ForeignKey(Location, on_delete=models.PROTECT, related_name='transactions')
    quantity_change = models.DecimalField(max_digits=15, decimal_places=4) # Positive for add, negative for remove
    document_type = models.CharField(max_length=20, choices=DOCUMENT_TYPES, default='RECEIPT', db_index=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    # Generic Foreign Key to link to the specific movement document
    content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE)
    object_id = models.PositiveIntegerField()
    movement_document = GenericForeignKey('content_type', 'object_id')

    def __str__(self):
        return f"{self.product.sku} {self.quantity} @ {self.location.name}"
