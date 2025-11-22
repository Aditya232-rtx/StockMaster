from django.contrib import admin
from .models import (
    Product, ProductCategory, UnitOfMeasure, Location,
    Vendor, PurchaseOrder, PurchaseOrderLineItem, PriceHistory,
    Receipt, ReceiptLineItem,
    DeliveryOrder, DeliveryLineItem,
    InternalTransfer, TransferLineItem,
    StockAdjustment, AdjustmentLineItem,
    StockByLocation, StockTransaction
)

@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ['sku', 'name', 'category', 'uom', 'low_stock_threshold']
    search_fields = ['sku', 'name']
    list_filter = ['category', 'uom']

@admin.register(ProductCategory)
class ProductCategoryAdmin(admin.ModelAdmin):
    list_display = ['name', 'parent']
    search_fields = ['name']
    list_filter = ['parent']

@admin.register(UnitOfMeasure)
class UnitOfMeasureAdmin(admin.ModelAdmin):
    list_display = ['name', 'symbol']
    search_fields = ['name', 'symbol']

@admin.register(Location)
class LocationAdmin(admin.ModelAdmin):
    list_display = ['name', 'parent', 'is_warehouse']
    search_fields = ['name']
    list_filter = ['is_warehouse', 'parent']

@admin.register(Vendor)
class VendorAdmin(admin.ModelAdmin):
    list_display = ['code', 'name', 'reliability_score', 'payment_terms', 'is_active']
    search_fields = ['code', 'name']
    list_filter = ['is_active']

class PurchaseOrderLineItemInline(admin.TabularInline):
    model = PurchaseOrderLineItem
    extra = 0

@admin.register(PurchaseOrder)
class PurchaseOrderAdmin(admin.ModelAdmin):
    list_display = ['po_number', 'vendor', 'status', 'order_date', 'expected_delivery_date']
    search_fields = ['po_number', 'vendor__name']
    list_filter = ['status']
    inlines = [PurchaseOrderLineItemInline]

@admin.register(PriceHistory)
class PriceHistoryAdmin(admin.ModelAdmin):
    list_display = ['product', 'vendor', 'price', 'effective_date', 'currency']
    search_fields = ['product__sku', 'vendor__code']
    list_filter = ['currency', 'effective_date']

class ReceiptLineItemInline(admin.TabularInline):
    model = ReceiptLineItem
    extra = 0

@admin.register(Receipt)
class ReceiptAdmin(admin.ModelAdmin):
    list_display = ['document_number', 'vendor', 'status', 'date']
    search_fields = ['document_number', 'vendor']
    list_filter = ['status']
    inlines = [ReceiptLineItemInline]

class DeliveryLineItemInline(admin.TabularInline):
    model = DeliveryLineItem
    extra = 0

@admin.register(DeliveryOrder)
class DeliveryOrderAdmin(admin.ModelAdmin):
    list_display = ['document_number', 'customer', 'status', 'date']
    search_fields = ['document_number', 'customer']
    list_filter = ['status']
    inlines = [DeliveryLineItemInline]

class TransferLineItemInline(admin.TabularInline):
    model = TransferLineItem
    extra = 0

@admin.register(InternalTransfer)
class InternalTransferAdmin(admin.ModelAdmin):
    list_display = ['document_number', 'source_location', 'destination_location', 'status', 'date']
    search_fields = ['document_number']
    list_filter = ['status']
    inlines = [TransferLineItemInline]

class AdjustmentLineItemInline(admin.TabularInline):
    model = AdjustmentLineItem
    extra = 0

@admin.register(StockAdjustment)
class StockAdjustmentAdmin(admin.ModelAdmin):
    list_display = ['document_number', 'reason', 'status', 'date']
    search_fields = ['document_number', 'reason']
    list_filter = ['status']
    inlines = [AdjustmentLineItemInline]

@admin.register(StockByLocation)
class StockByLocationAdmin(admin.ModelAdmin):
    list_display = ['product', 'location', 'quantity']
    search_fields = ['product__sku', 'location__name']
    list_filter = ['location']

@admin.register(StockTransaction)
class StockTransactionAdmin(admin.ModelAdmin):
    list_display = ['product', 'location', 'quantity_change', 'document_type', 'timestamp']
    search_fields = ['product__sku', 'location__name']
    list_filter = ['document_type', 'timestamp']
