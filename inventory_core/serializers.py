from rest_framework import serializers
from django.db import transaction, models
from django.contrib.contenttypes.models import ContentType
from .models import Product, Receipt, Location, StockTransaction, StockByLocation, ReceiptLineItem, DeliveryOrder, InternalTransfer, StockAdjustment, ProductCategory, UnitOfMeasure, DeliveryLineItem, TransferLineItem, AdjustmentLineItem

class ProductSerializer(serializers.ModelSerializer):
    initial_stock = serializers.DecimalField(max_digits=12, decimal_places=3, write_only=True, required=False, min_value=0)
    initial_location = serializers.PrimaryKeyRelatedField(queryset=Location.objects.all(), write_only=True, required=False)

    class Meta:
        model = Product
        fields = ['id', 'sku', 'name', 'uom', 'category', 'low_stock_threshold', 'description', 'initial_stock', 'initial_location']

    def create(self, validated_data):
        initial_stock = validated_data.pop('initial_stock', 0)
        initial_location = validated_data.pop('initial_location', None)
        
        product = Product.objects.create(**validated_data)
        
        if initial_stock > 0 and initial_location:
            # Create a Receipt for the initial stock
            with transaction.atomic():
                receipt = Receipt.objects.create(
                    document_number=f"INIT-{product.sku}",
                    status='DONE',
                    notes="Initial stock entry"
                )
                receipt_content_type = ContentType.objects.get_for_model(Receipt)
                
                # Log Transaction
                StockTransaction.objects.create(
                    product=product,
                    location=initial_location,
                    quantity_change=initial_stock,
                    document_type='RECEIPT',
                    content_type=receipt_content_type,
                    object_id=receipt.pk
                )
                
                # Update Stock
                stock_record, created = StockByLocation.objects.get_or_create(
                    product=product,
                    location=initial_location,
                    defaults={'quantity': 0}
                )
                stock_record.quantity = models.F('quantity') + initial_stock
                stock_record.save()
                
                # Create Line Item
                ReceiptLineItem.objects.create(
                    receipt=receipt,
                    product=product,
                    location=initial_location,
                    quantity=initial_stock
                )
                
        return product

class ReceiptLineItemSerializer(serializers.Serializer):
    """Defines the expected input structure for an item being received."""
    product_sku = serializers.CharField(max_length=50, help_text="The SKU of the product received.")
    location_id = serializers.PrimaryKeyRelatedField(queryset=Location.objects.all(), help_text="The ID of the location where stock is received.")
    quantity = serializers.DecimalField(max_digits=12, decimal_places=3, min_value=0, help_text="The quantity received.")

class ReceiptSerializer(serializers.ModelSerializer):
    """The main serializer for the Receipt document."""
    items = ReceiptLineItemSerializer(many=True, write_only=True) # Nested items for processing
    
    class Meta:
        model = Receipt
        fields = ['id', 'document_number', 'vendor', 'status', 'items']
        read_only_fields = ['status']

    def create(self, validated_data):
        items_data = validated_data.pop('items')
        
        # 3.3.1. Atomic Database Transaction: Guarantees All-or-Nothing
        with transaction.atomic():
            # 3.3.2. Core Processing Flow: Create the document first
            receipt = Receipt.objects.create(**validated_data, status='READY') # Set status to 'Ready' or 'Done' immediately for receipt
            receipt_content_type = ContentType.objects.get_for_model(Receipt)

            # Process each line item
            for item_data in items_data:
                # Retrieve Product and Location instances
                try:
                    product = Product.objects.get(sku=item_data['product_sku'])
                except Product.DoesNotExist:
                    # Application-layer validation
                    raise serializers.ValidationError(f"Product with SKU {item_data['product_sku']} not found.")
                
                location = item_data['location_id']
                quantity = item_data['quantity']

                # Save the line item for the record
                ReceiptLineItem.objects.create(
                    receipt=receipt,
                    product=product,
                    location=location,
                    quantity=quantity
                )

                # 1. Log History: Create StockTransaction (The Ledger Write)
                StockTransaction.objects.create(
                    product=product,
                    location=location,
                    quantity_change=quantity, # Positive for receipt
                    document_type='RECEIPT',
                    content_type=receipt_content_type,
                    object_id=receipt.pk
                )

                # 2. Update State: Locate or create StockByLocation and apply change
                stock_record, created = StockByLocation.objects.get_or_create(
                    product=product,
                    location=location,
                    defaults={'quantity': 0}
                )
                
                # Apply the quantity change (atomic update to prevent race conditions)
                stock_record.quantity += quantity
                stock_record.save()
            
            # The transaction automatically commits here if successful. 

        return receipt

class DeliveryLineItemSerializer(serializers.ModelSerializer):
    """Defines the expected input structure for an item being delivered."""
    product_sku = serializers.SlugRelatedField(source='product', slug_field='sku', queryset=Product.objects.all())
    location_id = serializers.PrimaryKeyRelatedField(source='location', queryset=Location.objects.all())
    
    class Meta:
        model = DeliveryLineItem
        fields = ['product_sku', 'location_id', 'quantity']

class DeliveryOrderSerializer(serializers.ModelSerializer):
    """The main serializer for the Delivery Order document."""
    items = DeliveryLineItemSerializer(many=True)
    
    class Meta:
        model = DeliveryOrder
        fields = ['id', 'document_number', 'customer', 'status', 'items']
        read_only_fields = ['status']
    
    def create(self, validated_data):
        items_data = validated_data.pop('items')
        
        # --- IV.1.1. Pre-Commit Business Validation ---
        # Proactively check stock availability before starting the atomic transaction.
        for item_data in items_data:
            product = item_data['product']
            location = item_data['location']
            quantity_to_deliver = item_data['quantity']
            
            # Query the current stock level
            try:
                stock_record = StockByLocation.objects.get(
                    product=product,
                    location=location
                )
                
                if stock_record.quantity < quantity_to_deliver:
                    raise serializers.ValidationError({
                        'items': f"Insufficient stock for {product.sku} at {location.name}. Available: {stock_record.quantity}, Requested: {quantity_to_deliver}"
                    })
                    
            except StockByLocation.DoesNotExist:
                # If no record exists, stock is implicitly zero.
                if quantity_to_deliver > 0:
                    raise serializers.ValidationError({
                        'items': f"No stock record found for {product.sku} at {location.name} (Available: 0)."
                    })

        # --- III.3. Atomic Database Transaction (Post-Validation) ---
        with transaction.atomic():
            delivery = DeliveryOrder.objects.create(**validated_data, status='DONE')
            delivery_content_type = ContentType.objects.get_for_model(DeliveryOrder)

            for item_data in items_data:
                product = item_data['product']
                location = item_data['location']
                quantity = item_data['quantity']
                
                # NOTE: The quantity_change is NEGATIVE for outgoing stock (Debit)
                quantity_change = -quantity 

                # 1. Log History: Create StockTransaction (Ledger Write)
                StockTransaction.objects.create(
                    product=product,
                    location=location,
                    quantity_change=quantity_change,
                    document_type='DELIVERY',
                    content_type=delivery_content_type,
                    object_id=delivery.pk
                )

                # 2. Update State: Locate StockByLocation and apply change
                stock_record = StockByLocation.objects.select_for_update().get( # Use select_for_update for robust concurrency
                    product=product,
                    location=location,
                )
                
                stock_record.quantity = models.F('quantity') + quantity_change
                stock_record.save()
                
                # Save Line Item
                DeliveryLineItem.objects.create(
                    delivery=delivery,
                    product=product,
                    location=location,
                    quantity=quantity
                )
            
            # The database's CheckConstraint (quantity >= 0) acts as a final fail-safe.
        return delivery

class TransferLineItemSerializer(serializers.ModelSerializer):
    """Defines the expected input structure for an item being transferred."""
    product_sku = serializers.SlugRelatedField(source='product', slug_field='sku', queryset=Product.objects.all())
    
    class Meta:
        model = TransferLineItem
        fields = ['product_sku', 'quantity']

class InternalTransferSerializer(serializers.ModelSerializer):
    """The main serializer for the Internal Transfer document."""
    items = TransferLineItemSerializer(many=True) # Changed to allow read/write
    
    class Meta:
        model = InternalTransfer
        fields = ['id', 'document_number', 'source_location', 'destination_location', 'status', 'items']
        read_only_fields = ['status']
    
    def create(self, validated_data):
        items_data = validated_data.pop('items')
        
        source_location = validated_data['source_location']
        destination_location = validated_data['destination_location']

        # Pre-validation: Check for stock availability at the source location
        # (Similar logic as DeliveryOrder, checking all items against the source)
        for item_data in items_data:
            product = item_data['product']
            quantity_to_move = item_data['quantity']
            
            try:
                stock_record = StockByLocation.objects.get(
                    product=product,
                    location=source_location
                )
                if stock_record.quantity < quantity_to_move:
                    raise serializers.ValidationError({
                        'items': f"Insufficient stock for {product.sku} at source {source_location.name}. Available: {stock_record.quantity}, Requested: {quantity_to_move}"
                    })
            except StockByLocation.DoesNotExist:
                if quantity_to_move > 0:
                     raise serializers.ValidationError(f"No stock record found for {product.sku} at source {source_location.name} (Available: 0).")


        # --- III.3.3. Handling Internal Transfers: Dual-Entry Atomic Transaction ---
        with transaction.atomic():
            transfer = InternalTransfer.objects.create(**validated_data, status='DONE')
            transfer_content_type = ContentType.objects.get_for_model(InternalTransfer)

            for item_data in items_data:
                product = item_data['product']
                quantity = item_data['quantity']
                
                # --- LEG 1: DEBIT (Decrease Stock at Source) ---
                quantity_debit = -quantity
                
                # 1. Log History: Debit transaction at source
                StockTransaction.objects.create(
                    product=product,
                    location=source_location,
                    quantity_change=quantity_debit,
                    document_type='TRANSFER',
                    content_type=transfer_content_type,
                    object_id=transfer.pk
                )
                # 2. Update State: Decrease stock at source
                source_stock = StockByLocation.objects.select_for_update().get(
                    product=product,
                    location=source_location
                )
                source_stock.quantity = models.F('quantity') + quantity_debit
                source_stock.save()

                # --- LEG 2: CREDIT (Increase Stock at Destination) ---
                quantity_credit = quantity
                
                # 1. Log History: Credit transaction at destination
                StockTransaction.objects.create(
                    product=product,
                    location=destination_location,
                    quantity_change=quantity_credit,
                    document_type='TRANSFER',
                    content_type=transfer_content_type,
                    object_id=transfer.pk
                )
                # 2. Update State: Increase stock at destination
                destination_stock, created = StockByLocation.objects.get_or_create(
                    product=product,
                    location=destination_location,
                    defaults={'quantity': 0}
                )
                destination_stock.quantity = models.F('quantity') + quantity_credit
                destination_stock.save()

                # Save Line Item
                TransferLineItem.objects.create(
                    transfer=transfer,
                    product=product,
                    quantity=quantity
                )

        return transfer

class AdjustmentLineItemSerializer(serializers.ModelSerializer):
    """Defines the expected input structure for an adjustment item."""
    product_sku = serializers.SlugRelatedField(source='product', slug_field='sku', queryset=Product.objects.all())
    location_id = serializers.PrimaryKeyRelatedField(source='location', queryset=Location.objects.all())
    # Note: quantity is the change value, positive or negative
    quantity_change = serializers.DecimalField(max_digits=12, decimal_places=3, help_text="Signed value: Positive for gain, Negative for loss.")
    
    class Meta:
        model = AdjustmentLineItem
        fields = ['product_sku', 'location_id', 'quantity_change']

class StockAdjustmentSerializer(serializers.ModelSerializer):
    """The main serializer for the Stock Adjustment document."""
    items = AdjustmentLineItemSerializer(many=True) # Changed to allow read/write
    
    class Meta:
        model = StockAdjustment
        fields = ['id', 'document_number', 'reason', 'status', 'items']
        read_only_fields = ['status']
    
    def create(self, validated_data):
        items_data = validated_data.pop('items')
        
        # Pre-validation: Check if any negative adjustments exceed current stock
        for item_data in items_data:
            quantity_change = item_data['quantity_change']
            
            # Only check if the adjustment is a negative (debit/loss)
            if quantity_change < 0:
                product = item_data['product']
                location = item_data['location']
                
                try:
                    stock_record = StockByLocation.objects.get(
                        product=product,
                        location=location
                    )
                    # Check if the loss is greater than the available stock
                    if stock_record.quantity < abs(quantity_change):
                         raise serializers.ValidationError({
                            'items': f"Cannot adjust for loss of {abs(quantity_change)} for {product.sku} at {location.name}. Only {stock_record.quantity} available."
                        })
                        
                except StockByLocation.DoesNotExist:
                    if abs(quantity_change) > 0:
                         raise serializers.ValidationError(f"No stock record found for {product.sku} at source {location.name} (Available: 0).")

        # Atomic Database Transaction
        with transaction.atomic():
            adjustment = StockAdjustment.objects.create(**validated_data, status='DONE')
            adjustment_content_type = ContentType.objects.get_for_model(StockAdjustment)

            for item_data in items_data:
                product = item_data['product']
                location = item_data['location']
                quantity_change = item_data['quantity_change'] 

                # 1. Log History: Create StockTransaction
                StockTransaction.objects.create(
                    product=product,
                    location=location,
                    quantity_change=quantity_change,
                    document_type='ADJUSTMENT',
                    content_type=adjustment_content_type,
                    object_id=adjustment.pk
                )

                # 2. Update State: Locate or create StockByLocation
                stock_record, created = StockByLocation.objects.get_or_create(
                    product=product,
                    location=location,
                    defaults={'quantity': 0}
                )
                
                # Apply the signed quantity change
                stock_record.quantity = models.F('quantity') + quantity_change
                stock_record.save() 

                # Save Line Item
                AdjustmentLineItem.objects.create(
                    adjustment=adjustment,
                    product=product,
                    location=location,
                    quantity_change=quantity_change
                )
                
        return adjustment

# Serializer for Hierarchical Lookups
class LocationSerializer(serializers.ModelSerializer):
    """Serializer for managing locations and warehouse hierarchy."""
    class Meta:
        model = Location
        fields = ['id', 'name', 'parent', 'children']
        
class ProductCategorySerializer(serializers.ModelSerializer):
    """Serializer for managing product category hierarchy."""
    class Meta:
        model = ProductCategory
        fields = ['id', 'name', 'parent', 'children']

class UnitOfMeasureSerializer(serializers.ModelSerializer):
    """Serializer for managing units of measure."""
    class Meta:
        model = UnitOfMeasure
        fields = ['id', 'name']

# --- Stock Lookup (for inventory tracking by location) ---

class StockByLocationSerializer(serializers.ModelSerializer):
    product_sku = serializers.CharField(source='product.sku', read_only=True)
    product_name = serializers.CharField(source='product.name', read_only=True)
    location_name = serializers.CharField(source='location.name', read_only=True)
    
    class Meta:
        model = StockByLocation
        fields = ['id', 'product', 'location', 'product_sku', 'product_name', 'location_name', 'quantity', 'updated_at']
