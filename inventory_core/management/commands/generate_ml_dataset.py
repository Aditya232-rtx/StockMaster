"""
Django management command to generate comprehensive ML dataset.

Generates 3.5 years of realistic inventory data (June 2021 - Dec 2024):
- 100 products across 4 categories
- 50 vendors with varying reliability
- 8 locations (hierarchical)
- ~7,770 transactions with seasonal patterns
- Price history with inflation and market shocks
- CSV exports for ML training
"""

import random
import csv
from datetime import datetime, timedelta, date
from decimal import Decimal
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone
from inventory_core.models import (
    UnitOfMeasure, ProductCategory, Product, Location, Vendor,
    PurchaseOrder, PurchaseOrderLineItem, Receipt, ReceiptLineItem,
    DeliveryOrder, DeliveryLineItem, InternalTransfer, TransferLineItem,
    StockAdjustment, AdjustmentLineItem, StockByLocation, StockTransaction,
    PriceHistory
)
from django.contrib.contenttypes.models import ContentType


class Command(BaseCommand):
    help = 'Generate 3.5 years of ML training dataset'

    def __init__(self):
        super().__init__()
        self.start_date = date(2010, 1, 1)  # Extended to 15 years
        self.end_date = date(2024, 12, 31)
        self.products = []
        self.vendors = []
        self.locations = []
        self.categories = []
        
    def add_arguments(self, parser):
        parser.add_argument(
            '--export-csv',
            action='store_true',
            help='Export data to CSV files after generation',
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting ML dataset generation...'))
        
        # Clear existing data
        self.stdout.write('Clearing existing data...')
        self.clear_existing_data()
        
        with transaction.atomic():
            self.stdout.write('Creating base data...')
            self.create_base_data()
            
            self.stdout.write('Generating 3.5 years of transactions...')
            self.generate_transactions()
            
            self.stdout.write(self.style.SUCCESS('✓ Dataset generation complete!'))
            
        if options['export_csv']:
            self.stdout.write('Exporting to CSV...')
            self.export_to_csv()
            self.stdout.write(self.style.SUCCESS('✓ CSV export complete!'))
            
        self.print_summary()

    def clear_existing_data(self):
        """Clear existing generated data to start fresh"""
        StockTransaction.objects.all().delete()
        StockByLocation.objects.all().delete()
        PriceHistory.objects.all().delete()
        ReceiptLineItem.objects.all().delete()
        DeliveryLineItem.objects.all().delete()
        TransferLineItem.objects.all().delete()
        AdjustmentLineItem.objects.all().delete()
        PurchaseOrderLineItem.objects.all().delete()
        Receipt.objects.all().delete()
        DeliveryOrder.objects.all().delete()
        InternalTransfer.objects.all().delete()
        StockAdjustment.objects.all().delete()
        PurchaseOrder.objects.all().delete()
        Product.objects.all().delete()
        Vendor.objects.all().delete()
        Location.objects.all().delete()
        ProductCategory.objects.all().delete()
        UnitOfMeasure.objects.all().delete()
        self.stdout.write('  ✓ Existing data cleared')

    def create_base_data(self):
        """Create foundational data: UOM, Categories, Products, Locations, Vendors"""
        
        # Units of Measure
        uoms = [
            ('Kilogram', 'kg'),
            ('Units', 'pcs'),
            ('Liter', 'L'),
            ('Meter', 'm'),
            ('Box', 'box'),
        ]
        uom_objects = {}
        for name, symbol in uoms:
            uom, _ = UnitOfMeasure.objects.get_or_create(name=name, symbol=symbol)
            uom_objects[symbol] = uom  # Use symbol as key
        
        # Categories
        categories_data = [
            ('Raw Materials', None),
            ('Finished Goods', None),
            ('Components', None),
            ('Consumables', None),
        ]
        for cat_name, parent_name in categories_data:
            cat, _ = ProductCategory.objects.get_or_create(name=cat_name)
            self.categories.append(cat)
        
        # Locations (hierarchical)
        main_wh, _ = Location.objects.get_or_create(
            name='Main Warehouse',
            defaults={'is_warehouse': True, 'address': '123 Main St'}
        )
        
        locations_data = [
            ('Rack A', main_wh),
            ('Rack B', main_wh),
            ('Production Floor 1', main_wh),
            ('Production Floor 2', main_wh),
            ('Regional Warehouse East', None),
            ('Regional Warehouse West', None),
            ('Retail Store Downtown', None),
        ]
        
        self.locations = [main_wh]
        for loc_name, parent in locations_data:
            loc, _ = Location.objects.get_or_create(
                name=loc_name,
                defaults={'parent': parent, 'is_warehouse': parent is None}
            )
            self.locations.append(loc)
        
        # Vendors (200 vendors) - INCREASED
        self.stdout.write('  Creating 200 vendors...')
        vendor_prefixes = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta', 'Iota', 'Kappa', 'Lambda', 'Sigma']
        vendor_types = ['Supplies', 'Materials', 'Components', 'Industries', 'Corp', 'Ltd', 'Inc', 'Group', 'Trading', 'International']
        
        for i in range(200):  # INCREASED from 50 to 200
            prefix = random.choice(vendor_prefixes)
            vtype = random.choice(vendor_types)
            code = f'VEN-{i+1:04d}'  # Changed to 4 digits
            name = f'{prefix} {vtype} #{i+1}'
            
            # Reliability tiers
            if i < 60:  # Excellent (30%)
                reliability = random.randint(95, 100)
            elif i < 160:  # Good (50%)
                reliability = random.randint(85, 95)
            else:  # Average (20%)
                reliability = random.randint(70, 85)
            
            vendor, _ = Vendor.objects.get_or_create(
                code=code,
                defaults={
                    'name': name,
                    'email': f'{code.lower()}@example.com',
                    'phone': f'+1-555-{random.randint(1000,9999)}',
                    'reliability_score': reliability,
                    'payment_terms': random.choice([15, 30, 45, 60]),
                    'is_active': True
                }
            )
            self.vendors.append(vendor)
        
        # Products (500 products) - INCREASED
        self.stdout.write('  Creating 500 products...')
        product_templates = {
            'Raw Materials': [
                ('Steel Rods', 'kg', 25, 50),
                ('Aluminum Sheets', 'kg', 30, 40),
                ('Copper Wire', 'kg', 45, 35),
                ('Plastic Pellets', 'kg', 15, 60),
                ('Rubber Compound', 'kg', 20, 45),
                ('Titanium Alloy', 'kg', 120, 20),
                ('Carbon Fiber', 'kg', 85, 25),
                ('Stainless Steel', 'kg', 35, 45),
            ],
            'Finished Goods': [
                ('Widget Model A', 'pcs', 150, 25),
                ('Widget Model B', 'pcs', 200, 20),
                ('Assembly Unit X', 'pcs', 350, 15),
                ('Assembly Unit Y', 'pcs', 400, 12),
                ('Premium Widget', 'pcs', 550, 10),
                ('Industrial Unit', 'pcs', 750, 8),
            ],
            'Components': [
                ('Bearing Type 1', 'pcs', 5, 100),
                ('Bolt M8', 'pcs', 0.5, 500),
                ('Circuit Board', 'pcs', 25, 50),
                ('Motor 12V', 'pcs', 35, 40),
                ('Sensor Module', 'pcs', 45, 35),
                ('Control Unit', 'pcs', 65, 30),
            ],
            'Consumables': [
                ('Lubricant Oil', 'L', 12, 30),
                ('Cleaning Solution', 'L', 8, 40),
                ('Packaging Tape', 'box', 3, 100),
                ('Adhesive', 'L', 15, 35),
            ],
        }
        
        product_counter = 1
        for cat_name, templates in product_templates.items():
            category = next(c for c in self.categories if c.name == cat_name)
            products_per_template = 500 // len(self.categories) // len(templates)  # INCREASED
            
            for template_name, uom_name, base_cost, threshold in templates:
                for variant in range(products_per_template):
                    sku = f'PRD-{product_counter:05d}'  # Changed to 5 digits
                    name = f'{template_name} Variant {variant+1}' if variant > 0 else template_name
                    
                    # Demand tier
                    if product_counter <= 100:  # High demand (20%)
                        reorder_qty = random.randint(300, 600)
                    elif product_counter <= 350:  # Medium demand (50%)
                        reorder_qty = random.randint(80, 250)
                    else:  # Low demand (30%)
                        reorder_qty = random.randint(30, 80)
                    
                    product, _ = Product.objects.get_or_create(
                        sku=sku,
                        defaults={
                            'name': name,
                            'category': category,
                            'uom': uom_objects[uom_name],
                            'unit_cost': Decimal(str(base_cost * (0.8 + random.random() * 0.4))),
                            'low_stock_threshold': threshold,
                            'reorder_quantity': reorder_qty,
                            'lead_time_days': random.randint(3, 14),
                        }
                    )
                    self.products.append(product)
                    product_counter += 1
                    
                    if product_counter > 500:  # INCREASED
                        break
                if product_counter > 500:  # INCREASED
                    break

    def generate_transactions(self):
        """Generate 3.5 years of transactions with realistic patterns"""
        
        current_date = self.start_date
        month_counter = 0
        
        # Track stock levels
        stock_levels = {(p.id, l.id): 0 for p in self.products for l in self.locations[:4]}  # Main locations
        
        while current_date <= self.end_date:
            # Monthly progress
            if current_date.day == 1:
                month_counter += 1
                self.stdout.write(f'  Month {month_counter}/180: {current_date.strftime("%b %Y")}')  # 15 years = 180 months
            
            # Skip weekends for most operations
            is_weekend = current_date.weekday() >= 5
            
            # Seasonal multiplier
            month = current_date.month
            if month in [1, 2, 3]:  # Q1
                seasonal_mult = 1.0
            elif month in [4, 5, 6]:  # Q2
                seasonal_mult = 1.15
            elif month in [7, 8, 9]:  # Q3
                seasonal_mult = 1.30
            else:  # Q4
                seasonal_mult = 1.50
            
            # Year-over-year growth (10% annual)
            year_mult = 1.0 + (0.1 * (current_date.year - 2021))
            
            # COVID impact (2021-2022)
            if current_date.year <= 2022:
                covid_mult = 0.85  # 15% reduction
            else:
                covid_mult = 1.0
            
            daily_mult = seasonal_mult * year_mult * covid_mult
            
            # Generate receipts (~4 per day avg) - INCREASED for larger dataset
            if not is_weekend and random.random() < (4.0 * daily_mult / 30):
                self.generate_receipt(current_date, stock_levels)
            
            # Generate deliveries (~8 per day avg) - INCREASED
            if random.random() < (8.0 * daily_mult / 30):
                self.generate_delivery(current_date, stock_levels)
            
            # Generate transfers (~2 per day avg) - INCREASED
            if not is_weekend and random.random() < (2.0 * daily_mult / 30):
                self.generate_transfer(current_date, stock_levels)
            
            # Generate adjustments (~1 per day avg) - INCREASED
            if random.random() < (1.0 / 30):
                self.generate_adjustment(current_date, stock_levels)
            
            current_date += timedelta(days=1)

    def generate_receipt(self, current_date, stock_levels):
        """Generate a purchase order and receipt"""
        vendor = random.choice(self.vendors)
        location = random.choice(self.locations[:4])  # Main locations
        
        # Create PO
        po_number = f'PO-{current_date.strftime("%Y%m%d")}-{random.randint(1000,9999)}'
        lead_time = random.randint(3, 14)
        expected_delivery = current_date + timedelta(days=lead_time)
        
        po = PurchaseOrder.objects.create(
            po_number=po_number,
            vendor=vendor,
            expected_delivery_date=expected_delivery,
            status='RECEIVED'
        )
        
        # Add 1-5 products
        num_products = random.randint(1, 5)
        selected_products = random.sample(self.products, num_products)
        total_amount = Decimal('0')
        
        for product in selected_products:
            # Get or create price
            price = self.get_vendor_price(product, vendor, current_date)
            quantity = Decimal(str(random.randint(50, 500)))
            
            PurchaseOrderLineItem.objects.create(
                purchase_order=po,
                product=product,
                quantity=quantity,
                unit_price=price,
                received_quantity=quantity
            )
            total_amount += price * quantity
        
        po.total_amount = total_amount
        po.save()
        
        # Create Receipt
        receipt_number = f'RCP-{current_date.strftime("%Y%m%d")}-{random.randint(1000,9999)}'
        receipt = Receipt.objects.create(
            document_number=receipt_number,
            vendor=vendor.name,
            purchase_order=po,
            received_date=current_date,
            status='DONE'
        )
        
        receipt_ct = ContentType.objects.get_for_model(Receipt)
        
        for po_item in po.items.all():
            # Create receipt line item
            ReceiptLineItem.objects.create(
                receipt=receipt,
                product=po_item.product,
                location=location,
                quantity=po_item.quantity
            )
            
            # Update stock
            key = (po_item.product.id, location.id)
            stock_levels[key] = stock_levels.get(key, 0) + float(po_item.quantity)
            
            # Create stock transaction with historical timestamp
            txn = StockTransaction(
                product=po_item.product,
                location=location,
                quantity_change=po_item.quantity,
                document_type='RECEIPT',
                content_type=receipt_ct,
                object_id=receipt.pk
            )
            txn.save()
            # Manually update timestamp to historical date
            StockTransaction.objects.filter(pk=txn.pk).update(
                timestamp=timezone.make_aware(datetime.combine(current_date, datetime.min.time()))
            )
            
            # Update or create StockByLocation
            stock_record, _ = StockByLocation.objects.get_or_create(
                product=po_item.product,
                location=location,
                defaults={'quantity': 0}
            )
            stock_record.quantity += po_item.quantity
            stock_record.save()

    def generate_delivery(self, current_date, stock_levels):
        """Generate a delivery order"""
        location = random.choice(self.locations[:4])
        
        # Select products with available stock
        available_products = [
            p for p in self.products
            if stock_levels.get((p.id, location.id), 0) > 50
        ]
        
        if not available_products:
            return
        
        delivery_number = f'DEL-{current_date.strftime("%Y%m%d")}-{random.randint(1000,9999)}'
        delivery = DeliveryOrder.objects.create(
            document_number=delivery_number,
            customer=f'Customer-{random.randint(1,100)}',
            status='DONE'
        )
        
        delivery_ct = ContentType.objects.get_for_model(DeliveryOrder)
        
        # Add 1-3 products
        num_products = min(random.randint(1, 3), len(available_products))
        selected_products = random.sample(available_products, num_products)
        
        for product in selected_products:
            key = (product.id, location.id)
            available = stock_levels.get(key, 0)
            quantity = Decimal(str(min(random.randint(10, 100), available * 0.3)))
            
            if quantity <= 0:
                continue
            
            DeliveryLineItem.objects.create(
                delivery=delivery,
                product=product,
                location=location,
                quantity=quantity
            )
            
            # Update stock
            stock_levels[key] -= float(quantity)
            
            # Create stock transaction with historical timestamp
            txn = StockTransaction(
                product=product,
                location=location,
                quantity_change=-quantity,
                document_type='DELIVERY',
                content_type=delivery_ct,
                object_id=delivery.pk
            )
            txn.save()
            StockTransaction.objects.filter(pk=txn.pk).update(
                timestamp=timezone.make_aware(datetime.combine(current_date, datetime.min.time()))
            )
            
            # Update StockByLocation
            stock_record = StockByLocation.objects.get(product=product, location=location)
            stock_record.quantity -= quantity
            stock_record.save()

    def generate_transfer(self, current_date, stock_levels):
        """Generate an internal transfer"""
        source = random.choice(self.locations[:4])
        dest = random.choice([l for l in self.locations[:4] if l != source])
        
        # Find products with stock at source
        available_products = [
            p for p in self.products
            if stock_levels.get((p.id, source.id), 0) > 30
        ]
        
        if not available_products:
            return
        
        transfer_number = f'TRF-{current_date.strftime("%Y%m%d")}-{random.randint(1000,9999)}'
        transfer = InternalTransfer.objects.create(
            document_number=transfer_number,
            source_location=source,
            destination_location=dest,
            status='DONE'
        )
        
        transfer_ct = ContentType.objects.get_for_model(InternalTransfer)
        
        product = random.choice(available_products)
        key_source = (product.id, source.id)
        key_dest = (product.id, dest.id)
        
        available = stock_levels.get(key_source, 0)
        quantity = Decimal(str(min(random.randint(20, 100), available * 0.4)))
        
        if quantity <= 0:
            return
        
        TransferLineItem.objects.create(
            transfer=transfer,
            product=product,
            quantity=quantity
        )
        
        # Update stock levels
        stock_levels[key_source] -= float(quantity)
        stock_levels[key_dest] = stock_levels.get(key_dest, 0) + float(quantity)
        
        # Create transactions (debit and credit) with historical timestamp
        txn1 = StockTransaction(
            product=product,
            location=source,
            quantity_change=-quantity,
            document_type='TRANSFER',
            content_type=transfer_ct,
            object_id=transfer.pk
        )
        txn1.save()
        StockTransaction.objects.filter(pk=txn1.pk).update(
            timestamp=timezone.make_aware(datetime.combine(current_date, datetime.min.time()))
        )
        
        txn2 = StockTransaction(
            product=product,
            location=dest,
            quantity_change=quantity,
            document_type='TRANSFER',
            content_type=transfer_ct,
            object_id=transfer.pk
        )
        txn2.save()
        StockTransaction.objects.filter(pk=txn2.pk).update(
            timestamp=timezone.make_aware(datetime.combine(current_date, datetime.min.time()))
        )
        
        # Update StockByLocation
        source_stock = StockByLocation.objects.get(product=product, location=source)
        source_stock.quantity -= quantity
        source_stock.save()
        
        dest_stock, _ = StockByLocation.objects.get_or_create(
            product=product,
            location=dest,
            defaults={'quantity': 0}
        )
        dest_stock.quantity += quantity
        dest_stock.save()

    def generate_adjustment(self, current_date, stock_levels):
        """Generate a stock adjustment (shrinkage/damage)"""
        location = random.choice(self.locations[:4])
        product = random.choice(self.products)
        
        key = (product.id, location.id)
        available = stock_levels.get(key, 0)
        
        if available < 10:
            return
        
        # Small adjustment (1-2% shrinkage)
        max_adjustment = max(1, int(available * 0.02))
        quantity_change = -Decimal(str(random.randint(1, max_adjustment)))
        
        adjustment_number = f'ADJ-{current_date.strftime("%Y%m%d")}-{random.randint(1000,9999)}'
        adjustment = StockAdjustment.objects.create(
            document_number=adjustment_number,
            reason=random.choice(['Damaged', 'Lost', 'Expired', 'Quality Issue']),
            status='DONE'
        )
        
        adjustment_ct = ContentType.objects.get_for_model(StockAdjustment)
        
        AdjustmentLineItem.objects.create(
            adjustment=adjustment,
            product=product,
            location=location,
            quantity_change=quantity_change
        )
        
        # Update stock
        stock_levels[key] += float(quantity_change)
        
        # Create transaction with historical timestamp
        txn = StockTransaction(
            product=product,
            location=location,
            quantity_change=quantity_change,
            document_type='ADJUSTMENT',
            content_type=adjustment_ct,
            object_id=adjustment.pk
        )
        txn.save()
        StockTransaction.objects.filter(pk=txn.pk).update(
            timestamp=timezone.make_aware(datetime.combine(current_date, datetime.min.time()))
        )
        
        # Update StockByLocation
        try:
            stock_record = StockByLocation.objects.get(product=product, location=location)
            stock_record.quantity += quantity_change
            stock_record.save()
        except StockByLocation.DoesNotExist:
            pass

    def get_vendor_price(self, product, vendor, current_date):
        """Get or create vendor price with inflation"""
        # Check existing price
        existing_price = PriceHistory.objects.filter(
            product=product,
            vendor=vendor,
            effective_date__lte=current_date
        ).order_by('-effective_date').first()
        
        if existing_price:
            # Apply inflation (3-5% annual)
            years_diff = (current_date - existing_price.effective_date).days / 365
            inflation_rate = 1 + (random.uniform(0.03, 0.05) * years_diff)
            new_price = existing_price.price * Decimal(str(inflation_rate))
            
            # Only create new price if significant change (>5%)
            if abs(new_price - existing_price.price) / existing_price.price > 0.05:
                PriceHistory.objects.create(
                    product=product,
                    vendor=vendor,
                    price=new_price.quantize(Decimal('0.01')),
                    effective_date=current_date
                )
                return new_price.quantize(Decimal('0.01'))
            return existing_price.price
        else:
            # Create initial price (base cost + 20-50% markup)
            markup = Decimal(str(1.2 + random.random() * 0.3))
            price = (product.unit_cost * markup).quantize(Decimal('0.01'))
            
            PriceHistory.objects.create(
                product=product,
                vendor=vendor,
                price=price,
                effective_date=current_date
            )
            return price

    def export_to_csv(self):
        """Export data to CSV files for ML training"""
        
        # 1. Transactions history for demand forecasting
        with open('ml_data_transactions.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'date', 'product_sku', 'product_name', 'category', 'transaction_type',
                'quantity_change', 'location', 'day_of_week', 'month', 'quarter', 'year'
            ])
            
            for txn in StockTransaction.objects.select_related('product', 'location').all():
                writer.writerow([
                    txn.timestamp.date(),
                    txn.product.sku,
                    txn.product.name,
                    txn.product.category.name if txn.product.category else '',
                    txn.document_type,
                    float(txn.quantity_change),
                    txn.location.name,
                    txn.timestamp.weekday(),
                    txn.timestamp.month,
                    (txn.timestamp.month - 1) // 3 + 1,
                    txn.timestamp.year
                ])
        
        self.stdout.write('  ✓ ml_data_transactions.csv')
        
        # 2. Vendor prices for price prediction
        with open('ml_data_vendor_prices.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'product_sku', 'vendor_code', 'vendor_name', 'price', 'effective_date',
                'vendor_reliability', 'product_category', 'product_unit_cost'
            ])
            
            for price in PriceHistory.objects.select_related('product', 'vendor').all():
                writer.writerow([
                    price.product.sku,
                    price.vendor.code,
                    price.vendor.name,
                    float(price.price),
                    price.effective_date,
                    price.vendor.reliability_score,
                    price.product.category.name if price.product.category else '',
                    float(price.product.unit_cost)
                ])
        
        self.stdout.write('  ✓ ml_data_vendor_prices.csv')
        
        # 3. Product master data
        with open('ml_data_products.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'sku', 'name', 'category', 'unit_cost', 'low_stock_threshold',
                'reorder_quantity', 'lead_time_days'
            ])
            
            for product in Product.objects.select_related('category').all():
                writer.writerow([
                    product.sku,
                    product.name,
                    product.category.name if product.category else '',
                    float(product.unit_cost),
                    product.low_stock_threshold,
                    product.reorder_quantity,
                    product.lead_time_days
                ])
        
        self.stdout.write('  ✓ ml_data_products.csv')

    def print_summary(self):
        """Print dataset summary"""
        self.stdout.write('\n' + '='*60)
        self.stdout.write(self.style.SUCCESS('DATASET SUMMARY'))
        self.stdout.write('='*60)
        self.stdout.write(f'Products: {Product.objects.count()}')
        self.stdout.write(f'Vendors: {Vendor.objects.count()}')
        self.stdout.write(f'Locations: {Location.objects.count()}')
        self.stdout.write(f'Purchase Orders: {PurchaseOrder.objects.count()}')
        self.stdout.write(f'Receipts: {Receipt.objects.count()}')
        self.stdout.write(f'Deliveries: {DeliveryOrder.objects.count()}')
        self.stdout.write(f'Transfers: {InternalTransfer.objects.count()}')
        self.stdout.write(f'Adjustments: {StockAdjustment.objects.count()}')
        self.stdout.write(f'Stock Transactions: {StockTransaction.objects.count()}')
        self.stdout.write(f'Price History Records: {PriceHistory.objects.count()}')
        self.stdout.write('='*60)
