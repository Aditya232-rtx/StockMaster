from django.core.management.base import BaseCommand
from django.db import transaction
from inventory_core.models import UnitOfMeasure, Location, ProductCategory, Product, Receipt, StockByLocation
from inventory_core.serializers import ReceiptSerializer

class Command(BaseCommand):
    help = 'Populates the database with initial sample data for StockMaster prototype.'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('--- Starting StockMaster Data Seeding ---'))

        with transaction.atomic():
            # 1. Units of Measure
            uom_kg, _ = UnitOfMeasure.objects.get_or_create(name='Kilogram', defaults={'symbol': 'kg'})
            uom_units, _ = UnitOfMeasure.objects.get_or_create(name='Units', defaults={'symbol': 'pcs'})
            self.stdout.write(self.style.SUCCESS('1. Created Unit of Measures.'))

            # 2. Locations (Hierarchy)
            wh1, _ = Location.objects.get_or_create(name='Main Warehouse', parent=None, defaults={'is_warehouse': True})
            rack_a, _ = Location.objects.get_or_create(name='Rack A', parent=wh1)
            prod_floor, _ = Location.objects.get_or_create(name='Production Floor', parent=wh1)
            self.stdout.write(self.style.SUCCESS('2. Created Location Hierarchy (Main Warehouse, Racks, Prod Floor).'))

            # 3. Categories
            raw_mat, _ = ProductCategory.objects.get_or_create(name='Raw Materials', parent=None)
            finished_goods, _ = ProductCategory.objects.get_or_create(name='Finished Goods', parent=None)
            self.stdout.write(self.style.SUCCESS('3. Created Product Categories.'))

            # 4. Products (Including Low Stock Threshold)
            steel, _ = Product.objects.get_or_create(sku='SR-100', defaults={'name': 'Steel Rods (100mm)', 'category': raw_mat, 'uom': uom_kg, 'low_stock_threshold': 50})
            widget, _ = Product.objects.get_or_create(sku='FG-WID-1', defaults={'name': 'Finished Widget', 'category': finished_goods, 'uom': uom_units, 'low_stock_threshold': 200})
            self.stdout.write(self.style.SUCCESS('4. Created Sample Products.'))

            # 5. Initial Transaction (Receipt)
            # Use the atomic logic in the serializer to ensure stock and ledger are updated correctly.
            receipt_data = {
                'document_number': 'R0001',
                'vendor': 'SteelCo Inc.',
                'items': [
                    {'product_sku': 'SR-100', 'location_id': rack_a.id, 'quantity': 500.0}, # 500 kg steel received at Rack A
                    {'product_sku': 'FG-WID-1', 'location_id': wh1.id, 'quantity': 150.0} # 150 widgets received at Main Warehouse
                ]
            }
            
            # Check if receipt already exists to avoid duplication on re-runs
            if not Receipt.objects.filter(document_number='R0001').exists():
                serializer = ReceiptSerializer(data=receipt_data)
                serializer.is_valid(raise_exception=True)
                serializer.save()
                
                self.stdout.write(self.style.SUCCESS(f'5. Created Initial Receipt (R0001) and updated Stock Ledger/State.'))
                self.stdout.write(self.style.SUCCESS(f'Stock Check: Steel Rods at Rack A: {StockByLocation.objects.get(product=steel, location=rack_a).quantity} KG'))
                self.stdout.write(self.style.SUCCESS(f'Stock Check: Finished Widgets at Main Warehouse: {StockByLocation.objects.get(product=widget, location=wh1).quantity} Units'))
            else:
                self.stdout.write(self.style.WARNING('5. Receipt R0001 already exists. Skipping receipt creation.'))

        self.stdout.write(self.style.SUCCESS('--- Data Seeding Complete. Prototype ready for testing. ---'))
