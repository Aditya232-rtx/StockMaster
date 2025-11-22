""
Setup script for the Inventory Anomaly Detection project.
Creates the necessary directory structure and copies data files.
"""

import os
import shutil
import sys
from pathlib import Path

def setup_project_structure() -> None:
    """Set up the project directory structure."""
    # Define paths
    project_dir = Path(__file__).parent
    data_dir = project_dir / 'data'
    output_dir = project_dir / 'output'
    models_dir = output_dir / 'models'
    reports_dir = output_dir / 'reports'
    output_data_dir = output_dir / 'data'
    
    # Create directories if they don't exist
    for directory in [data_dir, models_dir, reports_dir, output_data_dir]:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Copy data files if they exist in the project root
    source_dir = project_dir.parent
    data_files = [
        'ml_data_products.csv',
        'ml_data_transactions.csv',
        'ml_data_vendor_prices.csv'
    ]
    
    for file in data_files:
        src = source_dir / file
        dst = data_dir / file
        
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"Copied {file} to {dst}")
        elif not src.exists():
            print(f"Warning: Source file not found: {src}")
        else:
            print(f"File already exists: {dst}")
    
    print("\nProject setup complete!")
    print("\nNext steps:")
    print("1. Install the required packages: pip install -r requirements.txt")
    print("2. Run the anomaly detection system: python -m src.main --train --data-dir data --output-dir output")

if __name__ == "__main__":
    setup_project_structure()
