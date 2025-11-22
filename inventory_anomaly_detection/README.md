# Inventory Anomaly Detection System

An end-to-end machine learning system for detecting anomalies in inventory transactions using Isolation Forest and ensemble methods.

## Features

- **Data Processing**: Handles loading, cleaning, and preparing inventory transaction data
- **Feature Engineering**: Creates 25+ temporal, behavioral, and quantitative features
- **Anomaly Detection**: Implements Isolation Forest with ensemble methods for robust detection
- **Risk Assessment**: Classifies anomalies into CRITICAL, HIGH, MEDIUM, and LOW risk levels
- **Evaluation**: Provides comprehensive model evaluation metrics and visualizations
- **Deployment**: Includes scripts for training, evaluating, and making predictions

## Requirements

- Python 3.8+
- Required packages are listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd inventory_anomaly_detection
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
inventory_anomaly_detection/
├── data/                    # Directory for input data files
│   ├── ml_data_products.csv
│   ├── ml_data_transactions.csv
│   └── ml_data_vendor_prices.csv
├── src/                     # Source code
│   ├── __init__.py
│   ├── data_processor.py    # Data loading and preprocessing
│   ├── feature_engineer.py  # Feature engineering
│   ├── anomaly_detector.py  # Anomaly detection model
│   ├── evaluator.py         # Model evaluation
│   └── main.py              # Command-line interface
├── output/                  # Output directory for results
│   ├── models/              # Saved models
│   ├── reports/             # Evaluation reports and plots
│   └── data/                # Processed data and predictions
├── app/                     # FastAPI micro frontend for interactive exploration
│   ├── __init__.py
│   ├── api.py               # FastAPI application entry point
│   ├── service.py           # Service layer orchestrating predictions
│   └── static/              # Micro frontend assets (HTML/CSS/JS)
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Usage

### Training a New Model

To train a new anomaly detection model:

```bash
python -m src.main --train --data-dir data --output-dir output
```

### Evaluating the Model

To evaluate the trained model:

```bash
python -m src.main --evaluate --data-dir data --output-dir output --model-path output/models/inventory_anomaly_detector.joblib
```

### Making Predictions

To make predictions on new data:

```bash
python -m src.main --predict --data-dir data --output-dir output --model-path output/models/inventory_anomaly_detector.joblib
```

### Running the Micro Frontend (FastAPI + Uvicorn)

Use the bundled FastAPI service to explore anomalies from your browser:

```bash
pip install -r requirements.txt
uvicorn app.api:app --reload --port 8000
```

Then navigate to http://127.0.0.1:8000/ to:

1. Adjust model parameters (contamination, estimators, etc.)
2. Trigger the pipeline to retrain if required and generate predictions
3. Review a summary dashboard and the top-N anomalous transactions

The API also exposes:

- `GET /health` – lightweight readiness probe
- `POST /predict` – JSON endpoint returning summary statistics and anomaly records

### All Options

```
usage: main.py [-h] [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR]
               [--contamination CONTAMINATION] [--n-estimators N_ESTIMATORS]
               [--random-state RANDOM_STATE] [--train] [--evaluate] [--predict]
               [--model-path MODEL_PATH]

Inventory Anomaly Detection System

optional arguments:
  -h, --help            show this help message and exit
  --data-dir DATA_DIR   Directory containing the input data files
  --output-dir OUTPUT_DIR
                        Directory to save outputs
  --contamination CONTAMINATION
                        Expected proportion of anomalies in the data (0-0.5)
  --n-estimators N_ESTIMATORS
                        Number of base estimators in the ensemble
  --random-state RANDOM_STATE
                        Random seed for reproducibility
  --train               Train a new model
  --evaluate            Evaluate model performance
  --predict             Make predictions on new data
  --model-path MODEL_PATH
                        Path to save/load the model
```

## Model Details

The anomaly detection system uses an ensemble of Isolation Forest models to detect unusual patterns in inventory transactions. Key features include:

- **Temporal Features**: Time-based patterns (hour, day, week, month, etc.)
- **Behavioral Features**: Transaction frequency, location patterns, user behavior
- **Quantitative Features**: Transaction values, inventory levels, reorder points

Anomalies are classified into four risk levels:
- **CRITICAL**: High confidence, high impact anomalies
- **HIGH**: High confidence, medium impact anomalies
- **MEDIUM**: Medium confidence anomalies
- **LOW**: Low confidence anomalies

## Output

The system generates the following outputs:

- **Trained Model**: Saved in the `output/models` directory
- **Evaluation Reports**: Saved in the `output/reports` directory, including:
  - Confusion matrix
  - ROC curve
  - Precision-recall curve
  - Score distribution
  - Detailed metrics report
- **Predictions**: CSV file with anomaly scores and risk levels for each transaction

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For any questions or feedback, please open an issue in the repository.
