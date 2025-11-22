"""Service layer for exposing anomaly detection functionality via a web API."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.anomaly_detector import AnomalyDetector
from src.data_processor import DataProcessor
from src.feature_engineer import FeatureEngineer


@dataclass
class PredictionSummary:
    total_transactions: int
    anomalies_detected: int
    anomaly_rate: float
    risk_distribution: Dict[str, int]


class InventoryAnomalyService:
    """Facade over the anomaly detection pipeline for serving predictions."""

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        model_path: Optional[Path] = None,
    ) -> None:
        self.data_dir = self._resolve_data_dir(Path(data_dir))
        self.output_dir = Path(output_dir)
        self.model_path = (
            Path(model_path)
            if model_path is not None
            else self.output_dir / "models" / "inventory_anomaly_detector.joblib"
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        self._detector: Optional[AnomalyDetector] = None
        self._model_signature: Optional[Tuple[float, int, int]] = None

    # ------------------------------------------------------------------
    # Model lifecycle helpers
    # ------------------------------------------------------------------
    def ensure_model(
        self, contamination: float, n_estimators: int, random_state: int
    ) -> AnomalyDetector:
        """Load an existing model or train a new one if necessary."""
        signature = (contamination, n_estimators, random_state)

        if self._detector is not None and self._model_signature == signature:
            return self._detector

        if self.model_path.exists():
            detector = AnomalyDetector.load_model(self.model_path)
            current_signature = (
                detector.contamination,
                detector.n_estimators,
                detector.random_state,
            )
            if current_signature == signature:
                self._detector = detector
                self._model_signature = signature
                return detector

        features, feature_matrix = self._prepare_features()

        detector = AnomalyDetector(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        detector.fit(feature_matrix)
        detector.save_model(self.model_path)

        self._detector = detector
        self._model_signature = signature
        return detector

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------
    def generate_predictions(
        self,
        contamination: float,
        n_estimators: int,
        random_state: int,
        top_n: int = 20,
    ) -> Tuple[pd.DataFrame, PredictionSummary]:
        """Produce anomaly predictions and a high-level summary."""
        features, feature_matrix = self._prepare_features()
        detector = self.ensure_model(contamination, n_estimators, random_state)

        scores = detector.score_samples(feature_matrix)
        predictions = detector.predict(feature_matrix)
        risk_levels = detector.get_risk_levels(scores)

        results = features.copy()
        results["anomaly_score"] = scores
        results["prediction"] = predictions
        results["risk_level"] = risk_levels

        sorted_results = results.sort_values("anomaly_score", ascending=False)
        top_results = sorted_results.head(top_n)

        anomalies_detected = int((predictions == -1).sum())
        summary = PredictionSummary(
            total_transactions=len(results),
            anomalies_detected=anomalies_detected,
            anomaly_rate=float(anomalies_detected / max(len(results), 1)),
            risk_distribution=self._compute_risk_distribution(risk_levels),
        )

        return top_results, summary

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _prepare_features(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data, engineer features, and return feature matrix."""
        processor = DataProcessor(str(self.data_dir))
        processor.load_data()
        processor.clean_data()
        products, transactions, _ = processor.get_processed_data()

        feature_engineer = FeatureEngineer(transactions, products)
        features = feature_engineer.create_all_features()
        feature_cols = feature_engineer.get_feature_columns()

        feature_matrix = features[feature_cols].fillna(0)
        return features, feature_matrix

    @staticmethod
    def _compute_risk_distribution(scores: np.ndarray) -> Dict[str, int]:
        unique, counts = np.unique(scores, return_counts=True)
        return {str(level): int(count) for level, count in zip(unique, counts)}

    @staticmethod
    def _resolve_data_dir(candidate: Path) -> Path:
        required_files = {
            "ml_data_products.csv",
            "ml_data_transactions.csv",
            "ml_data_vendor_prices.csv",
        }

        search_roots = [
            candidate,
            candidate / "data",
            candidate.parent,
            candidate.parent / "data",
        ]

        for root in search_roots:
            if root and root.exists() and all((root / file).exists() for file in required_files):
                return root

        missing_hint = "\n".join(str(root) for root in search_roots if root)
        raise FileNotFoundError(
            "Unable to locate inventory data files. Looked in:\n"
            f"{missing_hint}\n"
            "Ensure ml_data_products.csv, ml_data_transactions.csv, and "
            "ml_data_vendor_prices.csv are present."
        )

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    @staticmethod
    def serialize_results(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert the results DataFrame to JSON-serialisable dictionaries."""
        columns_of_interest = [
            col
            for col in [
                "transaction_id",
                "date",
                "product_sku",
                "transaction_type",
                "quantity_change",
                "location",
                "anomaly_score",
                "risk_level",
            ]
            if col in df.columns
        ]
        if not columns_of_interest:
            return []

        serializable_df = df[columns_of_interest].copy()

        if "date" in serializable_df.columns:
            serializable_df["date"] = serializable_df["date"].astype(str)

        if "anomaly_score" in serializable_df.columns:
            serializable_df["anomaly_score"] = serializable_df["anomaly_score"].astype(float)

        return serializable_df.to_dict(orient="records")

    @staticmethod
    def serialize_summary(summary: PredictionSummary) -> Dict[str, Any]:
        return {
            "total_transactions": summary.total_transactions,
            "anomalies_detected": summary.anomalies_detected,
            "anomaly_rate": summary.anomaly_rate,
            "risk_distribution": summary.risk_distribution,
        }
