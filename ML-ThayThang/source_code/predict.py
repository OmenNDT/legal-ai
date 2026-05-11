from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config.config_loader import CONFIG
from config.paths import PATHS
from ml_pipeline.data_loader import PostgresLoader, CsvLoader
from ml_pipeline.feature_engineering import FeatureBuilder
from ml_pipeline.preprocessing import Preprocessor, FEATURE_COLS

logger = logging.getLogger(__name__)

def setup_logging() -> None:
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
        handlers = [logging.StreamHandler(sys.stdout)]
    )

class DelayPredictor:
    def __init__(self) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self._model_name = self._load_best_model_name()
        self._model = self._load_model()
        self._preprocessor = self._load_preprocessor()

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load_best_model_name(self) -> str:
        path = PATHS.models / "best_model.txt"
        if not path.exists():
            raise FileNotFoundError(f"Best model marker not found: {path}. Run train.py first.")
        name = path.read_text(encoding = "utf-8").strip()
        self._logger.info("Best model selected: %s", name)
        return name

    def _load_model(self):
        path = PATHS.models / f"{self._model_name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}. Run train.py first.")
        self._logger.info("Loading model: %s", path)
        return joblib.load(path)

    def _load_preprocessor(self) -> Preprocessor:
        path = PATHS.models / "preprocessor.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor not found: {path}. Run train.py first.")
        self._logger.info("Loading preprocessor: %s", path)
        return joblib.load(path)

    def _load_tables(self) -> dict:
        try:
            with PostgresLoader(CONFIG.database) as loader:
                return loader.load_all_tables()
        except Exception as e:
            self._logger.warning("PostgreSQL load failed (%s). Falling back to CSV.", e)
            return CsvLoader(PATHS.data_raw).load_all_tables()

    def predict_from_db(self, trip_ids: list[str] | None = None) -> pd.DataFrame:
        tables = self._load_tables()
        builder = FeatureBuilder(tables)
        df = builder.build()

        if trip_ids is not None:
            df = df[df["trip_id"].isin(trip_ids)].copy()
            if df.empty:
                raise ValueError(f"No trips found for IDs: {trip_ids}")

        return self._predict_dataframe(df)

    def _predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        meta_cols = [c for c in ["trip_id", "load_id", "driver_id", "truck_id", "scheduled_datetime"] if c in df.columns]
        meta = df[meta_cols].copy().reset_index(drop = True)
        has_actual = "delay_minutes" in df.columns

        proc = self._preprocessor.transform(df)
        feat_cols = [c for c in FEATURE_COLS if c in proc.columns]
        X = proc[feat_cols]

        y_pred = self._model.predict(X)
        meta["predicted_delay_minutes"] = np.round(y_pred, 2)
        meta["predicted_status"] = np.where(y_pred > 15, "LATE", np.where(y_pred < -15, "EARLY", "ON_TIME"))

        if has_actual:
            meta["actual_delay_minutes"] = df["delay_minutes"].reset_index(drop = True).round(2)
            meta["error_minutes"] = (meta["predicted_delay_minutes"] - meta["actual_delay_minutes"]).round(2)

        return meta

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Predict delivery delay for trips using the best trained model.")
    parser.add_argument("--trip-ids", nargs = "*", default = None, help = "Specific trip_id(s). Default: all trips")
    parser.add_argument("--limit", type = int, default = 20, help = "Show only first N rows in output")
    parser.add_argument("--output", default = None, help = "Optional CSV path to save full predictions")
    args = parser.parse_args()

    setup_logging()

    predictor = DelayPredictor()
    result = predictor.predict_from_db(trip_ids = args.trip_ids)

    print("\n" + "=" * 80)
    print(f"PREDICTIONS using {predictor.model_name} (showing first {min(args.limit, len(result))} of {len(result)})")
    print("=" * 80)
    print(result.head(args.limit).to_string(index = False))

    if "actual_delay_minutes" in result.columns:
        mae = result["error_minutes"].abs().mean()
        within_30 = (result["error_minutes"].abs() <= 30).mean() * 100
        within_60 = (result["error_minutes"].abs() <= 60).mean() * 100
        print("\n" + "-" * 80)
        print(f"Verification ({len(result)} samples with ground truth):")
        print(f"=> MAE: {mae:.2f} min")
        print(f"=> Within 30 min: {within_30:.1f}%")
        print(f"=> Within 60 min: {within_60:.1f}%")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents = True, exist_ok = True)
        result.to_csv(out_path, index = False)
        print(f"\nFull predictions saved to {out_path}")
