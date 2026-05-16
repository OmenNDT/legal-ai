from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config.config_loader import CONFIG
from config.paths import PATHS
from ml_pipeline.data_loader import PostgresLoader, CsvLoader
from ml_pipeline.feature_engineering import FeatureBuilder
from ml_pipeline.preprocessing import Preprocessor, DataSplitter
from ml_pipeline.models import BaseModel, LinearRegressionModel, RandomForestModel, XGBoostModel, ModelRegistry
from ml_pipeline.evaluation import ModelEvaluator

def setup_logging() -> None:
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
        handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(Path(__file__).parent / "train.log", mode = "w", encoding = "utf-8")]
    )

class TrainingPipeline:
    def __init__(self) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        PATHS.ensure_dirs()

    def run(self) -> None:
        self._logger.info("========================================")
        self._logger.info("  DELIVERY DELAY PREDICTION - PIPELINE  ")
        self._logger.info("========================================")

        tables = self._load_data()
        df_master = self._build_features(tables)
        train_proc, val_proc, test_proc = self._preprocess_and_split(df_master)
        registry = self._train_models(train_proc, val_proc)
        self._evaluate_and_report(registry, test_proc)

        self._logger.info("Pipeline finished successfully.")

    def _load_data(self) -> dict:
        self._logger.info("")
        self._logger.info("--- STEP 1: Load data from PostgreSQL ---")
        try:
            with PostgresLoader(CONFIG.database) as loader:
                tables = loader.load_all_tables()
        except Exception as e:
            self._logger.warning("PostgreSQL load failed (%s). Falling back to CSV: %s", e, PATHS.data_raw)
            tables = CsvLoader(PATHS.data_raw).load_all_tables()
        self._logger.info("Tables loaded: %s", list(tables.keys()))
        for name, df in tables.items():
            self._logger.info("%s: %d rows", name, len(df))
        return tables

    def _build_features(self, tables: dict) -> object:
        self._logger.info("")
        self._logger.info("--- STEP 2: Feature engineering ---")
        builder = FeatureBuilder(tables)
        df = builder.build()
        self._logger.info(
            "Master dataset: %d rows | target = delay_minutes | "
            "mean = %.1f | std = %.1f | min = %.1f | max = %.1f",
            len(df),
            df["delay_minutes"].mean(),
            df["delay_minutes"].std(),
            df["delay_minutes"].min(),
            df["delay_minutes"].max()
        )
        return df

    def _preprocess_and_split(self, df) -> tuple:
        self._logger.info("")
        self._logger.info("--- STEP 3: Preprocessing & split ---")
        splitter = DataSplitter(train_ratio = 0.70, val_ratio = 0.15)
        train_raw, val_raw, test_raw = splitter.split(df)

        preprocessor = Preprocessor()
        train_proc = preprocessor.fit_transform(train_raw)
        val_proc = preprocessor.transform(val_raw)
        test_proc = preprocessor.transform(test_raw)

        import joblib
        preprocessor_path = PATHS.models / "preprocessor.pkl"
        joblib.dump(preprocessor, preprocessor_path)
        self._logger.info("Preprocessor saved to %s", preprocessor_path)

        self._logger.info("Processed shapes - train: %s | val: %s | test: %s", train_proc.shape, val_proc.shape, test_proc.shape)
        return train_proc, val_proc, test_proc

    def _train_models(self, train_proc, val_proc) -> ModelRegistry:
        self._logger.info("")
        self._logger.info("--- STEP 4: Training models ---")
        splitter = DataSplitter()
        X_train, y_train = splitter.get_xy(train_proc)
        X_val, y_val = splitter.get_xy(val_proc)

        self._logger.info("Training features: %d | train samples: %d | val samples: %d", X_train.shape[1], len(X_train), len(X_val))

        registry = ModelRegistry()

        lr = LinearRegressionModel()
        lr.fit(X_train, y_train)
        registry.register(lr)

        rf = RandomForestModel()
        rf.fit(X_train, y_train)
        registry.register(rf)

        xgb_model = XGBoostModel()
        xgb_model.fit(X_train, y_train, X_val = X_val, y_val = y_val)
        registry.register(xgb_model)

        model_dir = PATHS.models
        for model in registry.all_models():
            model.save(model_dir / f"{model.name}.pkl")

        return registry

    def _evaluate_and_report(self, registry: ModelRegistry, test_proc) -> None:
        self._logger.info("")
        self._logger.info("--- STEP 5: Evaluation on test set ---")
        splitter = DataSplitter()
        X_test, y_test = splitter.get_xy(test_proc)
        feat_cols = list(X_test.columns)

        evaluator = ModelEvaluator()

        print("\n" + "=" * 70)
        print("DELIVERY DELAY PREDICTION - TEST SET EVALUATION RESULTS")
        print("=" * 70)
        print(f"Test samples: {len(X_test)}")
        print(f"Features: {len(feat_cols)}")
        print(f"Target: delay_minutes (positive = late, negative = early)")

        y_test_arr = np.asarray(y_test)
        model: BaseModel
        for model in registry.all_models():
            self._logger.info("Evaluating %s on test set...", model.name)
            y_pred = model.predict(X_test)
            evaluator.evaluate(model.name, y_test_arr, y_pred)
            fi = model.feature_importance(feat_cols)
            evaluator.print_model_result(
                model.name,
                y_test_arr,
                y_pred,
                train_time = model.train_time_seconds,
                feature_importance = fi
            )

        winner = evaluator.print_comparison()
        if winner is not None:
            best_path = PATHS.models / "best_model.txt"
            best_path.write_text(winner, encoding = "utf-8")
            self._logger.info("Best model name saved to %s: %s", best_path, winner)

if __name__ == "__main__":
    setup_logging()
    pipeline = TrainingPipeline()
    pipeline.run()
