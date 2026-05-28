from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent))

from config.config_loader import CONFIG
from config.paths import PATHS
from ml_pipeline.data_loader import PostgresLoader, CsvLoader
from ml_pipeline.feature_engineering import FeatureBuilder
from ml_pipeline.preprocessing import Preprocessor, DataSplitter, FEATURE_COLS, TARGET_COL
from ml_pipeline.models import LinearRegressionModel, RandomForestModel, XGBoostModel, BaseModel

logger = logging.getLogger(__name__)

def setup_logging() -> None:
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
        handlers = [logging.StreamHandler(sys.stdout)]
    )

class Validator:
    def __init__(self, n_splits: int = 5) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self._n_splits = n_splits

    def _load_master(self) -> pd.DataFrame:
        try:
            with PostgresLoader(CONFIG.database) as loader:
                tables = loader.load_all_tables()
        except Exception as e:
            self._logger.warning("PostgreSQL load failed (%s). Falling back to CSV.", e)
            tables = CsvLoader(PATHS.data_raw).load_all_tables()
        return FeatureBuilder(tables).build()

    def _new_model(self, name: str) -> BaseModel:
        if name == "LinearRegression":
            return LinearRegressionModel()
        if name == "RandomForest":
            return RandomForestModel()
        if name == "XGBoost":
            return XGBoostModel()
        raise ValueError(f"Unknown model: {name}")

    def kfold_cv(self) -> pd.DataFrame:
        self._logger.info("=" * 70)
        self._logger.info("K-FOLD CROSS-VALIDATION (k = %d)", self._n_splits)
        self._logger.info("=" * 70)

        df = self._load_master()
        results = []
        kf = KFold(n_splits = self._n_splits, shuffle = True, random_state = 42)

        for model_name in ["LinearRegression", "RandomForest", "XGBoost"]:
            self._logger.info("\n>>> %s", model_name)
            fold_metrics: list[dict] = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(df), start = 1):
                train_raw = df.iloc[train_idx].copy()
                val_raw = df.iloc[val_idx].copy()

                pre = Preprocessor()
                train_proc = pre.fit_transform(train_raw)
                val_proc = pre.transform(val_raw)

                splitter = DataSplitter()
                X_tr, y_tr = splitter.get_xy(train_proc)
                X_val, y_val = splitter.get_xy(val_proc)

                model = self._new_model(model_name)
                if model_name == "XGBoost":
                    model.fit(X_tr, y_tr, X_val = X_val, y_val = y_val)  # type: ignore[call-arg]
                else:
                    model.fit(X_tr, y_tr)

                y_pred = model.predict(X_val)
                mae = mean_absolute_error(y_val, y_pred)
                rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
                r2 = r2_score(y_val, y_pred)
                fold_metrics.append({"fold": fold, "MAE": mae, "RMSE": rmse, "R2": r2})
                self._logger.info("Fold %d: MAE = %.2f, RMSE = %.2f, R2 = %.4f", fold, mae, rmse, r2)

            fold_df = pd.DataFrame(fold_metrics)
            mean_row = {"model": model_name, "MAE_mean": fold_df["MAE"].mean(), "MAE_std": fold_df["MAE"].std(), "RMSE_mean": fold_df["RMSE"].mean(), "RMSE_std": fold_df["RMSE"].std(), "R2_mean": fold_df["R2"].mean(), "R2_std": fold_df["R2"].std()}
            results.append(mean_row)

        return pd.DataFrame(results)

    def _load_best_model_name(self) -> str:
        path = PATHS.models / "best_model.txt"
        if not path.exists():
            raise FileNotFoundError(f"Best model marker not found: {path}. Run train.py first.")
        return path.read_text(encoding = "utf-8").strip()

    def error_analysis(self, model_name: str | None = None) -> None:
        import joblib
        if model_name is None:
            model_name = self._load_best_model_name()
        self._logger.info("=" * 70)
        self._logger.info("ERROR ANALYSIS - %s (best model)", model_name)
        self._logger.info("=" * 70)

        df_master = self._load_master()
        splitter = DataSplitter()
        _, _, test_raw = splitter.split(df_master)

        preprocessor: Preprocessor = joblib.load(PATHS.models / "preprocessor.pkl")
        model = joblib.load(PATHS.models / f"{model_name}.pkl")

        proc = preprocessor.transform(test_raw)
        feat_cols = [c for c in FEATURE_COLS if c in proc.columns]
        X_test = proc[feat_cols]
        y_true = proc[TARGET_COL].values

        y_pred = model.predict(X_test)
        residuals = y_true - y_pred

        meta = test_raw.reset_index(drop = True).copy()
        meta["y_true"] = y_true
        meta["y_pred"] = y_pred
        meta["abs_error"] = np.abs(residuals)

        print("\n" + "=" * 70)
        print(f"ERROR BY origin_state (top 10 worst MAE)")
        print("=" * 70)
        by_state = meta.groupby("origin_state").agg(n = ("abs_error", "size"), mae = ("abs_error", "mean")).sort_values("mae", ascending = False).head(10)
        print(by_state.to_string())

        print("\n" + "=" * 70)
        print("ERROR BY load_type")
        print("=" * 70)
        by_load = meta.groupby("load_type").agg(n = ("abs_error", "size"), mae = ("abs_error", "mean"))
        print(by_load.to_string())

        print("\n" + "=" * 70)
        print("ERROR BY booking_type")
        print("=" * 70)
        by_book = meta.groupby("booking_type").agg(n = ("abs_error", "size"), mae = ("abs_error", "mean"))
        print(by_book.to_string())

        print("\n" + "=" * 70)
        print("ERROR BY scheduled month")
        print("=" * 70)
        meta["sched_month"] = pd.to_datetime(meta["scheduled_datetime"]).dt.month
        by_month = meta.groupby("sched_month").agg(n = ("abs_error", "size"), mae = ("abs_error", "mean"))
        print(by_month.to_string())

        print("\n" + "=" * 70)
        print(f"WORST 10 PREDICTIONS")
        print("=" * 70)
        worst = meta.nlargest(10, "abs_error")[["trip_id", "scheduled_datetime", "origin_state", "destination_state", "load_type", "y_true", "y_pred", "abs_error"]]
        print(worst.to_string(index = False))


def main() -> None:
    setup_logging()
    validator = Validator(n_splits = 5)

    cv_df = validator.kfold_cv()
    print("\n" + "=" * 70)
    print("K-FOLD CV SUMMARY")
    print("=" * 70)
    print(cv_df.to_string(index = False))

    print()
    validator.error_analysis()


if __name__ == "__main__":
    main()
