from __future__ import annotations

import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "typical_distance_miles",
    "base_rate_per_mile",
    "fuel_surcharge_rate",
    "typical_transit_days",
    "weight_lbs",
    "pieces",
    "revenue",
    "fuel_surcharge",
    "accessorial_charges",
    "detention_minutes",
    "years_experience",
    "truck_model_year",
    "tank_capacity_gallons",
    "truck_age_years",
    "driver_tenure_days",
    "driver_on_time_rate_prev",
    "driver_avg_mpg_prev",
    "driver_avg_idle_prev",
    "pickup_delay_minutes",
    "sched_month",
    "sched_quarter",
    "sched_day_of_week",
    "sched_hour",
    "sched_is_weekend",
    "sched_season",
    "distance_x_weight",
    "is_long_haul",
    "is_rush_hour",
    "is_night",
    "is_winter",
    "log_distance",
    "log_weight",
    "revenue_per_mile",
    "load_type",
    "booking_type",
    "origin_state",
    "destination_state",
    "employment_status",
    "fuel_type"
]

CATEGORICAL_COLS = [
    "load_type",
    "booking_type",
    "origin_state",
    "destination_state",
    "employment_status",
    "fuel_type"
]

NUMERICAL_COLS = [c for c in FEATURE_COLS if c not in CATEGORICAL_COLS]

TARGET_COL = "delay_minutes"
DATE_COL = "scheduled_datetime"

class Preprocessor:
    def __init__(self) -> None:
        self._label_encoders: dict[str, LabelEncoder] = {}
        self._medians: dict[str, float] = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preprocessor.fit_transform: %d rows", len(df))
        df = self._select_and_copy(df)
        df = self._handle_missing(df, fit=True)
        df = self._encode_categoricals(df, fit=True)
        logger.info("fit_transform done. Shape: %s", df.shape)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preprocessor.transform: %d rows", len(df))
        df = self._select_and_copy(df)
        df = self._handle_missing(df, fit=False)
        df = self._encode_categoricals(df, fit=False)
        logger.info("transform done. Shape: %s", df.shape)
        return df

    def _select_and_copy(self, df: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in FEATURE_COLS + [TARGET_COL, DATE_COL] if c in df.columns]
        missing_cols = set(FEATURE_COLS) - set(df.columns)
        if missing_cols:
            logger.warning("Missing feature columns (will skip): %s", missing_cols)
        return df[available].copy()

    def _handle_missing(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        num_cols = [c for c in NUMERICAL_COLS if c in df.columns]
        for col in num_cols:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                if fit:
                    self._medians[col] = float(df[col].median())
                fill_val = self._medians.get(col, 0.0)
                df[col] = df[col].fillna(fill_val)
                logger.debug("Filled %d NaN in '%s' with median = %.4f", n_missing, col, fill_val)

        cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
        for col in cat_cols:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                df[col] = df[col].fillna("Unknown")
                logger.debug("Filled %d NaN in '%s' with 'Unknown'", n_missing, col)

        return df

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
        for col in cat_cols:
            if fit:
                le = LabelEncoder()
                df[col] = pd.Series(le.fit_transform(df[col].astype(str)), index = df.index)
                self._label_encoders[col] = le
                logger.debug("LabelEncoded '%s': %d classes", col, len(le.classes_))
            else:
                le = self._label_encoders.get(col)
                if le is None:
                    logger.warning("No encoder found for '%s', skipping.", col)
                    continue
                known = set(le.classes_)
                fallback = le.classes_[0]
                df[col] = df[col].astype(str).apply(lambda x: x if x in known else fallback)
                df[col] = pd.Series(le.transform(df[col]), index = df.index)
        return df

class DataSplitter:
    def __init__(self, train_ratio: float = 0.70, val_ratio: float = 0.15) -> None:
        self._train_ratio = train_ratio
        self._val_ratio = val_ratio

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = df.sort_values(DATE_COL).reset_index(drop = True)
        n = len(df)
        train_end = int(n * self._train_ratio)
        val_end = int(n * (self._train_ratio + self._val_ratio))

        train = df.iloc[:train_end].copy()
        val = df.iloc[train_end:val_end].copy()
        test = df.iloc[val_end:].copy()

        train_min = pd.to_datetime(train[DATE_COL]).min()
        train_max = pd.to_datetime(train[DATE_COL]).max()
        val_min = pd.to_datetime(val[DATE_COL]).min()
        val_max = pd.to_datetime(val[DATE_COL]).max()
        test_min = pd.to_datetime(test[DATE_COL]).min()
        test_max = pd.to_datetime(test[DATE_COL]).max()

        logger.info(
            "Split (time-based): train = %d [%s -> %s] | val=%d [%s -> %s] | test = %d [%s -> %s]",
            len(train), train_min.date(), train_max.date(),
            len(val), val_min.date(), val_max.date(),
            len(test), test_min.date(), test_max.date()
        )
        return train, val, test

    def get_xy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        feat_cols = [c for c in FEATURE_COLS if c in df.columns]
        X = df[feat_cols]
        y = df[TARGET_COL]
        return X, y
