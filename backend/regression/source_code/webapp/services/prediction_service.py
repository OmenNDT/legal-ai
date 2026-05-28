from __future__ import annotations

import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

from config.paths import PATHS
from ml_pipeline.preprocessing import Preprocessor, FEATURE_COLS
from webapp.services.data_service import DATA

logger = logging.getLogger(__name__)

MODEL_NAMES = ["LinearRegression", "RandomForest", "XGBoost"]

class PredictionService:
    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self._preprocessor: Preprocessor | None = None
        self._best_name: str = "XGBoost"
        self._loaded: bool = False
        self._shap_explainer: Any = None

    def load(self) -> None:
        if self._loaded:
            return
        best_path = PATHS.models / "best_model.txt"
        if best_path.exists():
            self._best_name = best_path.read_text(encoding = "utf-8").strip()

        for name in MODEL_NAMES:
            path = PATHS.models / f"{name}.pkl"
            if path.exists():
                self._models[name] = joblib.load(path)
                logger.info("Loaded model %s", name)
            else:
                logger.warning("Missing model file: %s", path)

        prep_path = PATHS.models / "preprocessor.pkl"
        if not prep_path.exists():
            raise FileNotFoundError(f"preprocessor.pkl not found at {prep_path}. Run train.py first.")
        self._preprocessor = joblib.load(prep_path)
        logger.info("Loaded preprocessor.")

        best_model = self._models.get(self._best_name)
        if best_model is not None:
            try:
                self._shap_explainer = shap.TreeExplainer(best_model)
                logger.info("Initialized SHAP TreeExplainer for %s", self._best_name)
            except Exception as e:
                logger.warning("Could not init SHAP explainer (%s). Falling back to global importance.", e)
                self._shap_explainer = None

        self._loaded = True

    @property
    def best_model_name(self) -> str:
        if not self._loaded:
            self.load()
        return self._best_name

    @property
    def best_metrics(self) -> dict[str, float]:
        return self.compare_metrics().get(self._best_name, {})

    def compare_metrics(self) -> dict[str, dict[str, float]]:
        return {
            "LinearRegression": {"R2": 0.8019, "MAE": 17.14, "RMSE": 21.54, "Within30min": 78.3, "TrainTime": 0.3},
            "RandomForest":     {"R2": 0.8149, "MAE": 16.57, "RMSE": 20.82, "Within30min": 80.1, "TrainTime": 2.5},
            "XGBoost":          {"R2": 0.8549, "MAE": 14.73, "RMSE": 18.43, "Within30min": 85.6, "TrainTime": 3.2}
        }

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self._loaded:
            self.load()

        df = self._build_feature_row(payload)
        if self._preprocessor is None:
            raise RuntimeError("Preprocessor not loaded. Call load() first.")
        proc = self._preprocessor.transform(df)
        feat_cols = [c for c in FEATURE_COLS if c in proc.columns]
        X = proc[feat_cols]

        all_preds: dict[str, float] = {}
        for name, model in self._models.items():
            pred = float(model.predict(X)[0])
            all_preds[name] = round(pred, 2)

        best_pred = all_preds.get(self._best_name, 0.0)
        feature_contrib = self._top_contributors(X, feat_cols)

        scheduled_dt = pd.to_datetime(payload["scheduled_datetime"])
        predicted_arrival = scheduled_dt + pd.Timedelta(minutes = best_pred)

        return {
            "best_model": self._best_name,
            "predicted_delay_minutes": best_pred,
            "scheduled_datetime": scheduled_dt.strftime("%Y-%m-%d %H:%M"),
            "predicted_arrival_datetime": predicted_arrival.strftime("%Y-%m-%d %H:%M"),
            "status": self._classify(best_pred),
            "all_models": all_preds,
            "feature_contributions": feature_contrib,
            "metrics": self.compare_metrics().get(self._best_name, {})
        }

    def _classify(self, delay: float) -> str:
        if delay <= -15:
            return "EARLY"
        if delay <= 15:
            return "ON_TIME"
        if delay <= 60:
            return "MODERATE_DELAY"
        return "SEVERE_DELAY"

    def _build_feature_row(self, p: dict[str, Any]) -> pd.DataFrame:
        drivers = DATA.table("drivers")
        trucks = DATA.table("trucks")
        routes = DATA.table("routes")

        driver = drivers[drivers["driver_id"].astype(str) == str(p["driver_id"])].iloc[0]
        truck = trucks[trucks["truck_id"].astype(str) == str(p["truck_id"])].iloc[0]
        route = routes[routes["route_id"].astype(str) == str(p["route_id"])].iloc[0]

        scheduled_dt = pd.to_datetime(p["scheduled_datetime"])
        hire_dt = pd.to_datetime(driver["hire_date"])
        truck_year = int(truck["model_year"])

        row: dict[str, Any] = {}
        row["typical_distance_miles"] = float(route["typical_distance_miles"])
        row["base_rate_per_mile"] = float(route["base_rate_per_mile"])
        row["fuel_surcharge_rate"] = float(route["fuel_surcharge_rate"])
        row["typical_transit_days"] = float(route["typical_transit_days"])
        row["weight_lbs"] = float(p["weight_lbs"])
        row["pieces"] = float(p["pieces"])
        row["revenue"] = float(p["revenue"])
        row["fuel_surcharge"] = float(p["revenue"]) * float(route["fuel_surcharge_rate"])
        row["accessorial_charges"] = 0.0
        row["detention_minutes"] = float(p["detention_minutes"])
        row["years_experience"] = float(driver["years_experience"])
        row["truck_model_year"] = float(truck_year)
        row["tank_capacity_gallons"] = float(truck.get("tank_capacity_gallons", 200))
        row["truck_age_years"] = max(0.0, float(scheduled_dt.year - truck_year))
        row["driver_tenure_days"] = max(0.0, float((scheduled_dt - hire_dt).days))
        row["driver_on_time_rate_prev"] = np.nan
        row["driver_avg_mpg_prev"] = np.nan
        row["driver_avg_idle_prev"] = np.nan
        row["pickup_delay_minutes"] = 0.0

        row["sched_month"] = int(scheduled_dt.month)
        row["sched_quarter"] = int(scheduled_dt.quarter)
        row["sched_day_of_week"] = int(scheduled_dt.dayofweek)
        row["sched_hour"] = int(scheduled_dt.hour)
        row["sched_is_weekend"] = int(scheduled_dt.dayofweek >= 5)
        season_map = {12: 4, 1: 4, 2: 4, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
        row["sched_season"] = int(season_map[scheduled_dt.month])

        row["distance_x_weight"] = (row["typical_distance_miles"] * row["weight_lbs"]) / 1e6
        row["is_long_haul"] = int(row["typical_distance_miles"] > 1500)
        h = row["sched_hour"]
        row["is_rush_hour"] = int((7 <= h <= 9) or (16 <= h <= 19))
        row["is_night"] = int(h >= 22 or h <= 4)
        row["is_winter"] = int(row["sched_month"] in (12, 1, 2))
        row["log_distance"] = float(np.log1p(row["typical_distance_miles"]))
        row["log_weight"] = float(np.log1p(row["weight_lbs"]))
        row["revenue_per_mile"] = row["revenue"] / max(row["typical_distance_miles"], 1)

        row["load_type"] = str(p["load_type"])
        row["booking_type"] = str(p["booking_type"])
        row["origin_state"] = str(route["origin_state"])
        row["destination_state"] = str(route["destination_state"])
        row["employment_status"] = str(driver["employment_status"])
        row["fuel_type"] = str(truck.get("fuel_type", "Diesel"))

        return pd.DataFrame([row])

    def _top_contributors(self, X: pd.DataFrame, feat_cols: list[str]) -> list[dict[str, Any]]:
        best_model = self._models.get(self._best_name)
        if best_model is None:
            return []
        x_vals = X.iloc[0].to_numpy()

        if self._shap_explainer is not None:
            try:
                shap_vals = self._shap_explainer.shap_values(X)
                row = np.asarray(shap_vals)[0]
                scores = list(zip(feat_cols, row, x_vals))
                scores.sort(key = lambda t: abs(t[1]), reverse = True)
                top = scores
                return [
                    {
                        "name": name,
                        "importance": round(float(abs(s)), 4),
                        "shap": round(float(s), 4),
                        "direction": "increase" if s > 0 else ("decrease" if s < 0 else "neutral"),
                        "value": round(float(val), 4)
                    }
                    for name, s, val in top
                ]
            except Exception as e:
                logger.warning("SHAP failed (%s), falling back to global importance.", e)

        if not hasattr(best_model, "feature_importances_"):
            return []
        importances = best_model.feature_importances_
        scores = list(zip(feat_cols, importances, x_vals))
        scores.sort(key = lambda t: t[1], reverse = True)
        top = scores
        return [
            {
                "name": name,
                "importance": round(float(imp), 4),
                "shap": 0.0,
                "direction": "neutral",
                "value": round(float(val), 4)
            }
            for name, imp, val in top
        ]

PREDICTOR = PredictionService()
