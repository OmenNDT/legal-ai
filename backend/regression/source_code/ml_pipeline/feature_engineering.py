from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class FeatureBuilder:
    TARGET_COL = "delay_minutes"
    DATE_COL = "scheduled_datetime"

    def __init__(self, tables: dict[str, pd.DataFrame]) -> None:
        self._tables = tables

    def build(self) -> pd.DataFrame:
        logger.info("=== FeatureBuilder.build() start ===")

        delivery = self._prepare_delivery()
        trips = self._prepare_trips()
        loads = self._prepare_loads()
        routes = self._prepare_routes()
        drivers = self._prepare_drivers()
        trucks = self._prepare_trucks()
        driver_metrics = self._prepare_driver_metrics()

        logger.info(f"Joining delivery -> trips...")
        df = delivery.merge(trips, on = "trip_id", how = "inner")
        logger.info("=> After merge trips: %d rows", len(df))

        logger.info("Joining -> loads...")
        df = df.merge(loads, on = "load_id", how = "left")
        logger.info("=> After merge loads: %d rows", len(df))

        logger.info("Joining -> routes...")
        df = df.merge(routes, on = "route_id", how = "left")
        logger.info("=> After merge routes: %d rows", len(df))

        logger.info("Joining -> drivers...")
        df = df.merge(drivers, on = "driver_id", how = "left")
        logger.info("=> After merge drivers: %d rows", len(df))

        logger.info("Joining -> trucks...")
        df = df.merge(trucks, on = "truck_id", how = "left")
        logger.info("=> After merge trucks: %d rows", len(df))

        logger.info("Joining -> driver_monthly_metrics (Lag 1 month)...")
        df = self._merge_driver_metrics(df, driver_metrics)
        logger.info("=> After merge driver_metrics: %d rows", len(df))

        logger.info("Adding temporal features...")
        df = self._add_temporal_features(df)

        logger.info("Adding pickup delay feature...")
        df = self._add_pickup_delay(df)

        logger.info("Adding truck age feature...")
        df["truck_age_years"] = (pd.to_datetime(df["scheduled_datetime"]).dt.year - df["truck_model_year"]).clip(lower=0)

        logger.info("Adding driver tenure feature...")
        df["driver_tenure_days"] = (pd.to_datetime(df["scheduled_datetime"]) - pd.to_datetime(df["hire_date"])).dt.days.clip(lower=0)

        logger.info("Adding interaction & non-linear features...")
        df["distance_x_weight"] = (df["typical_distance_miles"].fillna(0) * df["weight_lbs"].fillna(0)) / 1e6
        df["is_long_haul"] = (df["typical_distance_miles"] > 1500).astype(int)
        df["is_rush_hour"] = (((df["sched_hour"] >= 7) & (df["sched_hour"] <= 9)) | ((df["sched_hour"] >= 16) & (df["sched_hour"] <= 19))).astype(int)
        df["is_night"] = ((df["sched_hour"] >= 22) | (df["sched_hour"] <= 4)).astype(int)
        df["is_winter"] = df["sched_month"].isin([12, 1, 2]).astype(int)
        df["log_distance"] = np.log1p(df["typical_distance_miles"].fillna(0))
        df["log_weight"] = np.log1p(df["weight_lbs"].fillna(0))
        df["revenue_per_mile"] = df["revenue"].fillna(0) / df["typical_distance_miles"].replace(0, np.nan).fillna(1)

        logger.info("Final dataset: %d rows, %d cols", df.shape[0], df.shape[1])
        logger.info("=== FeatureBuilder.build() done ===")
        return df

    def _prepare_delivery(self) -> pd.DataFrame:
        logger.info("Preparing delivery_events (Delivery only)...")
        de = self._tables["delivery_events"].copy()
        de = de[de["event_type"] == "Delivery"].copy()
        de["scheduled_datetime"] = pd.to_datetime(de["scheduled_datetime"])
        de["actual_datetime"] = pd.to_datetime(de["actual_datetime"])
        de[self.TARGET_COL] = (de["actual_datetime"] - de["scheduled_datetime"]).dt.total_seconds() / 60.0
        logger.info("Delivery events: %d rows | delay_minutes mean = %.1f, min = %.1f, max = %.1f", len(de), de[self.TARGET_COL].mean(), de[self.TARGET_COL].min(), de[self.TARGET_COL].max())
        return de[["trip_id", "load_id", "facility_id", "scheduled_datetime", "detention_minutes", self.TARGET_COL]].copy()

    def _prepare_trips(self) -> pd.DataFrame:
        logger.info("Preparing trips (safe features only)...")
        trips = self._tables["trips"].copy()
        keep = ["trip_id", "driver_id", "truck_id", "trailer_id", "dispatch_date", "trip_status"]
        return trips[keep].copy()

    def _prepare_loads(self) -> pd.DataFrame:
        logger.info("Preparing loads...")
        loads = self._tables["loads"].copy()
        keep = ["load_id", "customer_id", "route_id", "load_date", "load_type", "weight_lbs", "pieces", "revenue", "fuel_surcharge", "accessorial_charges", "booking_type"]
        return loads[keep].copy()

    def _prepare_routes(self) -> pd.DataFrame:
        logger.info("Preparing routes...")
        routes = self._tables["routes"].copy()
        keep = ["route_id", "origin_state", "destination_state", "typical_distance_miles", "base_rate_per_mile", "fuel_surcharge_rate", "typical_transit_days"]
        return routes[keep].copy()

    def _prepare_drivers(self) -> pd.DataFrame:
        logger.info("Preparing drivers...")
        drivers = self._tables["drivers"].copy()
        keep = ["driver_id", "hire_date", "years_experience", "employment_status", "home_terminal"]
        return drivers[keep].copy()

    def _prepare_trucks(self) -> pd.DataFrame:
        logger.info("Preparing trucks...")
        trucks = self._tables["trucks"].copy()
        trucks = trucks.rename(columns = {"model_year": "truck_model_year", "status": "truck_status"})
        keep = ["truck_id", "truck_model_year", "fuel_type", "tank_capacity_gallons", "truck_status"]
        return trucks[keep].copy()

    def _prepare_driver_metrics(self) -> pd.DataFrame:
        logger.info("Preparing driver_monthly_metrics...")
        dm = self._tables["driver_monthly_metrics"].copy()
        dm["month"] = pd.to_datetime(dm["month"])
        return dm[["driver_id", "month", "on_time_delivery_rate", "average_mpg", "average_idle_hours"]].copy()

    def _merge_driver_metrics(self, df: pd.DataFrame, driver_metrics: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        dm = driver_metrics.copy().rename(columns = {
            "on_time_delivery_rate": "driver_on_time_rate_prev",
            "average_mpg": "driver_avg_mpg_prev",
            "average_idle_hours": "driver_avg_idle_prev"
        })
        dm["month"] = pd.to_datetime(dm["month"]) + pd.offsets.MonthBegin(1)
        dm = dm.sort_values("month")
        df["_dispatch_dt"] = pd.to_datetime(df["dispatch_date"])
        df = df.sort_values("_dispatch_dt")
        df = pd.merge_asof(df, dm, left_on = "_dispatch_dt", right_on = "month", by = "driver_id", direction = "backward")
        df = df.drop(columns = ["_dispatch_dt", "month"], errors = "ignore")
        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        dt = pd.to_datetime(df["scheduled_datetime"])
        df["sched_month"] = dt.dt.month
        df["sched_quarter"] = dt.dt.quarter
        df["sched_day_of_week"] = dt.dt.dayofweek
        df["sched_hour"] = dt.dt.hour
        df["sched_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        season_map = {
            12: 4, 1: 4, 2: 4,
            3: 1, 4: 1, 5: 1,
            6: 2, 7: 2, 8: 2,
            9: 3, 10: 3, 11: 3,
        }
        df["sched_season"] = dt.dt.month.map(season_map)
        return df

    def _add_pickup_delay(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing pickup delay per trip...")
        de = self._tables["delivery_events"].copy()
        pickup = de[de["event_type"] == "Pickup"].copy()
        pickup["scheduled_datetime"] = pd.to_datetime(pickup["scheduled_datetime"])
        pickup["actual_datetime"] = pd.to_datetime(pickup["actual_datetime"])
        pickup["pickup_delay_minutes"] = (pickup["actual_datetime"] - pickup["scheduled_datetime"]).dt.total_seconds() / 60.0
        pickup_agg = (pickup.groupby("trip_id")["pickup_delay_minutes"].mean().reset_index())
        return df.merge(pickup_agg, on = "trip_id", how = "left")
