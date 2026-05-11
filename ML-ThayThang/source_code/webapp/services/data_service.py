from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from config.config_loader import CONFIG
from config.paths import PATHS
from ml_pipeline.data_loader import PostgresLoader, CsvLoader

logger = logging.getLogger(__name__)

class DataService:
    def __init__(self) -> None:
        self._tables: dict[str, pd.DataFrame] | None = None
        self._source: str = "unknown"

    def load(self) -> None:
        try:
            with PostgresLoader(CONFIG.database) as loader:
                self._tables = loader.load_all_tables()
            self._source = "postgresql"
            logger.info("Data loaded from PostgreSQL.")
        except Exception as e:
            logger.warning("PostgreSQL load failed (%s). Falling back to CSV.", e)
            self._tables = CsvLoader(PATHS.data_raw).load_all_tables()
            self._source = "csv"
            logger.info("Data loaded from CSV directory: %s", PATHS.data_raw)

    @property
    def source(self) -> str:
        return self._source

    @property
    def tables(self) -> dict[str, pd.DataFrame]:
        if self._tables is None:
            self.load()
        assert self._tables is not None
        return self._tables

    def table(self, name: str) -> pd.DataFrame:
        return self.tables[name]

    def options(self) -> dict[str, Any]:
        drivers = self.table("drivers").copy()
        trucks = self.table("trucks").copy()
        routes = self.table("routes").copy()
        loads = self.table("loads")

        drivers["full_name"] = drivers["first_name"].fillna("") + " " + drivers["last_name"].fillna("")
        drivers_list = [
            {
                "id": str(r["driver_id"]),
                "label": f"{r['driver_id']} — {r['full_name'].strip()} ({r['years_experience']:.0f} yrs)"
            }
            for _, r in drivers.sort_values("driver_id").iterrows()
        ]

        trucks_list = [
            {
                "id": str(r["truck_id"]),
                "label": f"{r['truck_id']} — {int(r['model_year'])} {r.get('make', '')}".strip()
            }
            for _, r in trucks.sort_values("truck_id").iterrows()
        ]

        routes_list = [
            {
                "id": str(r["route_id"]),
                "label": f"{r['origin_state']} → {r['destination_state']} ({r['typical_distance_miles']:.0f} mi)"
            }
            for _, r in routes.sort_values(["origin_state", "destination_state"]).iterrows()
        ]

        return {
            "drivers": drivers_list,
            "trucks": trucks_list,
            "routes": routes_list,
            "load_types": sorted(loads["load_type"].dropna().unique().tolist()),
            "bookings": sorted(loads["booking_type"].dropna().unique().tolist())
        }

    def dashboard_stats(self) -> dict[str, Any]:
        trips = self.table("trips")
        drivers = self.table("drivers")
        de = self.table("delivery_events").copy()
        routes = self.table("routes")
        loads = self.table("loads")

        de["scheduled_datetime"] = pd.to_datetime(de["scheduled_datetime"])
        de["actual_datetime"] = pd.to_datetime(de["actual_datetime"])
        delivery = de[de["event_type"] == "Delivery"].copy()
        delivery["delay"] = (delivery["actual_datetime"] - delivery["scheduled_datetime"]).dt.total_seconds() / 60.0

        on_time_rate = float((delivery["delay"] <= 30).mean() * 100)
        avg_delay = float(delivery["delay"].mean())
        active_drivers = int((drivers["employment_status"] == "Active").sum())
        total_drivers = int(len(drivers))

        bins = [-180, -120, -60, -30, 0, 15, 30, 60, 120, 180, 360]
        hist_labels = ["<-120", "-120/-60", "-60/-30", "-30/0", "0/15", "15/30", "30/60", "60/120", "120/180", ">180"]
        cuts = pd.cut(delivery["delay"], bins = bins, labels = hist_labels, include_lowest = True)
        hist_counts = cuts.value_counts().reindex(hist_labels, fill_value = 0).tolist()

        delivery["month"] = delivery["scheduled_datetime"].dt.to_period("M").astype(str)
        monthly = (
            delivery.groupby("month")
            .apply(lambda g: float((g["delay"] <= 30).mean() * 100))
            .reset_index(name = "on_time_pct")
            .sort_values("month")
            .tail(12)
        )
        monthly_labels = monthly["month"].tolist()
        monthly_values = [round(v, 2) for v in monthly["on_time_pct"].tolist()]

        delivery_with_route = delivery.merge(loads[["load_id", "route_id"]], on = "load_id", how = "left").merge(
            routes[["route_id", "origin_state", "destination_state"]], on = "route_id", how = "left"
        )
        route_stats = (
            delivery_with_route.groupby(["origin_state", "destination_state"])
            .agg(avg_delay = ("delay", "mean"), on_time_pct = ("delay", lambda s: (s <= 30).mean() * 100), trips = ("delay", "size"))
            .reset_index()
            .sort_values("avg_delay", ascending = False)
            .head(5)
        )
        top_routes = [
            {
                "route": f"{r['origin_state']} → {r['destination_state']}",
                "avg_delay": round(float(r["avg_delay"]), 1),
                "on_time_pct": round(float(r["on_time_pct"]), 1),
                "trips": int(r["trips"])
            }
            for _, r in route_stats.iterrows()
        ]

        delivery_with_driver = delivery.merge(
            self.table("trips")[["trip_id", "driver_id"]], on = "trip_id", how = "left"
        )
        driver_stats = (
            delivery_with_driver.groupby("driver_id")
            .agg(on_time_pct = ("delay", lambda s: (s <= 30).mean() * 100), trips = ("delay", "size"))
            .reset_index()
        )
        driver_stats = driver_stats[driver_stats["trips"] >= 50].sort_values("on_time_pct", ascending = False).head(5)
        driver_stats = driver_stats.merge(drivers[["driver_id", "first_name", "last_name"]], on = "driver_id", how = "left")
        top_drivers = [
            {
                "driver_id": str(r["driver_id"]),
                "name": f"{r['first_name']} {r['last_name']}".strip(),
                "trips": int(r["trips"]),
                "on_time_pct": round(float(r["on_time_pct"]), 1)
            }
            for _, r in driver_stats.iterrows()
        ]

        date_min = delivery["scheduled_datetime"].min()
        date_max = delivery["scheduled_datetime"].max()

        return {
            "kpis": {
                "total_trips": int(len(trips)),
                "on_time_rate": round(on_time_rate, 1),
                "avg_delay": round(avg_delay, 1),
                "active_drivers": active_drivers,
                "total_drivers": total_drivers,
                "total_routes": int(len(routes))
            },
            "delay_histogram": {
                "labels": hist_labels,
                "values": hist_counts
            },
            "on_time_monthly": {
                "labels": monthly_labels,
                "values": monthly_values
            },
            "top_routes": top_routes,
            "top_drivers": top_drivers,
            "date_range": {
                "from": date_min.strftime("%Y-%m-%d") if pd.notna(date_min) else None,
                "to": date_max.strftime("%Y-%m-%d") if pd.notna(date_max) else None
            },
            "source": self._source
        }

DATA = DataService()
