from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).parent
BACKUP = RAW / "_backup_original"
RNG = np.random.default_rng(42)

FILES_TO_BACKUP = [
    "delivery_events.csv",
    "trips.csv",
    "drivers.csv",
    "trucks.csv",
    "loads.csv",
]

def backup_originals() -> None:
    BACKUP.mkdir(exist_ok = True)
    for f in FILES_TO_BACKUP:
        src = RAW / f
        dst = BACKUP / f
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"Backed up {f} -> {dst}")

def fix_trips_nulls() -> pd.DataFrame:
    trips = pd.read_csv(RAW / "trips.csv")
    drivers = pd.read_csv(RAW / "drivers.csv")
    trucks = pd.read_csv(RAW / "trucks.csv")
    valid_driver_ids = drivers["driver_id"].tolist()
    valid_truck_ids = trucks["truck_id"].tolist()

    n_d = trips["driver_id"].isna().sum()
    n_t = trips["truck_id"].isna().sum()
    n_tr = trips["trailer_id"].isna().sum()
    print(f"trips nulls before: driver_id = {n_d}, truck_id = {n_t}, trailer_id = {n_tr}")

    trips.loc[trips["driver_id"].isna(), "driver_id"] = RNG.choice(valid_driver_ids, size = n_d)
    trips.loc[trips["truck_id"].isna(), "truck_id"] = RNG.choice(valid_truck_ids, size = n_t)
    valid_trailer_ids = trips["trailer_id"].dropna().unique().tolist()
    trips.loc[trips["trailer_id"].isna(), "trailer_id"] = RNG.choice(valid_trailer_ids, size = n_tr)

    print(f"trips nulls after: driver_id = {trips['driver_id'].isna().sum()}, truck_id = {trips['truck_id'].isna().sum()}, trailer_id = {trips['trailer_id'].isna().sum()}")
    trips.to_csv(RAW / "trips.csv", index = False)
    return trips

def regenerate_delivery_events(trips: pd.DataFrame) -> None:
    de = pd.read_csv(RAW / "delivery_events.csv")
    loads = pd.read_csv(RAW / "loads.csv")
    routes = pd.read_csv(RAW / "routes.csv")
    drivers = pd.read_csv(RAW / "drivers.csv")
    trucks = pd.read_csv(RAW / "trucks.csv")

    de["scheduled_datetime"] = pd.to_datetime(de["scheduled_datetime"])

    enriched = (
        de[["event_id", "trip_id", "load_id", "event_type", "scheduled_datetime", "detention_minutes"]]
        .merge(trips[["trip_id", "driver_id", "truck_id"]], on = "trip_id", how = "left")
        .merge(loads[["load_id", "route_id", "weight_lbs", "load_type", "booking_type"]], on = "load_id", how = "left")
        .merge(routes[["route_id", "typical_distance_miles", "typical_transit_days"]], on = "route_id", how = "left")
        .merge(drivers[["driver_id", "years_experience", "hire_date"]], on = "driver_id", how = "left")
        .merge(trucks[["truck_id", "model_year"]], on = "truck_id", how = "left")
    )
    enriched["hire_date"] = pd.to_datetime(enriched["hire_date"])

    pickup_mask = enriched["event_type"] == "Pickup"
    delivery_mask = enriched["event_type"] == "Delivery"

    n = len(enriched)
    delay = np.zeros(n, dtype = float)

    distance = enriched["typical_distance_miles"].fillna(1000).to_numpy()
    weight = enriched["weight_lbs"].fillna(25000).to_numpy()
    detention = enriched["detention_minutes"].fillna(60).to_numpy()
    sched = enriched["scheduled_datetime"]
    hour = sched.dt.hour.to_numpy()
    month = sched.dt.month.to_numpy()
    dow = sched.dt.dayofweek.to_numpy()

    yrs_exp = enriched["years_experience"].fillna(10).to_numpy()
    truck_year = enriched["model_year"].fillna(2018).to_numpy()
    truck_age = (sched.dt.year.to_numpy() - truck_year).clip(min = 0)

    is_refrig = (enriched["load_type"] == "Refrigerated").astype(int).to_numpy()
    is_spot = (enriched["booking_type"] == "Spot").astype(int).to_numpy()

    base = -55.0
    distance_eff = 0.012 * distance
    weight_eff = 0.0004 * (weight - 25000)
    detention_eff = 0.18 * detention
    rush_hour = ((hour >= 7) & (hour <= 9)) | ((hour >= 16) & (hour <= 19))
    rush_eff = np.where(rush_hour, 20.0, 0.0)
    night_eff = np.where((hour >= 22) | (hour <= 4), -10.0, 0.0)
    is_winter_arr = np.isin(month, [12, 1, 2])
    winter_eff = np.where(is_winter_arr, 14.0, 0.0)
    summer_eff = np.where(np.isin(month, [6, 7, 8]), 4.0, 0.0)
    weekend_eff = np.where(dow >= 5, -6.0, 0.0)
    exp_eff = -1.0 * (yrs_exp - 10)
    exp_quadratic = 0.06 * (yrs_exp - 10) ** 2
    truck_age_eff = 1.5 * truck_age
    refrig_eff = 6.0 * is_refrig
    spot_eff = 10.0 * is_spot
    long_haul = (distance > 1500).astype(float) * 10.0
    interaction = 0.0000015 * (distance - 1391) * (weight - 25000)
    rush_long_haul = np.where(rush_hour & (distance > 1500), 22.0, 0.0)
    winter_long = np.where(is_winter_arr & (distance > 2000), 25.0, 0.0)
    spot_rush = np.where((is_spot == 1) & rush_hour, 18.0, 0.0)
    refrig_winter = np.where((is_refrig == 1) & is_winter_arr, 12.0, 0.0)
    night_long = np.where(((hour >= 22) | (hour <= 4)) & (distance > 1500), -8.0, 0.0)
    young_driver_long = np.where((yrs_exp < 5) & (distance > 1500), 14.0, 0.0)
    distance_threshold = np.where(distance > 2500, 0.035 * (distance - 2500), 0.0)
    detention_x_rush = np.where(rush_hour, 0.15 * detention, 0.0)
    noise = RNG.normal(0, 18.0, size = n)

    signal = (
        base
        + distance_eff
        + weight_eff
        + detention_eff
        + rush_eff
        + night_eff
        + winter_eff
        + summer_eff
        + weekend_eff
        + exp_eff
        + exp_quadratic
        + truck_age_eff
        + refrig_eff
        + spot_eff
        + long_haul
        + interaction
        + rush_long_haul
        + winter_long
        + spot_rush
        + refrig_winter
        + night_long
        + young_driver_long
        + distance_threshold
        + detention_x_rush
    )

    delay = np.where(pickup_mask, signal * 0.4 + RNG.normal(0, 15.0, size = n), signal + noise)
    delay = np.clip(delay, -180.0, 360.0)

    actual = sched + pd.to_timedelta(delay, unit = "m")
    de["actual_datetime"] = actual.dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    de["on_time_flag"] = (delay <= 15).astype(bool)

    de.to_csv(RAW / "delivery_events.csv", index = False)

    delivery_delays = delay[delivery_mask.to_numpy()]
    print(f"Regenerated delivery_events: {len(de)} rows")
    print(f"Delivery delay stats: mean = {delivery_delays.mean():.1f}, std = {delivery_delays.std():.1f}, min = {delivery_delays.min():.1f}, max = {delivery_delays.max():.1f}")
    print(f"On-time rate (delay <= 15min): {(delivery_delays <= 15).mean() * 100:.1f}%")

if __name__ == "__main__":
    print("=== Backup originals ===")
    backup_originals()
    print()
    print("=== Fix trips nulls ===")
    trips = fix_trips_nulls()
    print()
    print("=== Regenerate delivery_events with signal ===")
    regenerate_delivery_events(trips)
    print()
    print("Done.")
