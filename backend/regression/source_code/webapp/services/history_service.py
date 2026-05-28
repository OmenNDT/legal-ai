from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from typing import Any

from config.paths import PATHS

logger = logging.getLogger(__name__)

HISTORY_FILE = PATHS.source_code / "webapp" / "history.json"

class HistoryService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if HISTORY_FILE.exists():
            try:
                self._items = json.loads(HISTORY_FILE.read_text(encoding = "utf-8"))
            except Exception as e:
                logger.warning("Failed to load history file (%s). Starting empty.", e)
                self._items = []

    def _save(self) -> None:
        HISTORY_FILE.parent.mkdir(parents = True, exist_ok = True)
        HISTORY_FILE.write_text(json.dumps(self._items, indent = 2, ensure_ascii = False), encoding = "utf-8")

    def add(self, payload: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            next_id = (max([it["id"] for it in self._items], default = 0) + 1)
            item = {
                "id": next_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "driver_id": payload.get("driver_id"),
                "truck_id": payload.get("truck_id"),
                "route_id": payload.get("route_id"),
                "route_label": payload.get("route_label", ""),
                "load_type": payload.get("load_type"),
                "booking_type": payload.get("booking_type"),
                "weight_lbs": payload.get("weight_lbs"),
                "scheduled_datetime": payload.get("scheduled_datetime"),
                "model": result["best_model"],
                "predicted_delay_minutes": result["predicted_delay_minutes"],
                "status": result["status"],
                "actual_delay_minutes": None,
                "error_minutes": None
            }
            self._items.insert(0, item)
            self._items = self._items[:500]
            self._save()
        return item

    def list(self) -> list[dict[str, Any]]:
        return list(self._items)

    def stats(self) -> dict[str, Any]:
        items = self._items
        total = len(items)
        with_actual = [it for it in items if it.get("actual_delay_minutes") is not None]
        pending = total - len(with_actual)
        if with_actual:
            errors = [abs(it["error_minutes"]) for it in with_actual]
            avg_error = sum(errors) / len(errors)
            within_30 = sum(1 for e in errors if e <= 30) / len(errors) * 100
        else:
            avg_error = 0.0
            within_30 = 0.0
        return {
            "total": total,
            "pending": pending,
            "avg_error": round(avg_error, 2),
            "within_30_pct": round(within_30, 1)
        }

    def update_actual(self, item_id: int, actual: float) -> dict[str, Any] | None:
        with self._lock:
            for it in self._items:
                if it["id"] == item_id:
                    it["actual_delay_minutes"] = round(float(actual), 2)
                    it["error_minutes"] = round(it["predicted_delay_minutes"] - actual, 2)
                    self._save()
                    return it
        return None

HISTORY = HistoryService()
