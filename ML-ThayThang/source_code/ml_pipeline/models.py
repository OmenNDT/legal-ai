from __future__ import annotations

import logging
import time
from typing import Any
import joblib
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    def __init__(self, name: str) -> None:
        self.name = name
        self._model: Any = None
        self.train_time_seconds: float = 0.0

    @abstractmethod
    def _build(self) -> Any:
        ...

    @abstractmethod
    def feature_importance(self, feature_names: list[str]) -> pd.Series:
        ...

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        logger.info("[%s] Building model...", self.name)
        self._model = self._build()
        logger.info("[%s] Training on %d samples, %d features...", self.name, len(X), X.shape[1])
        t0 = time.time()
        self._model.fit(X, y)
        self.train_time_seconds = time.time() - t0
        logger.info("[%s] Training done in %.2fs", self.name, self.train_time_seconds)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError(f"Model '{self.name}' not trained yet.")
        return self._model.predict(X)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path)
        logger.info("[%s] Model saved to %s", self.name, path)

    def load(self, path: str | Path) -> None:
        self._model = joblib.load(path)
        logger.info("[%s] Model loaded from %s", self.name, path)

class LinearRegressionModel(BaseModel):
    def __init__(self) -> None:
        super().__init__("LinearRegression")

    def _build(self) -> LinearRegression:
        return LinearRegression(n_jobs = -1)

    def feature_importance(self, feature_names: list[str]) -> pd.Series:
        importances = np.abs(self._model.coef_)
        return pd.Series(importances, index = feature_names).sort_values(ascending = False)

class RandomForestModel(BaseModel):
    def __init__(self, **kwargs) -> None:
        super().__init__("RandomForest")
        self._kwargs = kwargs

    def _build(self) -> RandomForestRegressor:
        defaults: dict[str, Any] = dict(
            n_estimators = 200,
            max_depth = 12,
            min_samples_split = 10,
            min_samples_leaf = 5,
            max_features = "sqrt",
            n_jobs = -1,
            random_state = 42
        )
        defaults.update(self._kwargs)
        logger.info("[%s] Params: %s", self.name, defaults)
        return RandomForestRegressor(**defaults)

    def feature_importance(self, feature_names: list[str]) -> pd.Series:
        importances = self._model.feature_importances_
        return pd.Series(importances, index = feature_names).sort_values(ascending = False)

class XGBoostModel(BaseModel):
    def __init__(self, **kwargs) -> None:
        super().__init__("XGBoost")
        self._kwargs = kwargs

    def _build(self) -> xgb.XGBRegressor:
        defaults: dict[str, Any] = dict(
            n_estimators = 1500,
            max_depth = 8,
            learning_rate = 0.03,
            subsample = 0.85,
            colsample_bytree = 0.85,
            min_child_weight = 3,
            gamma = 0.1,
            reg_alpha = 0.05,
            reg_lambda = 1.5,
            objective = "reg:squarederror",
            tree_method = "hist",
            early_stopping_rounds = 50,
            n_jobs = -1,
            random_state = 42,
            verbosity = 0
        )
        defaults.update(self._kwargs)
        logger.info("[%s] Params: %s", self.name, defaults)
        return xgb.XGBRegressor(**defaults)

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame | None = None, y_val: pd.Series | None = None) -> None:
        self._model = self._build()
        logger.info("[%s] Training on %d samples, %d features...", self.name, len(X), X.shape[1])
        t0 = time.time()
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        self._model.fit(X, y, eval_set = eval_set, verbose = False)
        self.train_time_seconds = time.time() - t0
        logger.info("[%s] Training done in %.2fs", self.name, self.train_time_seconds)

    def feature_importance(self, feature_names: list[str]) -> pd.Series:
        importances = self._model.feature_importances_
        return pd.Series(importances, index = feature_names).sort_values(ascending = False)

class ModelRegistry:
    def __init__(self) -> None:
        self._models: list[BaseModel] = []

    def register(self, model: BaseModel) -> None:
        self._models.append(model)
        logger.info("Registered model: %s", model.name)

    def all_models(self) -> list[BaseModel]:
        return self._models

    def get(self, name: str) -> BaseModel | None:
        for m in self._models:
            if m.name == name:
                return m
        return None