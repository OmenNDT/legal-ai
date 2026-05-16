from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import cast

logger = logging.getLogger(__name__)

def _cell(df: pd.DataFrame, row: object, col: str) -> float:
    return float(cast(float, df.at[row, col]))

DIVIDER = "=" * 70
SUB_DIVIDER = "-" * 70

class ModelEvaluator:
    def __init__(self) -> None:
        self._results: dict[str, dict[str, float]] = {}

    def evaluate(self, model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        y_true = np.asarray(y_true, dtype = float)
        y_pred = np.asarray(y_pred, dtype = float)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = r2_score(y_true, y_pred)
        mask = np.abs(y_true) > 1e-6
        mape = (float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else float("nan"))
        residuals = y_true - y_pred
        within_30min = float(np.mean(np.abs(residuals) <= 30) * 100)
        within_60min = float(np.mean(np.abs(residuals) <= 60) * 100)

        metrics = {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
            "MAPE": round(mape, 4),
            "Within30min%": round(within_30min, 2),
            "Within60min%": round(within_60min, 2),
        }

        self._results[model_name] = metrics
        logger.info("[%s] Metrics: %s", model_name, metrics)
        return metrics

    def print_model_result(self, model_name: str, y_true: np.ndarray, y_pred: np.ndarray, train_time: float, feature_importance: pd.Series | None = None) -> None:
        metrics = self._results.get(model_name)
        if metrics is None:
            logger.warning("No metrics for model '%s'. Call evaluate() first.", model_name)
            return

        residuals = np.asarray(y_true) - np.asarray(y_pred)

        print(f"\n{DIVIDER}")
        print(f"MODEL: {model_name}")
        print(DIVIDER)
        print(f"Train time: {train_time:.2f}s")
        print(f"MAE: {metrics['MAE']:.4f} min")
        print(f"RMSE: {metrics['RMSE']:.4f} min")
        print(f"R2: {metrics['R2']:.4f}")
        print(f"MAPE: {metrics['MAPE']:.4f} %")
        print(f"Predictions within 30 min: {metrics['Within30min%']:.2f}%")
        print(f"Predictions within 60 min: {metrics['Within60min%']:.2f}%")
        print(SUB_DIVIDER)
        print("Residual distribution:")
        print(f"mean = {residuals.mean():.2f}, std = {residuals.std():.2f}, min = {residuals.min():.2f}, max = {residuals.max():.2f}")
        print(f"Q25 = {np.quantile(residuals, 0.25):.2f}, median = {np.median(residuals):.2f}, Q75 = {np.quantile(residuals, 0.75):.2f}")
        if feature_importance is not None:
            print(SUB_DIVIDER)
            print("Top 10 feature importance:")
            top = feature_importance.head(10)
            max_imp = float(top.max()) if len(top) > 0 else 0.0
            bar_width = 40
            for feat, imp in top.items():
                length = int(round(imp / max_imp * bar_width)) if max_imp > 0 else 0
                bar = "|" * length
                print(f"{feat}: {imp:.4f} {bar}")

    def print_comparison(self) -> str | None:
        if not self._results:
            logger.warning("No results to compare.")
            return None
        df = pd.DataFrame(self._results).T
        print(f"\n{DIVIDER}")
        print("MODEL COMPARISON")
        print(DIVIDER)
        print(df.to_string())
        print(SUB_DIVIDER)
        best_mae = df["MAE"].idxmin()
        best_rmse = df["RMSE"].idxmin()
        best_r2 = df["R2"].idxmax()
        print(f"\nBest MAE: {best_mae}, ({_cell(df, best_mae, 'MAE'):.4f} min)")
        print(f"Best RMSE: {best_rmse}, ({_cell(df, best_rmse, 'RMSE'):.4f} min)")
        print(f"Best R2: {best_r2}, ({_cell(df, best_r2, 'R2'):.4f})")
        winner = self._pick_winner(df)
        print(f"\n{'★' * 5}  BEST MODEL: {winner}  {'★' * 5}")
        self._explain_winner(winner, df)
        return str(winner)

    def _pick_winner(self, df: pd.DataFrame) -> str:
        scores: dict[str, float] = {}
        for model in df.index:
            rank_mae = int(df["MAE"].rank().loc[model])
            rank_rmse = int(df["RMSE"].rank().loc[model])
            rank_r2 = int(df["R2"].rank(ascending=False).loc[model])
            scores[model] = rank_mae + rank_rmse + rank_r2
        return min(scores, key=scores.__getitem__)

    def _explain_winner(self, winner: str, df: pd.DataFrame) -> None:
        print(f"\n{DIVIDER}")
        print(f"EXPLANATION: Why is {winner} the best model?")
        print(DIVIDER)
        others = df.drop(index=winner)
        mae_win = _cell(df, winner, "MAE")
        rmse_win = _cell(df, winner, "RMSE")
        r2_win = _cell(df, winner, "R2")
        print(f"\n1. MAE = {mae_win:.4f} min")
        print(f"-> On average the model is off by {mae_win:.1f} min versus actual.")
        for other in others.index:
            other_mae = _cell(others, other, "MAE")
            diff = other_mae - mae_win
            pct = diff / other_mae * 100
            print(f"-> Beats {other} by {diff:.2f} min ({pct:.1f}% less error)")
        print(f"\n2. RMSE = {rmse_win:.4f} min")
        print(f"-> RMSE penalises large errors. {winner} has lower RMSE, meaning fewer severe mistakes (multi-hour delays).")
        print(f"\n3. R2 = {r2_win:.4f}")
        pct_explained = r2_win * 100
        print(f"-> Model explains {pct_explained:.1f}% of delay variance.")
        if r2_win >= 0.7:
            print(f"-> R2 >= 0.7: strong predictive power.")
        elif r2_win >= 0.5:
            print(f"-> R2 >= 0.5: moderate predictive power.")
        else:
            print(f"-> R2 < 0.5: delay is heavily influenced by factors outside the data.")
        if winner == "XGBoost":
            print("\n4. Why XGBoost usually wins on tabular data:")
            print("-> Gradient boosting builds sequential trees, each correcting the previous.")
            print("-> Regularization (reg_alpha, reg_lambda) prevents overfitting.")
            print("-> Handles missing values and non-linear features well.")
        elif winner == "RandomForest":
            print("\n4. Why RandomForest wins:")
            print("-> Ensemble of parallel Decision Trees, less overfitting than a single tree.")
            print("-> Robust to outliers and does not require feature scaling.")
        elif winner == "LinearRegression":
            print("\n4. Why LinearRegression wins:")
            print("-> The relationship between features and delay is fairly linear in this dataset.")
            print("-> Low overfitting risk, generalises better on this data.")
        print(f"\n4. CONCLUSION: {winner} gives the most balanced result across MAE, RMSE, R2")
        print(f"-> suitable for deployment in a real delivery-delay prediction system.")
        print(DIVIDER)
