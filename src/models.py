"""
Model training module.
Defines model configurations and K-Fold training loop.
"""

import logging
from dataclasses import dataclass

import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb

from config.settings import (
    CAT_V1_PARAMS, CAT_V2_PARAMS,
    ET_PARAMS, RF_PARAMS,
    LGB_V1_PARAMS, LGB_V2_PARAMS,
    N_FOLDS, RANDOM_STATE,
    XGB_V1_PARAMS, XGB_V2_PARAMS,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Stores OOF predictions, test predictions, and CV score for a model."""
    name: str
    oof_preds: np.ndarray
    test_preds: np.ndarray
    cv_score: float

    @property
    def score_100(self) -> float:
        return max(0.0, 100.0 * self.cv_score)


def train_xgb(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    params: dict, name: str,
) -> ModelResult:
    """Train XGBoost with K-Fold CV."""
    logger.info(f"\n  ── {name} ──")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof = np.zeros(len(X_train))
    preds = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        oof[val_idx] = model.predict(X_val)
        preds += model.predict(X_test) / N_FOLDS
        logger.info(f"    Fold {fold + 1}: R² = {r2_score(y_val, oof[val_idx]):.6f}")

    cv = r2_score(y_train, oof)
    logger.info(f"    → {name} CV R² = {cv:.6f} | Score = {max(0, 100 * cv):.4f}")
    return ModelResult(name=name, oof_preds=oof, test_preds=preds, cv_score=cv)


def train_lgb(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    params: dict, name: str,
) -> ModelResult:
    """Train LightGBM with K-Fold CV."""
    logger.info(f"\n  ── {name} ──")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof = np.zeros(len(X_train))
    preds = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])

        oof[val_idx] = model.predict(X_val)
        preds += model.predict(X_test) / N_FOLDS
        logger.info(f"    Fold {fold + 1}: R² = {r2_score(y_val, oof[val_idx]):.6f}")

    cv = r2_score(y_train, oof)
    logger.info(f"    → {name} CV R² = {cv:.6f} | Score = {max(0, 100 * cv):.4f}")
    return ModelResult(name=name, oof_preds=oof, test_preds=preds, cv_score=cv)


def train_catboost(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    params: dict, name: str,
) -> ModelResult:
    """Train CatBoost with K-Fold CV."""
    logger.info(f"\n  ── {name} ──")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof = np.zeros(len(X_train))
    preds = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = CatBoostRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=200,
            verbose=0,
        )

        oof[val_idx] = model.predict(X_val)
        preds += model.predict(X_test) / N_FOLDS
        logger.info(f"    Fold {fold + 1}: R² = {r2_score(y_val, oof[val_idx]):.6f}")

    cv = r2_score(y_train, oof)
    logger.info(f"    → {name} CV R² = {cv:.6f} | Score = {max(0, 100 * cv):.4f}")
    return ModelResult(name=name, oof_preds=oof, test_preds=preds, cv_score=cv)


def train_sklearn(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    model_class, params: dict, name: str,
) -> ModelResult:
    """Train a sklearn ensemble (ExtraTrees / RandomForest) with K-Fold CV."""
    logger.info(f"\n  ── {name} ──")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof = np.zeros(len(X_train))
    preds = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = model_class(**params)
        model.fit(X_tr, y_tr)

        oof[val_idx] = model.predict(X_val)
        preds += model.predict(X_test) / N_FOLDS
        logger.info(f"    Fold {fold + 1}: R² = {r2_score(y_val, oof[val_idx]):.6f}")

    cv = r2_score(y_train, oof)
    logger.info(f"    → {name} CV R² = {cv:.6f} | Score = {max(0, 100 * cv):.4f}")
    return ModelResult(name=name, oof_preds=oof, test_preds=preds, cv_score=cv)


def train_all_models(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
) -> list[ModelResult]:
    """Train all 8 models and return results."""
    logger.info("\n[4/6] Training models with 5-Fold CV...")

    results = []

    results.append(train_xgb(X_train, y_train, X_test, XGB_V1_PARAMS, "XGBoost v1"))
    results.append(train_xgb(X_train, y_train, X_test, XGB_V2_PARAMS, "XGBoost v2"))
    results.append(train_lgb(X_train, y_train, X_test, LGB_V1_PARAMS, "LightGBM v1"))
    results.append(train_lgb(X_train, y_train, X_test, LGB_V2_PARAMS, "LightGBM v2"))
    results.append(
        train_catboost(X_train, y_train, X_test, CAT_V1_PARAMS, "CatBoost v1")
    )
    results.append(
        train_catboost(X_train, y_train, X_test, CAT_V2_PARAMS, "CatBoost v2")
    )
    results.append(
        train_sklearn(
            X_train, y_train, X_test, ExtraTreesRegressor, ET_PARAMS, "ExtraTrees"
        )
    )
    results.append(
        train_sklearn(
            X_train, y_train, X_test, RandomForestRegressor, RF_PARAMS, "RandomForest"
        )
    )

    return results
