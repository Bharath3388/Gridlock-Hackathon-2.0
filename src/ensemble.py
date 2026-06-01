"""
Ensemble module.
Implements weighted blending and stacking meta-learner strategies.
"""

import logging
from itertools import product

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from config.settings import META_LEARNER_ALPHA, N_FOLDS, RANDOM_STATE
from src.models import ModelResult

logger = logging.getLogger(__name__)


def find_optimal_blend(results: list[ModelResult], y_train: np.ndarray) -> tuple:
    """
    Grid-search for optimal weighted blend of gradient boosting models,
    then mix with tree ensemble models.

    Returns:
        (blend_oof, blend_test, blend_score)
    """
    logger.info("  Searching optimal blend weights...")

    # Simple average baseline
    simple_avg_oof = np.mean([r.oof_preds for r in results], axis=0)
    simple_avg_test = np.mean([r.test_preds for r in results], axis=0)
    simple_score = r2_score(y_train, simple_avg_oof)
    logger.info(
        f"  Simple Average CV R² = {simple_score:.6f} | "
        f"Score = {max(0, 100 * simple_score):.4f}"
    )

    # Separate gradient boosting models from tree ensembles
    gb_results = [r for r in results if r.name not in ("ExtraTrees", "RandomForest")]
    et_result = next((r for r in results if r.name == "ExtraTrees"), None)
    rf_result = next((r for r in results if r.name == "RandomForest"), None)

    # Grid search weights for GB models
    best_score = -np.inf
    best_weights = None
    weight_options = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    n_gb = len(gb_results)
    for weights_partial in product(weight_options, repeat=n_gb - 1):
        w_last = 1.0 - sum(weights_partial)
        if w_last < -0.01 or w_last > 0.5:
            continue
        w_last = max(0, w_last)
        weights = list(weights_partial) + [w_last]

        blend = sum(
            w * r.oof_preds for w, r in zip(weights, gb_results)
        )
        score = r2_score(y_train, blend)
        if score > best_score:
            best_score = score
            best_weights = weights

    logger.info(
        f"  Best GB Blend CV R² = {best_score:.6f} | "
        f"Score = {max(0, 100 * best_score):.4f}"
    )
    logger.info(
        f"  Weights: {dict(zip([r.name for r in gb_results], best_weights))}"
    )

    # Build GB blend
    gb_blend_oof = sum(
        w * r.oof_preds for w, r in zip(best_weights, gb_results)
    )
    gb_blend_test = sum(
        w * r.test_preds for w, r in zip(best_weights, gb_results)
    )

    # Mix GB blend with ET/RF
    best_final_score = best_score
    best_mix = (1.0, 0.0, 0.0)

    if et_result and rf_result:
        for gb_w in np.arange(0.7, 1.01, 0.05):
            for et_w in np.arange(0.0, 0.31, 0.05):
                rf_w = 1.0 - gb_w - et_w
                if rf_w < -0.01:
                    continue
                rf_w = max(0, rf_w)
                blend = (
                    gb_w * gb_blend_oof
                    + et_w * et_result.oof_preds
                    + rf_w * rf_result.oof_preds
                )
                score = r2_score(y_train, blend)
                if score > best_final_score:
                    best_final_score = score
                    best_mix = (gb_w, et_w, rf_w)

    logger.info(
        f"  Final Blend CV R² = {best_final_score:.6f} | "
        f"Score = {max(0, 100 * best_final_score):.4f}"
    )
    logger.info(
        f"  Mix: GB={best_mix[0]:.2f}, ET={best_mix[1]:.2f}, RF={best_mix[2]:.2f}"
    )

    final_oof = best_mix[0] * gb_blend_oof
    final_test = best_mix[0] * gb_blend_test
    if et_result:
        final_oof += best_mix[1] * et_result.oof_preds
        final_test += best_mix[1] * et_result.test_preds
    if rf_result:
        final_oof += best_mix[2] * rf_result.oof_preds
        final_test += best_mix[2] * rf_result.test_preds

    return final_oof, final_test, best_final_score, simple_avg_test, simple_score


def train_stacking(
    results: list[ModelResult],
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,
    feature_cols: list[str], stacking_features: list[str],
) -> tuple:
    """
    Train a Ridge stacking meta-learner on OOF predictions
    augmented with important original features.

    Returns:
        (stacking_oof, stacking_test, stacking_score)
    """
    logger.info("\n  Training stacking meta-learner...")

    stack_train = np.column_stack([r.oof_preds for r in results])
    stack_test = np.column_stack([r.test_preds for r in results])

    # Add important original features
    important_idx = [
        feature_cols.index(f) for f in stacking_features if f in feature_cols
    ]
    if important_idx:
        stack_train = np.column_stack([stack_train, X_train[:, important_idx]])
        stack_test = np.column_stack([stack_test, X_test[:, important_idx]])

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    meta_oof = np.zeros(len(X_train))
    meta_preds = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(kf.split(stack_train)):
        X_tr, X_val = stack_train[train_idx], stack_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        meta = Ridge(alpha=META_LEARNER_ALPHA)
        meta.fit(X_tr, y_tr)

        meta_oof[val_idx] = meta.predict(X_val)
        meta_preds += meta.predict(stack_test) / N_FOLDS

    meta_score = r2_score(y_train, meta_oof)
    logger.info(
        f"  Stacking CV R² = {meta_score:.6f} | "
        f"Score = {max(0, 100 * meta_score):.4f}"
    )

    return meta_oof, meta_preds, meta_score


def build_ensemble(
    results: list[ModelResult],
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,
    feature_cols: list[str], stacking_features: list[str],
) -> np.ndarray:
    """
    Build ensemble: compare weighted blend, stacking, and simple average.
    Returns the best final test predictions.
    """
    logger.info("\n[5/6] Building ensemble...")
    logger.info(f"  Models: {[r.name for r in results]}")

    # Weighted blend
    blend_oof, blend_test, blend_score, simple_test, simple_score = (
        find_optimal_blend(results, y_train)
    )

    # Stacking
    _, stacking_test, stacking_score = train_stacking(
        results, X_train, y_train, X_test, feature_cols, stacking_features,
    )

    # Compare strategies
    logger.info("\n  Comparing strategies:")
    logger.info(f"    Simple Average:   {max(0, 100 * simple_score):.4f}")
    logger.info(f"    Weighted Blend:   {max(0, 100 * blend_score):.4f}")
    logger.info(f"    Stacking:         {max(0, 100 * stacking_score):.4f}")

    strategies = {
        "SIMPLE_AVERAGE": (simple_score, simple_test),
        "WEIGHTED_BLEND": (blend_score, blend_test),
        "STACKING": (stacking_score, stacking_test),
    }
    best_name = max(strategies, key=lambda k: strategies[k][0])
    best_score, best_preds = strategies[best_name]

    logger.info(
        f"\n  → Using: {best_name} (Score: {max(0, 100 * best_score):.4f})"
    )

    # Clip predictions to valid demand range [0, 1]
    return np.clip(best_preds, 0, 1), best_score, best_name
