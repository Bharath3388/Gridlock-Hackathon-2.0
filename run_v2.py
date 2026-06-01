"""
Traffic Demand Prediction — Gridlock Hackathon 2.0
===================================================
Improved Solution v2: Focus on generalization over CV inflation.

Key insight: Train=Day48(full)+Day49(0-2AM), Test=Day49(2-13PM).
Random KFold gives inflated CV (97.5) but LB is only 89.
This solution prioritizes features that generalize across days.

Strategy:
  1. CatBoost native categorical handling (regularized target stats)
  2. Smoothed Day48 lookup features (geo×hour demand)
  3. Multi-seed diverse ensemble with strong regularization
  4. Minimal manual target encoding to avoid overfitting
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_FOLDS = 5
RANDOM_STATE = 42


# =============================================================================
# DATA LOADING
# =============================================================================
def load_data():
    print("=" * 70)
    print("TRAFFIC DEMAND PREDICTION v2 — ANTI-OVERFIT APPROACH")
    print("=" * 70)

    print("\n[1/5] Loading data...")
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f"  Train: {train.shape}, Test: {test.shape}")
    return train, test


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def engineer_features(train, test):
    print("\n[2/5] Feature Engineering...")

    test_index = test["Index"].values.copy()

    # Parse timestamps
    def parse_ts(ts):
        parts = str(ts).split(":")
        h = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 else 0
        return h, m

    for df in [train, test]:
        hm = df["timestamp"].apply(parse_ts)
        df["hour"] = hm.apply(lambda x: x[0])
        df["minute"] = hm.apply(lambda x: x[1])
        df["minutes_of_day"] = df["hour"] * 60 + df["minute"]
        df["quarter_of_day"] = df["minutes_of_day"] // 15
        df["day_of_week"] = df["day"] % 7
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # Cyclical time
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        # Time periods
        df["is_rush"] = ((df["hour"].between(7, 9)) | (df["hour"].between(16, 19))).astype(int)
        df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)

        # Fill missing categoricals with 'Unknown' for CatBoost native handling
        df["RoadType"] = df["RoadType"].fillna("Unknown")
        df["Weather"] = df["Weather"].fillna("Unknown")
        df["LargeVehicles"] = df["LargeVehicles"].fillna("Unknown")
        df["Landmarks"] = df["Landmarks"].fillna("Unknown")

        # Temperature handling
        df["Temperature_missing"] = df["Temperature"].isnull().astype(int)

    # Fill temperature with global median (from train)
    temp_median = train["Temperature"].median()
    train["Temperature"] = train["Temperature"].fillna(temp_median)
    test["Temperature"] = test["Temperature"].fillna(temp_median)

    # ─── DAY 48 LOOKUP FEATURES (key for generalization) ───
    print("  → Creating Day48 lookup features...")
    day48 = train[train["day"] == 48].copy()
    day49_early = train[train["day"] == 49].copy()

    # Geohash mean demand from Day 48 (with smoothing)
    global_mean = train["demand"].mean()
    geo_stats_48 = day48.groupby("geohash")["demand"].agg(["mean", "count", "std"])
    geo_stats_48.columns = ["geo_mean_48", "geo_count_48", "geo_std_48"]
    geo_stats_48["geo_std_48"] = geo_stats_48["geo_std_48"].fillna(0)

    # Bayesian smoothed target encoding: shrink toward global mean
    # smoothed = (count * geo_mean + prior_weight * global_mean) / (count + prior_weight)
    PRIOR_WEIGHT = 20
    geo_stats_48["geo_mean_48_smooth"] = (
        geo_stats_48["geo_count_48"] * geo_stats_48["geo_mean_48"]
        + PRIOR_WEIGHT * global_mean
    ) / (geo_stats_48["geo_count_48"] + PRIOR_WEIGHT)

    for df in [train, test]:
        df["geo_mean_48_smooth"] = df["geohash"].map(
            geo_stats_48["geo_mean_48_smooth"]
        ).fillna(global_mean)
        df["geo_std_48"] = df["geohash"].map(geo_stats_48["geo_std_48"]).fillna(0)

    # Geohash × Hour demand from Day 48 (smoothed)
    geo_hour_48 = day48.groupby(["geohash", "hour"])["demand"].agg(["mean", "count"])
    geo_hour_48.columns = ["geo_hour_mean_48", "geo_hour_count_48"]
    geo_hour_dict = geo_hour_48["geo_hour_mean_48"].to_dict()

    for df in [train, test]:
        df["geo_hour_mean_48"] = df.apply(
            lambda r: geo_hour_dict.get(
                (r["geohash"], r["hour"]), r["geo_mean_48_smooth"]
            ),
            axis=1,
        )

    # Day 49 early hours geohash demand (for daily shift detection)
    geo_mean_49 = day49_early.groupby("geohash")["demand"].agg(["mean", "count"])
    geo_mean_49.columns = ["geo_mean_49early", "geo_count_49early"]
    # Only trust Day 49 stats if we have enough samples
    geo_mean_49["geo_mean_49_smooth"] = np.where(
        geo_mean_49["geo_count_49early"] >= 3,
        geo_mean_49["geo_mean_49early"],
        np.nan,
    )

    for df in [train, test]:
        df["geo_mean_49early"] = df["geohash"].map(
            geo_mean_49["geo_mean_49_smooth"]
        ).fillna(df["geo_mean_48_smooth"])
        # Daily adjustment ratio (clipped for stability)
        df["day_shift_ratio"] = (
            df["geo_mean_49early"] / (df["geo_mean_48_smooth"] + 1e-8)
        ).clip(0.5, 2.0)

    # Hour-level demand from Day 48
    hour_mean_48 = day48.groupby("hour")["demand"].mean()
    for df in [train, test]:
        df["hour_mean_48"] = df["hour"].map(hour_mean_48).fillna(global_mean)

    # RoadType demand from Day 48
    road_mean_48 = day48.groupby("RoadType")["demand"].mean()
    for df in [train, test]:
        df["road_mean_48"] = df["RoadType"].map(road_mean_48).fillna(global_mean)

    # ─── Geohash prefix features (area-level, more stable) ───
    print("  → Creating area-level features...")
    for df in [train, test]:
        df["geo_prefix4"] = df["geohash"].str[:4]
        df["geo_prefix5"] = df["geohash"].str[:5]

    prefix4_mean = day48.copy()
    prefix4_mean["geo_prefix4"] = prefix4_mean["geohash"].str[:4]
    prefix4_stats = prefix4_mean.groupby("geo_prefix4")["demand"].agg(["mean", "count"])
    prefix4_stats.columns = ["prefix4_mean_48", "prefix4_count_48"]
    prefix4_stats["prefix4_mean_smooth"] = (
        prefix4_stats["prefix4_count_48"] * prefix4_stats["prefix4_mean_48"]
        + PRIOR_WEIGHT * global_mean
    ) / (prefix4_stats["prefix4_count_48"] + PRIOR_WEIGHT)

    for df in [train, test]:
        df["prefix4_mean_smooth"] = df["geo_prefix4"].map(
            prefix4_stats["prefix4_mean_smooth"]
        ).fillna(global_mean)

    # ─── Interaction features ───
    print("  → Creating interaction features...")
    road_ord = {"Residential": 0, "Street": 1, "Highway": 2, "Unknown": 0}
    for df in [train, test]:
        df["RoadType_ord"] = df["RoadType"].map(road_ord).fillna(0).astype(int)
        df["NumberofLanes"] = df["NumberofLanes"].fillna(1)
        df["highway_flag"] = (
            (df["RoadType_ord"] == 2) | (df["NumberofLanes"] >= 4)
        ).astype(int)
        df["road_lanes"] = df["RoadType_ord"] * 10 + df["NumberofLanes"]
        df["LargeVehicles_bin"] = (df["LargeVehicles"] == "Allowed").astype(int)
        df["Landmarks_bin"] = (df["Landmarks"] == "Yes").astype(int)

    print(f"  Features ready.")
    return train, test, test_index


# =============================================================================
# MODEL TRAINING
# =============================================================================
def train_models(train, test, test_index):
    print("\n[3/5] Training models...")

    # Define features
    numeric_features = [
        "day", "hour", "minute", "minutes_of_day", "quarter_of_day",
        "day_of_week", "is_weekend", "hour_sin", "hour_cos",
        "is_rush", "is_night",
        "NumberofLanes", "Temperature", "Temperature_missing",
        "RoadType_ord", "highway_flag", "road_lanes",
        "LargeVehicles_bin", "Landmarks_bin",
        # Day48 lookup features (smoothed)
        "geo_mean_48_smooth", "geo_std_48",
        "geo_hour_mean_48", "geo_mean_49early", "day_shift_ratio",
        "hour_mean_48", "road_mean_48", "prefix4_mean_smooth",
    ]

    cat_features = ["geohash", "RoadType", "Weather", "LargeVehicles",
                    "Landmarks", "geo_prefix4", "geo_prefix5"]

    all_features = numeric_features + cat_features
    cat_indices = list(range(len(numeric_features), len(all_features)))

    X_train = train[all_features].copy()
    y_train = train["demand"].values
    X_test = test[all_features].copy()

    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"  Numeric: {len(numeric_features)}, Categorical: {len(cat_features)}")

    # ─── Model configs (diverse for ensemble) ───
    configs = [
        {
            "name": "CatBoost-d8-lr03",
            "params": {
                "iterations": 4000, "depth": 8, "learning_rate": 0.03,
                "l2_leaf_reg": 7, "random_seed": 42,
                "cat_features": cat_indices, "verbose": 0,
                "early_stopping_rounds": 300,
                "bagging_temperature": 0.8,
                "random_strength": 1.5,
            },
        },
        {
            "name": "CatBoost-d6-lr05",
            "params": {
                "iterations": 3000, "depth": 6, "learning_rate": 0.05,
                "l2_leaf_reg": 10, "random_seed": 123,
                "cat_features": cat_indices, "verbose": 0,
                "early_stopping_rounds": 300,
                "bagging_temperature": 1.0,
                "random_strength": 2.0,
            },
        },
        {
            "name": "CatBoost-d10-lr02",
            "params": {
                "iterations": 5000, "depth": 10, "learning_rate": 0.02,
                "l2_leaf_reg": 5, "random_seed": 777,
                "cat_features": cat_indices, "verbose": 0,
                "early_stopping_rounds": 300,
                "bagging_temperature": 0.5,
                "random_strength": 1.0,
            },
        },
        {
            "name": "CatBoost-d7-lr04-reg",
            "params": {
                "iterations": 3500, "depth": 7, "learning_rate": 0.04,
                "l2_leaf_reg": 15, "random_seed": 2024,
                "cat_features": cat_indices, "verbose": 0,
                "early_stopping_rounds": 300,
                "bagging_temperature": 1.2,
                "random_strength": 2.5,
            },
        },
        {
            "name": "CatBoost-d9-lr025",
            "params": {
                "iterations": 4500, "depth": 9, "learning_rate": 0.025,
                "l2_leaf_reg": 6, "random_seed": 555,
                "cat_features": cat_indices, "verbose": 0,
                "early_stopping_rounds": 300,
                "bagging_temperature": 0.7,
                "random_strength": 1.2,
            },
        },
    ]

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    all_oof = {}
    all_preds = {}

    for cfg in configs:
        name = cfg["name"]
        params = cfg["params"]
        print(f"\n  ── {name} ──")

        oof = np.zeros(len(X_train))
        preds = np.zeros(len(X_test))

        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
            model = CatBoostRegressor(**params)
            model.fit(
                X_train.iloc[tr_idx], y_train[tr_idx],
                eval_set=(X_train.iloc[val_idx], y_train[val_idx]),
                verbose=0,
            )
            oof[val_idx] = model.predict(X_train.iloc[val_idx])
            preds += model.predict(X_test) / N_FOLDS
            fold_r2 = r2_score(y_train[val_idx], oof[val_idx])
            print(f"    Fold {fold + 1}: R² = {fold_r2:.6f}")

        cv = r2_score(y_train, oof)
        print(f"    → CV R² = {cv:.6f} | Score = {100 * cv:.4f}")
        all_oof[name] = oof
        all_preds[name] = preds

    return all_oof, all_preds, y_train, X_test, test_index


# =============================================================================
# ENSEMBLE
# =============================================================================
def build_ensemble(all_oof, all_preds, y_train, test_index):
    print("\n[4/5] Building ensemble...")

    model_names = list(all_oof.keys())

    # Simple average (most robust against overfitting)
    avg_oof = np.mean([all_oof[n] for n in model_names], axis=0)
    avg_preds = np.mean([all_preds[n] for n in model_names], axis=0)
    avg_r2 = r2_score(y_train, avg_oof)
    print(f"  Simple Average CV R² = {avg_r2:.6f} | Score = {100 * avg_r2:.4f}")

    # Rank-based blending (more robust than weight optimization which can overfit)
    # Give slight preference to models with higher regularization (less overfit)
    # But use equal weights for maximum diversity benefit
    final_preds = avg_preds

    # Clip to valid range
    final_preds = np.clip(final_preds, 0, 1)

    print(f"\n  Using: Simple Average (most robust for LB)")
    print(f"  Prediction stats: min={final_preds.min():.4f}, max={final_preds.max():.4f}, "
          f"mean={final_preds.mean():.4f}")

    return final_preds


# =============================================================================
# MAIN
# =============================================================================
def main():
    start_time = time.time()

    train, test = load_data()
    train, test, test_index = engineer_features(train, test)
    all_oof, all_preds, y_train, X_test, test_index = train_models(train, test, test_index)
    final_preds = build_ensemble(all_oof, all_preds, y_train, test_index)

    # Generate submission
    print("\n[5/5] Generating submission...")
    submission = pd.DataFrame({"Index": test_index, "demand": final_preds})
    sub_path = os.path.join(OUTPUT_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)

    print(f"  Saved: {sub_path}")
    print(f"  Shape: {submission.shape}")
    print(f"  Head:\n{submission.head()}")

    elapsed = time.time() - start_time
    print(f"\n  Completed in {elapsed / 60:.1f} minutes")
    print("=" * 70)


if __name__ == "__main__":
    main()
