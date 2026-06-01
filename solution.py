"""
=============================================================================
Traffic Demand Prediction — Gridlock Hackathon 2.0
=============================================================================
Advanced Ensemble: XGBoost + LightGBM + CatBoost (×2 configs each) +
                   ExtraTrees + RandomForest with Ridge Stacking

Evaluation Metric: score = max(0, 100 * R²_score(actual, predicted))
Target: Predict 'demand' for 41,778 test samples
=============================================================================
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import pygeohash as pgh
from itertools import product
import time
import os

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "dataset")
OUTPUT_DIR = os.path.dirname(__file__)
N_FOLDS = 5
RANDOM_STATE = 42

# =============================================================================
# 2. DATA LOADING
# =============================================================================
print("=" * 70)
print("TRAFFIC DEMAND PREDICTION — GRIDLOCK HACKATHON 2.0")
print("=" * 70)

start_time = time.time()

print("\n[1/6] Loading data...")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

print(f"  Train shape: {train.shape}")
print(f"  Test shape:  {test.shape}")

# Save test Index for submission
test_index = test['Index'].values

# Combine for consistent feature engineering
train['is_train'] = 1
test['is_train'] = 0
test['demand'] = np.nan
combined = pd.concat([train, test], axis=0, ignore_index=True)
print(f"  Combined shape: {combined.shape}")

# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================
print("\n[2/6] Feature Engineering...")

# --- 3.1 Geohash Decoding ---
print("  → Decoding geohash to lat/lon...")
geo_cache = {}
for gh in combined['geohash'].unique():
    try:
        lat, lon = pgh.decode(gh)
        geo_cache[gh] = (lat, lon)
    except:
        geo_cache[gh] = (0.0, 0.0)

combined['latitude'] = combined['geohash'].map(lambda x: geo_cache[x][0])
combined['longitude'] = combined['geohash'].map(lambda x: geo_cache[x][1])

# Geohash prefixes for hierarchical location
combined['geo_prefix3'] = combined['geohash'].str[:3]
combined['geo_prefix4'] = combined['geohash'].str[:4]
combined['geo_prefix5'] = combined['geohash'].str[:5]

# --- 3.2 Temporal Features ---
print("  → Engineering temporal features...")


def parse_timestamp(ts):
    """Convert 'H:M' string to (hour, minute)."""
    parts = str(ts).split(':')
    return int(parts[0]), int(parts[1])


hours_minutes = combined['timestamp'].apply(parse_timestamp)
combined['hour'] = hours_minutes.apply(lambda x: x[0])
combined['minute'] = hours_minutes.apply(lambda x: x[1])
combined['minutes_of_day'] = combined['hour'] * 60 + combined['minute']
combined['quarter_of_day'] = combined['minutes_of_day'] // 15

# Cyclical encoding
combined['hour_sin'] = np.sin(2 * np.pi * combined['hour'] / 24)
combined['hour_cos'] = np.cos(2 * np.pi * combined['hour'] / 24)
combined['minute_sin'] = np.sin(2 * np.pi * combined['minutes_of_day'] / 1440)
combined['minute_cos'] = np.cos(2 * np.pi * combined['minutes_of_day'] / 1440)

# Time period indicators
combined['is_morning_rush'] = ((combined['hour'] >= 7) & (combined['hour'] <= 9)).astype(int)
combined['is_evening_rush'] = ((combined['hour'] >= 16) & (combined['hour'] <= 19)).astype(int)
combined['is_rush_hour'] = (combined['is_morning_rush'] | combined['is_evening_rush']).astype(int)
combined['is_night'] = ((combined['hour'] >= 22) | (combined['hour'] <= 5)).astype(int)
combined['is_midday'] = ((combined['hour'] >= 10) & (combined['hour'] <= 15)).astype(int)

# Day features
combined['day_of_week'] = combined['day'] % 7
combined['is_weekend'] = (combined['day_of_week'] >= 5).astype(int)

# --- 3.3 Missing Value Handling ---
print("  → Handling missing values...")

# RoadType: fill with geohash-group mode, fallback to global mode
road_mode_global = combined['RoadType'].mode()[0]
combined['RoadType_missing'] = combined['RoadType'].isnull().astype(int)
combined['RoadType'] = combined.groupby('geohash')['RoadType'].transform(
    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else road_mode_global)
)
combined['RoadType'] = combined['RoadType'].fillna(road_mode_global)

# Temperature: fill with geohash-group median, fallback to global median
temp_median_global = combined['Temperature'].median()
combined['Temperature_missing'] = combined['Temperature'].isnull().astype(int)
combined['Temperature'] = combined.groupby('geohash')['Temperature'].transform(
    lambda x: x.fillna(x.median())
)
combined['Temperature'] = combined['Temperature'].fillna(temp_median_global)

# Weather: fill with timestamp-group mode, fallback to global mode
weather_mode_global = combined['Weather'].mode()[0]
combined['Weather_missing'] = combined['Weather'].isnull().astype(int)
combined['Weather'] = combined.groupby('timestamp')['Weather'].transform(
    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else weather_mode_global)
)
combined['Weather'] = combined['Weather'].fillna(weather_mode_global)

# --- 3.4 Categorical Encoding ---
print("  → Encoding categorical features...")

# RoadType ordinal
road_type_map = {'Residential': 0, 'Street': 1, 'Highway': 2}
combined['RoadType_encoded'] = combined['RoadType'].map(road_type_map).fillna(0).astype(int)

# LargeVehicles binary
combined['LargeVehicles_encoded'] = (combined['LargeVehicles'] == 'Allowed').astype(int)

# Landmarks binary
combined['Landmarks_encoded'] = (combined['Landmarks'] == 'Yes').astype(int)

# Weather ordinal + one-hot
weather_map = {'Sunny': 0, 'Rainy': 1, 'Foggy': 2, 'Snowy': 3}
combined['Weather_encoded'] = combined['Weather'].map(weather_map).fillna(0).astype(int)
for w in ['Sunny', 'Rainy', 'Foggy', 'Snowy']:
    combined[f'weather_{w}'] = (combined['Weather'] == w).astype(int)

# One-hot for RoadType
for rt in ['Residential', 'Street', 'Highway']:
    combined[f'road_{rt}'] = (combined['RoadType'] == rt).astype(int)

# --- 3.5 Temperature Features ---
print("  → Engineering temperature features...")
combined['temp_bin'] = pd.cut(combined['Temperature'], bins=10, labels=False)
combined['temp_squared'] = combined['Temperature'] ** 2
combined['temp_abs'] = combined['Temperature'].abs()
combined['is_cold'] = (combined['Temperature'] < 5).astype(int)
combined['is_hot'] = (combined['Temperature'] > 30).astype(int)
combined['is_moderate'] = ((combined['Temperature'] >= 10) & (combined['Temperature'] <= 25)).astype(int)

# --- 3.6 Interaction Features ---
print("  → Creating interaction features...")
combined['road_lanes'] = combined['RoadType_encoded'] * 10 + combined['NumberofLanes']
combined['highway_flag'] = ((combined['RoadType_encoded'] == 2) | (combined['NumberofLanes'] >= 4)).astype(int)
combined['lane_pressure'] = combined['NumberofLanes'] / (combined['RoadType_encoded'] + 1)
combined['road_time'] = combined['RoadType_encoded'] * 10 + (combined['minutes_of_day'] // 60)
combined['lanes_large'] = combined['NumberofLanes'] * combined['LargeVehicles_encoded']
combined['temp_weather'] = combined['Temperature'] * combined['Weather_encoded']
combined['hour_road'] = combined['hour'] * 10 + combined['RoadType_encoded']
combined['lat_hour'] = combined['latitude'] * combined['hour']
combined['lon_hour'] = combined['longitude'] * combined['hour']

# --- 3.7 Target Encoding / Aggregation Features (LEAK-FREE) ---
print("  → Computing target encoding features (leak-free)...")
train_mask = combined['is_train'] == 1
train_data = combined[train_mask]
global_mean = train_data['demand'].mean()

# Geohash-level stats
geo_stats = train_data.groupby('geohash')['demand'].agg(['mean', 'std', 'median', 'count'])
geo_stats.columns = ['geo_demand_mean', 'geo_demand_std', 'geo_demand_median', 'geo_demand_count']
geo_stats['geo_demand_std'] = geo_stats['geo_demand_std'].fillna(0)
combined = combined.merge(geo_stats, on='geohash', how='left')
combined['geo_demand_mean'] = combined['geo_demand_mean'].fillna(global_mean)
combined['geo_demand_std'] = combined['geo_demand_std'].fillna(0)
combined['geo_demand_median'] = combined['geo_demand_median'].fillna(global_mean)
combined['geo_demand_count'] = combined['geo_demand_count'].fillna(1)

# Demand volatility (coefficient of variation)
combined['geo_demand_cv'] = combined['geo_demand_std'] / (combined['geo_demand_mean'] + 1e-8)

# Geohash prefix demand means
for prefix_col in ['geo_prefix4', 'geo_prefix5']:
    prefix_stats = train_data.groupby(prefix_col)['demand'].agg(['mean', 'std'])
    prefix_stats.columns = [f'{prefix_col}_demand_mean', f'{prefix_col}_demand_std']
    prefix_stats[f'{prefix_col}_demand_std'] = prefix_stats[f'{prefix_col}_demand_std'].fillna(0)
    combined = combined.merge(prefix_stats, on=prefix_col, how='left')
    combined[f'{prefix_col}_demand_mean'] = combined[f'{prefix_col}_demand_mean'].fillna(global_mean)
    combined[f'{prefix_col}_demand_std'] = combined[f'{prefix_col}_demand_std'].fillna(0)

# RoadType demand
road_stats = train_data.groupby('RoadType_encoded')['demand'].agg(['mean', 'std'])
road_stats.columns = ['road_demand_mean', 'road_demand_std']
combined = combined.merge(road_stats, on='RoadType_encoded', how='left')
combined['road_demand_mean'] = combined['road_demand_mean'].fillna(global_mean)
combined['road_demand_std'] = combined['road_demand_std'].fillna(0)

# Hour demand
hour_stats = train_data.groupby('hour')['demand'].agg(['mean', 'std'])
hour_stats.columns = ['hour_demand_mean', 'hour_demand_std']
combined = combined.merge(hour_stats, on='hour', how='left')
combined['hour_demand_mean'] = combined['hour_demand_mean'].fillna(global_mean)
combined['hour_demand_std'] = combined['hour_demand_std'].fillna(0)

# Geohash + hour demand
geo_hour_stats = train_data.groupby(['geohash', 'hour'])['demand'].agg(['mean', 'count'])
geo_hour_stats.columns = ['geo_hour_demand_mean', 'geo_hour_count']
combined = combined.merge(geo_hour_stats, on=['geohash', 'hour'], how='left')
combined['geo_hour_demand_mean'] = combined['geo_hour_demand_mean'].fillna(combined['geo_demand_mean'])
combined['geo_hour_count'] = combined['geo_hour_count'].fillna(0)

# RoadType + hour demand
road_hour_stats = train_data.groupby(['RoadType_encoded', 'hour'])['demand'].agg(['mean'])
road_hour_stats.columns = ['road_hour_demand_mean']
combined = combined.merge(road_hour_stats, on=['RoadType_encoded', 'hour'], how='left')
combined['road_hour_demand_mean'] = combined['road_hour_demand_mean'].fillna(global_mean)

# NumberofLanes demand
lanes_demand = train_data.groupby('NumberofLanes')['demand'].agg(['mean'])
lanes_demand.columns = ['lanes_demand_mean']
combined = combined.merge(lanes_demand, on='NumberofLanes', how='left')
combined['lanes_demand_mean'] = combined['lanes_demand_mean'].fillna(global_mean)

# Geohash + RoadType demand
geo_road_stats = train_data.groupby(['geohash', 'RoadType_encoded'])['demand'].agg(['mean'])
geo_road_stats.columns = ['geo_road_demand_mean']
combined = combined.merge(geo_road_stats, on=['geohash', 'RoadType_encoded'], how='left')
combined['geo_road_demand_mean'] = combined['geo_road_demand_mean'].fillna(combined['geo_demand_mean'])

# RoadType + NumberofLanes demand
road_lane_stats = train_data.groupby(['RoadType_encoded', 'NumberofLanes'])['demand'].agg(['mean'])
road_lane_stats.columns = ['road_lane_demand_mean']
combined = combined.merge(road_lane_stats, on=['RoadType_encoded', 'NumberofLanes'], how='left')
combined['road_lane_demand_mean'] = combined['road_lane_demand_mean'].fillna(global_mean)

# Timestamp demand
time_stats = train_data.groupby('timestamp')['demand'].agg(['mean'])
time_stats.columns = ['time_demand_mean']
combined = combined.merge(time_stats, on='timestamp', how='left')
combined['time_demand_mean'] = combined['time_demand_mean'].fillna(global_mean)

# --- 3.8 Geohash Label Encoding & Spatial Features ---
print("  → Creating spatial features...")
le_geo = LabelEncoder()
combined['geohash_le'] = le_geo.fit_transform(combined['geohash'])

le_prefix4 = LabelEncoder()
combined['geo_prefix4_le'] = le_prefix4.fit_transform(combined['geo_prefix4'])

le_prefix5 = LabelEncoder()
combined['geo_prefix5_le'] = le_prefix5.fit_transform(combined['geo_prefix5'])

# Distance from geographic center
lat_mean = combined['latitude'].mean()
lon_mean = combined['longitude'].mean()
combined['dist_from_center'] = np.sqrt(
    (combined['latitude'] - lat_mean) ** 2 +
    (combined['longitude'] - lon_mean) ** 2
)

# Spatial binning
combined['lat_bin'] = pd.cut(combined['latitude'], bins=20, labels=False)
combined['lon_bin'] = pd.cut(combined['longitude'], bins=20, labels=False)
combined['spatial_bin'] = combined['lat_bin'] * 20 + combined['lon_bin']

# --- 3.9 Temperature Anomaly ---
geo_temp_mean = combined.groupby('geohash')['Temperature'].transform('mean')
combined['temp_anomaly'] = combined['Temperature'] - geo_temp_mean

# Geohash frequency (popularity)
geo_freq = combined['geohash'].value_counts().to_dict()
combined['geohash_freq'] = combined['geohash'].map(geo_freq)

print(f"  Total features engineered: {combined.shape[1]}")

# =============================================================================
# 4. PREPARE TRAINING DATA
# =============================================================================
print("\n[3/6] Preparing training and test data...")

feature_cols = [
    # Core features
    'NumberofLanes', 'LargeVehicles_encoded', 'Landmarks_encoded',
    'Temperature', 'RoadType_encoded', 'Weather_encoded',

    # Geospatial
    'latitude', 'longitude', 'geohash_le', 'geo_prefix4_le', 'geo_prefix5_le',
    'dist_from_center', 'lat_bin', 'lon_bin', 'spatial_bin',

    # Temporal
    'day', 'minutes_of_day', 'hour', 'minute', 'quarter_of_day',
    'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',
    'is_morning_rush', 'is_evening_rush', 'is_rush_hour',
    'is_night', 'is_midday', 'day_of_week', 'is_weekend',

    # Temperature features
    'Temperature_missing', 'temp_bin', 'temp_squared', 'temp_abs',
    'is_cold', 'is_hot', 'is_moderate', 'temp_anomaly',

    # Missing indicators
    'RoadType_missing', 'Weather_missing',

    # One-hot
    'weather_Sunny', 'weather_Rainy', 'weather_Foggy', 'weather_Snowy',
    'road_Residential', 'road_Street', 'road_Highway',

    # Interaction features
    'road_lanes', 'highway_flag', 'lane_pressure', 'road_time',
    'lanes_large', 'temp_weather', 'hour_road', 'lat_hour', 'lon_hour',

    # Target encoding / aggregation
    'geo_demand_mean', 'geo_demand_std', 'geo_demand_median', 'geo_demand_count',
    'geo_demand_cv',
    'geo_prefix4_demand_mean', 'geo_prefix4_demand_std',
    'geo_prefix5_demand_mean', 'geo_prefix5_demand_std',
    'road_demand_mean', 'road_demand_std',
    'hour_demand_mean', 'hour_demand_std',
    'geo_hour_demand_mean', 'geo_hour_count',
    'road_hour_demand_mean', 'lanes_demand_mean',
    'geo_road_demand_mean', 'road_lane_demand_mean',
    'time_demand_mean',

    # Extra spatial
    'geohash_freq',
]

# Verify all features exist
missing_feats = [f for f in feature_cols if f not in combined.columns]
if missing_feats:
    print(f"  WARNING: Missing features: {missing_feats}")
    feature_cols = [f for f in feature_cols if f in combined.columns]

# Split back
train_processed = combined[combined['is_train'] == 1].copy()
test_processed = combined[combined['is_train'] == 0].copy()

X_train = train_processed[feature_cols].values.astype(np.float32)
y_train = train_processed['demand'].values.astype(np.float32)
X_test = test_processed[feature_cols].values.astype(np.float32)

print(f"  X_train shape: {X_train.shape}")
print(f"  X_test shape:  {X_test.shape}")
print(f"  Features: {len(feature_cols)}")

# =============================================================================
# 5. MODEL TRAINING WITH K-FOLD CV
# =============================================================================
print("\n[4/6] Training models with 5-Fold CV...")

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Storage for predictions
oof_preds = {}
test_preds = {}

# ---- 5.1 XGBoost v1 ----
print("\n  ── XGBoost v1 ──")
xgb_oof = np.zeros(len(X_train))
xgb_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    model = xgb.XGBRegressor(
        n_estimators=3000, max_depth=8, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, colsample_bylevel=0.7,
        min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
        tree_method='hist', random_state=RANDOM_STATE, n_jobs=-1,
        verbosity=0
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    xgb_oof[val_idx] = model.predict(X_val)
    xgb_preds += model.predict(X_test) / N_FOLDS
    print(f"    Fold {fold+1}: R² = {r2_score(y_val, xgb_oof[val_idx]):.6f}")

xgb_cv = r2_score(y_train, xgb_oof)
print(f"    → XGB v1 CV R² = {xgb_cv:.6f} | Score = {max(0, 100*xgb_cv):.4f}")
oof_preds['xgb1'] = xgb_oof
test_preds['xgb1'] = xgb_preds

# ---- 5.2 XGBoost v2 ----
print("\n  ── XGBoost v2 ──")
xgb2_oof = np.zeros(len(X_train))
xgb2_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    model = xgb.XGBRegressor(
        n_estimators=4000, max_depth=10, learning_rate=0.02,
        subsample=0.75, colsample_bytree=0.65, colsample_bylevel=0.65,
        min_child_weight=3, reg_alpha=0.05, reg_lambda=1.5,
        tree_method='hist', random_state=123, n_jobs=-1,
        verbosity=0
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    xgb2_oof[val_idx] = model.predict(X_val)
    xgb2_preds += model.predict(X_test) / N_FOLDS
    print(f"    Fold {fold+1}: R² = {r2_score(y_val, xgb2_oof[val_idx]):.6f}")

xgb2_cv = r2_score(y_train, xgb2_oof)
print(f"    → XGB v2 CV R² = {xgb2_cv:.6f} | Score = {max(0, 100*xgb2_cv):.4f}")
oof_preds['xgb2'] = xgb2_oof
test_preds['xgb2'] = xgb2_preds

# ---- 5.3 LightGBM v1 ----
print("\n  ── LightGBM v1 ──")
lgb_oof = np.zeros(len(X_train))
lgb_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    model = lgb.LGBMRegressor(
        n_estimators=3000, max_depth=8, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=1.0, num_leaves=127,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])

    lgb_oof[val_idx] = model.predict(X_val)
    lgb_preds += model.predict(X_test) / N_FOLDS
    print(f"    Fold {fold+1}: R² = {r2_score(y_val, lgb_oof[val_idx]):.6f}")

lgb_cv = r2_score(y_train, lgb_oof)
print(f"    → LGB v1 CV R² = {lgb_cv:.6f} | Score = {max(0, 100*lgb_cv):.4f}")
oof_preds['lgb1'] = lgb_oof
test_preds['lgb1'] = lgb_preds

# ---- 5.4 LightGBM v2 ----
print("\n  ── LightGBM v2 ──")
lgb2_oof = np.zeros(len(X_train))
lgb2_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    model = lgb.LGBMRegressor(
        n_estimators=4000, max_depth=-1, learning_rate=0.02,
        subsample=0.75, colsample_bytree=0.65, min_child_samples=10,
        reg_alpha=0.05, reg_lambda=1.5, num_leaves=255,
        random_state=123, n_jobs=-1, verbose=-1
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])

    lgb2_oof[val_idx] = model.predict(X_val)
    lgb2_preds += model.predict(X_test) / N_FOLDS
    print(f"    Fold {fold+1}: R² = {r2_score(y_val, lgb2_oof[val_idx]):.6f}")

lgb2_cv = r2_score(y_train, lgb2_oof)
print(f"    → LGB v2 CV R² = {lgb2_cv:.6f} | Score = {max(0, 100*lgb2_cv):.4f}")
oof_preds['lgb2'] = lgb2_oof
test_preds['lgb2'] = lgb2_preds

# ---- 5.5 CatBoost v1 ----
print("\n  ── CatBoost v1 ──")
cat_oof = np.zeros(len(X_train))
cat_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    model = CatBoostRegressor(
        iterations=3000, depth=8, learning_rate=0.03,
        l2_leaf_reg=3, random_seed=RANDOM_STATE,
        bootstrap_type='Bayesian', bagging_temperature=0.5,
        verbose=0
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=200, verbose=0)

    cat_oof[val_idx] = model.predict(X_val)
    cat_preds += model.predict(X_test) / N_FOLDS
    print(f"    Fold {fold+1}: R² = {r2_score(y_val, cat_oof[val_idx]):.6f}")

cat_cv = r2_score(y_train, cat_oof)
print(f"    → CAT v1 CV R² = {cat_cv:.6f} | Score = {max(0, 100*cat_cv):.4f}")
oof_preds['cat1'] = cat_oof
test_preds['cat1'] = cat_preds

# ---- 5.6 CatBoost v2 ----
print("\n  ── CatBoost v2 ──")
cat2_oof = np.zeros(len(X_train))
cat2_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    model = CatBoostRegressor(
        iterations=4000, depth=10, learning_rate=0.02,
        l2_leaf_reg=5, random_seed=123,
        bootstrap_type='Bayesian', bagging_temperature=1.0,
        verbose=0
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=200, verbose=0)

    cat2_oof[val_idx] = model.predict(X_val)
    cat2_preds += model.predict(X_test) / N_FOLDS
    print(f"    Fold {fold+1}: R² = {r2_score(y_val, cat2_oof[val_idx]):.6f}")

cat2_cv = r2_score(y_train, cat2_oof)
print(f"    → CAT v2 CV R² = {cat2_cv:.6f} | Score = {max(0, 100*cat2_cv):.4f}")
oof_preds['cat2'] = cat2_oof
test_preds['cat2'] = cat2_preds

# ---- 5.7 ExtraTrees ----
print("\n  ── ExtraTrees ──")
et_oof = np.zeros(len(X_train))
et_preds_arr = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    model = ExtraTreesRegressor(
        n_estimators=500, max_depth=30, min_samples_leaf=2,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_tr, y_tr)

    et_oof[val_idx] = model.predict(X_val)
    et_preds_arr += model.predict(X_test) / N_FOLDS
    print(f"    Fold {fold+1}: R² = {r2_score(y_val, et_oof[val_idx]):.6f}")

et_cv = r2_score(y_train, et_oof)
print(f"    → ET CV R² = {et_cv:.6f} | Score = {max(0, 100*et_cv):.4f}")
oof_preds['et'] = et_oof
test_preds['et'] = et_preds_arr

# ---- 5.8 RandomForest ----
print("\n  ── RandomForest ──")
rf_oof = np.zeros(len(X_train))
rf_preds_arr = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    model = RandomForestRegressor(
        n_estimators=500, max_depth=25, min_samples_leaf=2,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_tr, y_tr)

    rf_oof[val_idx] = model.predict(X_val)
    rf_preds_arr += model.predict(X_test) / N_FOLDS
    print(f"    Fold {fold+1}: R² = {r2_score(y_val, rf_oof[val_idx]):.6f}")

rf_cv = r2_score(y_train, rf_oof)
print(f"    → RF CV R² = {rf_cv:.6f} | Score = {max(0, 100*rf_cv):.4f}")
oof_preds['rf'] = rf_oof
test_preds['rf'] = rf_preds_arr

# =============================================================================
# 6. ENSEMBLE
# =============================================================================
print("\n[5/6] Building ensemble...")

model_names = list(oof_preds.keys())
print(f"  Models: {model_names}")

# ---- 6.1 Optimal Weighted Blend ----
print("  Searching optimal blend weights...")

# Simple average as baseline
simple_avg_oof = np.mean([oof_preds[name] for name in model_names], axis=0)
simple_avg_test = np.mean([test_preds[name] for name in model_names], axis=0)
simple_score = r2_score(y_train, simple_avg_oof)
print(f"  Simple Average CV R² = {simple_score:.6f} | Score = {max(0, 100*simple_score):.4f}")

# Grid search for weights (coarse, top 6 models only for speed)
best_score = -np.inf
best_weights = None
weight_options = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

# Use only gradient boosting models for weight search (faster)
gb_names = ['xgb1', 'xgb2', 'lgb1', 'lgb2', 'cat1', 'cat2']

for w1, w2, w3, w4, w5 in product(weight_options, repeat=5):
    w6 = 1.0 - w1 - w2 - w3 - w4 - w5
    if w6 < -0.01 or w6 > 0.5:
        continue
    w6 = max(0, w6)
    weights = [w1, w2, w3, w4, w5, w6]

    blend = sum(w * oof_preds[name] for w, name in zip(weights, gb_names))
    score = r2_score(y_train, blend)

    if score > best_score:
        best_score = score
        best_weights = weights

print(f"  Best GB Blend CV R² = {best_score:.6f} | Score = {max(0, 100*best_score):.4f}")
print(f"  Weights: {dict(zip(gb_names, best_weights))}")

# Build GB blend predictions
gb_blend_oof = sum(w * oof_preds[name] for w, name in zip(best_weights, gb_names))
gb_blend_test = sum(w * test_preds[name] for w, name in zip(best_weights, gb_names))

# Combine GB blend with tree models
# Try mixing in ET and RF
best_final_score = best_score
best_mix = (1.0, 0.0, 0.0)
for gb_w in np.arange(0.7, 1.01, 0.05):
    for et_w in np.arange(0.0, 0.31, 0.05):
        rf_w = 1.0 - gb_w - et_w
        if rf_w < -0.01:
            continue
        rf_w = max(0, rf_w)
        blend = gb_w * gb_blend_oof + et_w * et_oof + rf_w * rf_oof
        score = r2_score(y_train, blend)
        if score > best_final_score:
            best_final_score = score
            best_mix = (gb_w, et_w, rf_w)

print(f"  Final Blend CV R² = {best_final_score:.6f} | Score = {max(0, 100*best_final_score):.4f}")
print(f"  Mix: GB={best_mix[0]:.2f}, ET={best_mix[1]:.2f}, RF={best_mix[2]:.2f}")

final_blend_oof = best_mix[0] * gb_blend_oof + best_mix[1] * et_oof + best_mix[2] * rf_oof
final_blend_test = best_mix[0] * gb_blend_test + best_mix[1] * et_preds_arr + best_mix[2] * rf_preds_arr

# ---- 6.2 Stacking Meta-Learner ----
print("\n  Training stacking meta-learner...")

# Stack all OOF predictions
stack_train = np.column_stack([oof_preds[name] for name in model_names])
stack_test = np.column_stack([test_preds[name] for name in model_names])

# Add important original features to stacking
important_features = ['geo_demand_mean', 'road_demand_mean', 'hour_demand_mean',
                      'geo_hour_demand_mean', 'geo_road_demand_mean', 'highway_flag',
                      'RoadType_encoded', 'NumberofLanes']
important_idx = [feature_cols.index(f) for f in important_features if f in feature_cols]
stack_train = np.column_stack([stack_train, X_train[:, important_idx]])
stack_test = np.column_stack([stack_test, X_test[:, important_idx]])

meta_oof = np.zeros(len(X_train))
meta_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(stack_train)):
    X_tr, X_val = stack_train[train_idx], stack_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    meta = Ridge(alpha=1.0)
    meta.fit(X_tr, y_tr)

    meta_oof[val_idx] = meta.predict(X_val)
    meta_preds += meta.predict(stack_test) / N_FOLDS

meta_score = r2_score(y_train, meta_oof)
print(f"  Stacking CV R² = {meta_score:.6f} | Score = {max(0, 100*meta_score):.4f}")

# ---- 6.3 Choose Best Final Predictions ----
print("\n  Comparing strategies:")
print(f"    Simple Average:   {max(0, 100*simple_score):.4f}")
print(f"    Weighted Blend:   {max(0, 100*best_final_score):.4f}")
print(f"    Stacking:         {max(0, 100*meta_score):.4f}")

# Pick the best
scores = {
    'blend': (best_final_score, final_blend_test),
    'stacking': (meta_score, meta_preds),
    'simple': (simple_score, simple_avg_test),
}
best_method = max(scores, key=lambda k: scores[k][0])
final_score = scores[best_method][0]
submission_preds = scores[best_method][1]

print(f"\n  → Using: {best_method.upper()} (Score: {max(0, 100*final_score):.4f})")

# Clip to valid range
submission_preds = np.clip(submission_preds, 0, 1)

# =============================================================================
# 7. GENERATE SUBMISSION
# =============================================================================
print("\n[6/6] Generating submission...")

submission = pd.DataFrame({
    'Index': test_index,
    'demand': submission_preds
})

submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)

print(f"  Saved: {submission_path}")
print(f"  Shape: {submission.shape}")
print(f"  Head:\n{submission.head()}")
print(f"\n  Demand stats:")
print(f"    Min:  {submission['demand'].min():.6f}")
print(f"    Max:  {submission['demand'].max():.6f}")
print(f"    Mean: {submission['demand'].mean():.6f}")
print(f"    Std:  {submission['demand'].std():.6f}")

# =============================================================================
# SUMMARY
# =============================================================================
elapsed = time.time() - start_time
print(f"\n{'=' * 70}")
print("MODEL PERFORMANCE SUMMARY")
print(f"{'=' * 70}")
print(f"  {'Model':<20} {'CV R²':<12} {'Score':<10}")
print(f"  {'-'*42}")
print(f"  {'XGBoost v1':<20} {xgb_cv:<12.6f} {max(0,100*xgb_cv):<10.4f}")
print(f"  {'XGBoost v2':<20} {xgb2_cv:<12.6f} {max(0,100*xgb2_cv):<10.4f}")
print(f"  {'LightGBM v1':<20} {lgb_cv:<12.6f} {max(0,100*lgb_cv):<10.4f}")
print(f"  {'LightGBM v2':<20} {lgb2_cv:<12.6f} {max(0,100*lgb2_cv):<10.4f}")
print(f"  {'CatBoost v1':<20} {cat_cv:<12.6f} {max(0,100*cat_cv):<10.4f}")
print(f"  {'CatBoost v2':<20} {cat2_cv:<12.6f} {max(0,100*cat2_cv):<10.4f}")
print(f"  {'ExtraTrees':<20} {et_cv:<12.6f} {max(0,100*et_cv):<10.4f}")
print(f"  {'RandomForest':<20} {rf_cv:<12.6f} {max(0,100*rf_cv):<10.4f}")
print(f"  {'-'*42}")
print(f"  {'FINAL (' + best_method + ')':<20} {final_score:<12.6f} {max(0,100*final_score):<10.4f}")
print(f"\n  Completed in {elapsed/60:.1f} minutes")
print(f"{'=' * 70}")
