"""
Configuration settings for the Traffic Demand Prediction pipeline.
All hyperparameters, paths, and constants are centralized here.
"""

import os

# =============================================================================
# PATHS
# =============================================================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "dataset")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")

TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
SUBMISSION_FILE = os.path.join(OUTPUT_DIR, "submission.csv")

# =============================================================================
# CROSS-VALIDATION
# =============================================================================
N_FOLDS = 5
RANDOM_STATE = 42

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
XGB_V1_PARAMS = {
    "n_estimators": 3000,
    "max_depth": 8,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "colsample_bylevel": 0.7,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": 0,
}

XGB_V2_PARAMS = {
    "n_estimators": 4000,
    "max_depth": 10,
    "learning_rate": 0.02,
    "subsample": 0.75,
    "colsample_bytree": 0.65,
    "colsample_bylevel": 0.65,
    "min_child_weight": 3,
    "reg_alpha": 0.05,
    "reg_lambda": 1.5,
    "tree_method": "hist",
    "random_state": 123,
    "n_jobs": -1,
    "verbosity": 0,
}

LGB_V1_PARAMS = {
    "n_estimators": 3000,
    "max_depth": 8,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "num_leaves": 127,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

LGB_V2_PARAMS = {
    "n_estimators": 4000,
    "max_depth": -1,
    "learning_rate": 0.02,
    "subsample": 0.75,
    "colsample_bytree": 0.65,
    "min_child_samples": 10,
    "reg_alpha": 0.05,
    "reg_lambda": 1.5,
    "num_leaves": 255,
    "random_state": 123,
    "n_jobs": -1,
    "verbose": -1,
}

CAT_V1_PARAMS = {
    "iterations": 3000,
    "depth": 8,
    "learning_rate": 0.03,
    "l2_leaf_reg": 3,
    "random_seed": RANDOM_STATE,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 0.5,
    "verbose": 0,
}

CAT_V2_PARAMS = {
    "iterations": 4000,
    "depth": 10,
    "learning_rate": 0.02,
    "l2_leaf_reg": 5,
    "random_seed": 123,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 1.0,
    "verbose": 0,
}

ET_PARAMS = {
    "n_estimators": 500,
    "max_depth": 30,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 25,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# =============================================================================
# STACKING
# =============================================================================
META_LEARNER_ALPHA = 1.0

# =============================================================================
# FEATURE LISTS
# =============================================================================
FEATURE_COLS = [
    # Core features
    "NumberofLanes", "LargeVehicles_encoded", "Landmarks_encoded",
    "Temperature", "RoadType_encoded", "Weather_encoded",
    # Geospatial
    "latitude", "longitude", "geohash_le", "geo_prefix4_le", "geo_prefix5_le",
    "dist_from_center", "lat_bin", "lon_bin", "spatial_bin",
    # Temporal
    "day", "minutes_of_day", "hour", "minute", "quarter_of_day",
    "hour_sin", "hour_cos", "minute_sin", "minute_cos",
    "is_morning_rush", "is_evening_rush", "is_rush_hour",
    "is_night", "is_midday", "day_of_week", "is_weekend",
    # Temperature features
    "Temperature_missing", "temp_bin", "temp_squared", "temp_abs",
    "is_cold", "is_hot", "is_moderate", "temp_anomaly",
    # Missing indicators
    "RoadType_missing", "Weather_missing",
    # One-hot
    "weather_Sunny", "weather_Rainy", "weather_Foggy", "weather_Snowy",
    "road_Residential", "road_Street", "road_Highway",
    # Interaction features
    "road_lanes", "highway_flag", "lane_pressure", "road_time",
    "lanes_large", "temp_weather", "hour_road", "lat_hour", "lon_hour",
    # Target encoding / aggregation
    "geo_demand_mean", "geo_demand_std", "geo_demand_median", "geo_demand_count",
    "geo_demand_cv",
    "geo_prefix4_demand_mean", "geo_prefix4_demand_std",
    "geo_prefix5_demand_mean", "geo_prefix5_demand_std",
    "road_demand_mean", "road_demand_std",
    "hour_demand_mean", "hour_demand_std",
    "geo_hour_demand_mean", "geo_hour_count",
    "road_hour_demand_mean", "lanes_demand_mean",
    "geo_road_demand_mean", "road_lane_demand_mean",
    "time_demand_mean",
    # Extra spatial
    "geohash_freq",
]

STACKING_FEATURES = [
    "geo_demand_mean", "road_demand_mean", "hour_demand_mean",
    "geo_hour_demand_mean", "geo_road_demand_mean", "highway_flag",
    "RoadType_encoded", "NumberofLanes",
]
