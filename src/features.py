"""
Feature engineering pipeline.
Transforms raw combined DataFrame into model-ready feature matrix.
"""

import logging

import numpy as np
import pandas as pd
import pygeohash as pgh
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def decode_geohash(combined: pd.DataFrame) -> pd.DataFrame:
    """Decode geohash to latitude/longitude with error handling."""
    logger.info("  → Decoding geohash to lat/lon...")
    geo_cache = {}
    for gh in combined["geohash"].unique():
        try:
            lat, lon = pgh.decode(gh)
            geo_cache[gh] = (float(lat), float(lon))
        except (ValueError, TypeError) as e:
            logger.warning(f"  Failed to decode geohash '{gh}': {e}")
            geo_cache[gh] = (0.0, 0.0)

    combined["latitude"] = combined["geohash"].map(lambda x: geo_cache[x][0])
    combined["longitude"] = combined["geohash"].map(lambda x: geo_cache[x][1])

    combined["geo_prefix3"] = combined["geohash"].str[:3]
    combined["geo_prefix4"] = combined["geohash"].str[:4]
    combined["geo_prefix5"] = combined["geohash"].str[:5]

    return combined


def engineer_temporal_features(combined: pd.DataFrame) -> pd.DataFrame:
    """Create temporal/cyclical features from timestamp and day."""
    logger.info("  → Engineering temporal features...")

    def parse_timestamp(ts: str) -> tuple[int, int]:
        parts = str(ts).split(":")
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0

    hours_minutes = combined["timestamp"].apply(parse_timestamp)
    combined["hour"] = hours_minutes.apply(lambda x: x[0])
    combined["minute"] = hours_minutes.apply(lambda x: x[1])
    combined["minutes_of_day"] = combined["hour"] * 60 + combined["minute"]
    combined["quarter_of_day"] = combined["minutes_of_day"] // 15

    # Cyclical encoding
    combined["hour_sin"] = np.sin(2 * np.pi * combined["hour"] / 24)
    combined["hour_cos"] = np.cos(2 * np.pi * combined["hour"] / 24)
    combined["minute_sin"] = np.sin(2 * np.pi * combined["minutes_of_day"] / 1440)
    combined["minute_cos"] = np.cos(2 * np.pi * combined["minutes_of_day"] / 1440)

    # Time period indicators
    combined["is_morning_rush"] = (
        (combined["hour"] >= 7) & (combined["hour"] <= 9)
    ).astype(int)
    combined["is_evening_rush"] = (
        (combined["hour"] >= 16) & (combined["hour"] <= 19)
    ).astype(int)
    combined["is_rush_hour"] = (
        combined["is_morning_rush"] | combined["is_evening_rush"]
    ).astype(int)
    combined["is_night"] = (
        (combined["hour"] >= 22) | (combined["hour"] <= 5)
    ).astype(int)
    combined["is_midday"] = (
        (combined["hour"] >= 10) & (combined["hour"] <= 15)
    ).astype(int)

    # Day features
    combined["day_of_week"] = combined["day"] % 7
    combined["is_weekend"] = (combined["day_of_week"] >= 5).astype(int)

    return combined


def handle_missing_values(combined: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values using group-based strategies."""
    logger.info("  → Handling missing values...")

    # RoadType
    road_mode_global = combined["RoadType"].mode()[0]
    combined["RoadType_missing"] = combined["RoadType"].isnull().astype(int)
    combined["RoadType"] = combined.groupby("geohash")["RoadType"].transform(
        lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else road_mode_global)
    )
    combined["RoadType"] = combined["RoadType"].fillna(road_mode_global)

    # Temperature
    temp_median_global = combined["Temperature"].median()
    combined["Temperature_missing"] = combined["Temperature"].isnull().astype(int)
    combined["Temperature"] = combined.groupby("geohash")["Temperature"].transform(
        lambda x: x.fillna(x.median())
    )
    combined["Temperature"] = combined["Temperature"].fillna(temp_median_global)

    # Weather
    weather_mode_global = combined["Weather"].mode()[0]
    combined["Weather_missing"] = combined["Weather"].isnull().astype(int)
    combined["Weather"] = combined.groupby("timestamp")["Weather"].transform(
        lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else weather_mode_global)
    )
    combined["Weather"] = combined["Weather"].fillna(weather_mode_global)

    return combined


def encode_categoricals(combined: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features as numeric."""
    logger.info("  → Encoding categorical features...")

    # RoadType ordinal
    road_type_map = {"Residential": 0, "Street": 1, "Highway": 2}
    combined["RoadType_encoded"] = (
        combined["RoadType"].map(road_type_map).fillna(0).astype(int)
    )

    # LargeVehicles binary
    combined["LargeVehicles_encoded"] = (
        combined["LargeVehicles"] == "Allowed"
    ).astype(int)

    # Landmarks binary
    combined["Landmarks_encoded"] = (combined["Landmarks"] == "Yes").astype(int)

    # Weather ordinal + one-hot
    weather_map = {"Sunny": 0, "Rainy": 1, "Foggy": 2, "Snowy": 3}
    combined["Weather_encoded"] = (
        combined["Weather"].map(weather_map).fillna(0).astype(int)
    )
    for w in ["Sunny", "Rainy", "Foggy", "Snowy"]:
        combined[f"weather_{w}"] = (combined["Weather"] == w).astype(int)

    # One-hot for RoadType
    for rt in ["Residential", "Street", "Highway"]:
        combined[f"road_{rt}"] = (combined["RoadType"] == rt).astype(int)

    return combined


def engineer_temperature_features(combined: pd.DataFrame) -> pd.DataFrame:
    """Create temperature-derived features."""
    logger.info("  → Engineering temperature features...")

    combined["temp_bin"] = pd.cut(combined["Temperature"], bins=10, labels=False)
    combined["temp_squared"] = combined["Temperature"] ** 2
    combined["temp_abs"] = combined["Temperature"].abs()
    combined["is_cold"] = (combined["Temperature"] < 5).astype(int)
    combined["is_hot"] = (combined["Temperature"] > 30).astype(int)
    combined["is_moderate"] = (
        (combined["Temperature"] >= 10) & (combined["Temperature"] <= 25)
    ).astype(int)

    return combined


def create_interaction_features(combined: pd.DataFrame) -> pd.DataFrame:
    """Create cross-feature interactions."""
    logger.info("  → Creating interaction features...")

    combined["road_lanes"] = combined["RoadType_encoded"] * 10 + combined["NumberofLanes"]
    combined["highway_flag"] = (
        (combined["RoadType_encoded"] == 2) | (combined["NumberofLanes"] >= 4)
    ).astype(int)
    combined["lane_pressure"] = combined["NumberofLanes"] / (
        combined["RoadType_encoded"] + 1
    )
    combined["road_time"] = (
        combined["RoadType_encoded"] * 10 + (combined["minutes_of_day"] // 60)
    )
    combined["lanes_large"] = (
        combined["NumberofLanes"] * combined["LargeVehicles_encoded"]
    )
    combined["temp_weather"] = combined["Temperature"] * combined["Weather_encoded"]
    combined["hour_road"] = combined["hour"] * 10 + combined["RoadType_encoded"]
    combined["lat_hour"] = combined["latitude"] * combined["hour"]
    combined["lon_hour"] = combined["longitude"] * combined["hour"]

    return combined


def compute_target_encodings(combined: pd.DataFrame) -> pd.DataFrame:
    """
    Compute leak-free target encoding features using ONLY training data.
    This prevents data leakage from test set into feature generation.
    """
    logger.info("  → Computing target encoding features (leak-free)...")

    train_mask = combined["is_train"] == 1
    train_data = combined[train_mask]
    global_mean = train_data["demand"].mean()

    # Geohash-level stats
    geo_stats = train_data.groupby("geohash")["demand"].agg(
        ["mean", "std", "median", "count"]
    )
    geo_stats.columns = [
        "geo_demand_mean", "geo_demand_std", "geo_demand_median", "geo_demand_count",
    ]
    geo_stats["geo_demand_std"] = geo_stats["geo_demand_std"].fillna(0)
    combined = combined.merge(geo_stats, on="geohash", how="left")
    combined["geo_demand_mean"] = combined["geo_demand_mean"].fillna(global_mean)
    combined["geo_demand_std"] = combined["geo_demand_std"].fillna(0)
    combined["geo_demand_median"] = combined["geo_demand_median"].fillna(global_mean)
    combined["geo_demand_count"] = combined["geo_demand_count"].fillna(1)
    combined["geo_demand_cv"] = combined["geo_demand_std"] / (
        combined["geo_demand_mean"] + 1e-8
    )

    # Geohash prefix stats
    for prefix_col in ["geo_prefix4", "geo_prefix5"]:
        prefix_stats = train_data.groupby(prefix_col)["demand"].agg(["mean", "std"])
        prefix_stats.columns = [
            f"{prefix_col}_demand_mean", f"{prefix_col}_demand_std",
        ]
        prefix_stats[f"{prefix_col}_demand_std"] = (
            prefix_stats[f"{prefix_col}_demand_std"].fillna(0)
        )
        combined = combined.merge(prefix_stats, on=prefix_col, how="left")
        combined[f"{prefix_col}_demand_mean"] = (
            combined[f"{prefix_col}_demand_mean"].fillna(global_mean)
        )
        combined[f"{prefix_col}_demand_std"] = (
            combined[f"{prefix_col}_demand_std"].fillna(0)
        )

    # RoadType demand
    road_stats = train_data.groupby("RoadType_encoded")["demand"].agg(["mean", "std"])
    road_stats.columns = ["road_demand_mean", "road_demand_std"]
    combined = combined.merge(road_stats, on="RoadType_encoded", how="left")
    combined["road_demand_mean"] = combined["road_demand_mean"].fillna(global_mean)
    combined["road_demand_std"] = combined["road_demand_std"].fillna(0)

    # Hour demand
    hour_stats = train_data.groupby("hour")["demand"].agg(["mean", "std"])
    hour_stats.columns = ["hour_demand_mean", "hour_demand_std"]
    combined = combined.merge(hour_stats, on="hour", how="left")
    combined["hour_demand_mean"] = combined["hour_demand_mean"].fillna(global_mean)
    combined["hour_demand_std"] = combined["hour_demand_std"].fillna(0)

    # Geohash + hour demand
    geo_hour_stats = train_data.groupby(["geohash", "hour"])["demand"].agg(
        ["mean", "count"]
    )
    geo_hour_stats.columns = ["geo_hour_demand_mean", "geo_hour_count"]
    combined = combined.merge(geo_hour_stats, on=["geohash", "hour"], how="left")
    combined["geo_hour_demand_mean"] = combined["geo_hour_demand_mean"].fillna(
        combined["geo_demand_mean"]
    )
    combined["geo_hour_count"] = combined["geo_hour_count"].fillna(0)

    # RoadType + hour demand
    road_hour_stats = train_data.groupby(["RoadType_encoded", "hour"])["demand"].agg(
        ["mean"]
    )
    road_hour_stats.columns = ["road_hour_demand_mean"]
    combined = combined.merge(
        road_hour_stats, on=["RoadType_encoded", "hour"], how="left"
    )
    combined["road_hour_demand_mean"] = combined["road_hour_demand_mean"].fillna(
        global_mean
    )

    # NumberofLanes demand
    lanes_demand = train_data.groupby("NumberofLanes")["demand"].agg(["mean"])
    lanes_demand.columns = ["lanes_demand_mean"]
    combined = combined.merge(lanes_demand, on="NumberofLanes", how="left")
    combined["lanes_demand_mean"] = combined["lanes_demand_mean"].fillna(global_mean)

    # Geohash + RoadType demand
    geo_road_stats = train_data.groupby(["geohash", "RoadType_encoded"])[
        "demand"
    ].agg(["mean"])
    geo_road_stats.columns = ["geo_road_demand_mean"]
    combined = combined.merge(
        geo_road_stats, on=["geohash", "RoadType_encoded"], how="left"
    )
    combined["geo_road_demand_mean"] = combined["geo_road_demand_mean"].fillna(
        combined["geo_demand_mean"]
    )

    # RoadType + NumberofLanes demand
    road_lane_stats = train_data.groupby(["RoadType_encoded", "NumberofLanes"])[
        "demand"
    ].agg(["mean"])
    road_lane_stats.columns = ["road_lane_demand_mean"]
    combined = combined.merge(
        road_lane_stats, on=["RoadType_encoded", "NumberofLanes"], how="left"
    )
    combined["road_lane_demand_mean"] = combined["road_lane_demand_mean"].fillna(
        global_mean
    )

    # Timestamp demand
    time_stats = train_data.groupby("timestamp")["demand"].agg(["mean"])
    time_stats.columns = ["time_demand_mean"]
    combined = combined.merge(time_stats, on="timestamp", how="left")
    combined["time_demand_mean"] = combined["time_demand_mean"].fillna(global_mean)

    return combined


def create_spatial_features(combined: pd.DataFrame) -> pd.DataFrame:
    """Create spatial features: label encoding, distance, binning."""
    logger.info("  → Creating spatial features...")

    le_geo = LabelEncoder()
    combined["geohash_le"] = le_geo.fit_transform(combined["geohash"])

    le_prefix4 = LabelEncoder()
    combined["geo_prefix4_le"] = le_prefix4.fit_transform(combined["geo_prefix4"])

    le_prefix5 = LabelEncoder()
    combined["geo_prefix5_le"] = le_prefix5.fit_transform(combined["geo_prefix5"])

    lat_mean = combined["latitude"].mean()
    lon_mean = combined["longitude"].mean()
    combined["dist_from_center"] = np.sqrt(
        (combined["latitude"] - lat_mean) ** 2
        + (combined["longitude"] - lon_mean) ** 2
    )

    combined["lat_bin"] = pd.cut(combined["latitude"], bins=20, labels=False)
    combined["lon_bin"] = pd.cut(combined["longitude"], bins=20, labels=False)
    combined["spatial_bin"] = combined["lat_bin"] * 20 + combined["lon_bin"]

    # Temperature anomaly
    geo_temp_mean = combined.groupby("geohash")["Temperature"].transform("mean")
    combined["temp_anomaly"] = combined["Temperature"] - geo_temp_mean

    # Geohash frequency
    geo_freq = combined["geohash"].value_counts().to_dict()
    combined["geohash_freq"] = combined["geohash"].map(geo_freq)

    return combined


def run_feature_pipeline(combined: pd.DataFrame) -> pd.DataFrame:
    """Execute the full feature engineering pipeline in sequence."""
    logger.info("[2/6] Feature Engineering...")

    combined = decode_geohash(combined)
    combined = engineer_temporal_features(combined)
    combined = handle_missing_values(combined)
    combined = encode_categoricals(combined)
    combined = engineer_temperature_features(combined)
    combined = create_interaction_features(combined)
    combined = compute_target_encodings(combined)
    combined = create_spatial_features(combined)

    logger.info(f"  Total features engineered: {combined.shape[1]}")
    return combined
