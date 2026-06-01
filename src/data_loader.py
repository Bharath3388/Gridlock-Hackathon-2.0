"""
Data loading and validation module.
Handles reading CSVs with schema validation and basic integrity checks.
"""

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_TRAIN_COLS = {
    "Index", "geohash", "day", "timestamp", "demand",
    "RoadType", "NumberofLanes", "LargeVehicles", "Landmarks",
    "Temperature", "Weather",
}

EXPECTED_TEST_COLS = {
    "Index", "geohash", "day", "timestamp",
    "RoadType", "NumberofLanes", "LargeVehicles", "Landmarks",
    "Temperature", "Weather",
}


def validate_dataframe(df: pd.DataFrame, expected_cols: set, name: str) -> None:
    """Validate that a DataFrame has expected columns and no corrupt data."""
    actual_cols = set(df.columns)
    missing = expected_cols - actual_cols
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
    if df.empty:
        raise ValueError(f"{name} is empty")
    if df["Index"].duplicated().any():
        logger.warning(f"{name} contains duplicate Index values")


def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and validate train/test CSV files.

    Returns:
        Tuple of (train_df, test_df) with validated schemas.
    """
    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not os.path.isfile(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    logger.info("Loading training data...")
    train = pd.read_csv(train_path)
    validate_dataframe(train, EXPECTED_TRAIN_COLS, "train.csv")

    logger.info("Loading test data...")
    test = pd.read_csv(test_path)
    validate_dataframe(test, EXPECTED_TEST_COLS, "test.csv")

    # Validate target column
    if train["demand"].isnull().all():
        raise ValueError("Target column 'demand' is entirely null")
    if (train["demand"] < 0).any():
        logger.warning("Negative demand values detected — clipping to 0")
        train["demand"] = train["demand"].clip(lower=0)

    logger.info(f"  Train shape: {train.shape}")
    logger.info(f"  Test shape:  {test.shape}")

    return train, test


def combine_datasets(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Combine train and test for consistent feature engineering."""
    test_index = test["Index"].values.copy()

    train = train.copy()
    test = test.copy()
    train["is_train"] = 1
    test["is_train"] = 0
    test["demand"] = np.nan

    combined = pd.concat([train, test], axis=0, ignore_index=True)
    logger.info(f"  Combined shape: {combined.shape}")

    return combined, test_index
