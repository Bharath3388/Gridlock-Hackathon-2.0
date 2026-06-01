"""
Traffic Demand Prediction — Gridlock Hackathon 2.0
===================================================
Main entry point. Orchestrates the full ML pipeline:
  1. Load & validate data
  2. Feature engineering
  3. Model training (8 models × 5-fold CV)
  4. Ensemble (weighted blend + stacking)
  5. Generate submission

Usage:
    python run.py
"""

import logging
import warnings

import numpy as np

from config.settings import (
    FEATURE_COLS, OUTPUT_DIR, STACKING_FEATURES,
    SUBMISSION_FILE, TEST_FILE, TRAIN_FILE,
)
from src.data_loader import combine_datasets, load_data
from src.ensemble import build_ensemble
from src.features import run_feature_pipeline
from src.models import train_all_models
from src.utils import Timer, generate_submission, print_summary, setup_logging

warnings.filterwarnings("ignore")


def main():
    setup_logging(logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("TRAFFIC DEMAND PREDICTION — GRIDLOCK HACKATHON 2.0")
    logger.info("=" * 70)

    with Timer() as timer:
        # Step 1: Load data
        logger.info("\n[1/6] Loading data...")
        train, test = load_data(TRAIN_FILE, TEST_FILE)
        combined, test_index = combine_datasets(train, test)

        # Step 2: Feature engineering
        combined = run_feature_pipeline(combined)

        # Step 3: Prepare matrices
        logger.info("\n[3/6] Preparing training and test data...")
        feature_cols = [f for f in FEATURE_COLS if f in combined.columns]
        missing_feats = set(FEATURE_COLS) - set(feature_cols)
        if missing_feats:
            logger.warning(f"  Missing features (skipped): {missing_feats}")

        train_processed = combined[combined["is_train"] == 1]
        test_processed = combined[combined["is_train"] == 0]

        X_train = train_processed[feature_cols].values.astype(np.float32)
        y_train = train_processed["demand"].values.astype(np.float32)
        X_test = test_processed[feature_cols].values.astype(np.float32)

        logger.info(f"  X_train shape: {X_train.shape}")
        logger.info(f"  X_test shape:  {X_test.shape}")
        logger.info(f"  Features: {len(feature_cols)}")

        # Step 4: Train models
        results = train_all_models(X_train, y_train, X_test)

        # Step 5: Ensemble
        predictions, final_score, best_method = build_ensemble(
            results, X_train, y_train, X_test, feature_cols, STACKING_FEATURES,
        )

        # Step 6: Submission
        logger.info("\n[6/6] Generating submission...")
        generate_submission(test_index, predictions, SUBMISSION_FILE)

    print_summary(results, final_score, best_method, timer.minutes)


if __name__ == "__main__":
    main()
