"""
Utility functions: logging setup, timing, submission generation.
"""

import logging
import os
import sys
import time

import numpy as np
import pandas as pd


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure structured logging to stdout."""
    logger = logging.getLogger()
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self):
        self.start = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start

    @property
    def minutes(self) -> float:
        return self.elapsed / 60.0


def generate_submission(
    test_index: np.ndarray,
    predictions: np.ndarray,
    output_path: str,
) -> pd.DataFrame:
    """
    Generate and save submission CSV with validation.

    Args:
        test_index: Array of test sample indices.
        predictions: Predicted demand values.
        output_path: Path to save submission CSV.

    Returns:
        Submission DataFrame.
    """
    logger = logging.getLogger(__name__)

    if len(test_index) != len(predictions):
        raise ValueError(
            f"Index length ({len(test_index)}) != predictions length ({len(predictions)})"
        )

    submission = pd.DataFrame({
        "Index": test_index,
        "demand": predictions,
    })

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission.to_csv(output_path, index=False)

    logger.info(f"  Saved: {output_path}")
    logger.info(f"  Shape: {submission.shape}")
    logger.info(f"  Head:\n{submission.head()}")
    logger.info(f"\n  Demand stats:")
    logger.info(f"    Min:  {submission['demand'].min():.6f}")
    logger.info(f"    Max:  {submission['demand'].max():.6f}")
    logger.info(f"    Mean: {submission['demand'].mean():.6f}")
    logger.info(f"    Std:  {submission['demand'].std():.6f}")

    return submission


def print_summary(results, final_score: float, best_method: str, elapsed_min: float):
    """Print final model performance summary table."""
    logger = logging.getLogger(__name__)

    logger.info(f"\n{'=' * 70}")
    logger.info("MODEL PERFORMANCE SUMMARY")
    logger.info(f"{'=' * 70}")
    logger.info(f"  {'Model':<20} {'CV R²':<12} {'Score':<10}")
    logger.info(f"  {'-' * 42}")
    for r in results:
        logger.info(f"  {r.name:<20} {r.cv_score:<12.6f} {r.score_100:<10.4f}")
    logger.info(f"  {'-' * 42}")
    logger.info(
        f"  {'FINAL (' + best_method + ')':<20} {final_score:<12.6f} "
        f"{max(0, 100 * final_score):<10.4f}"
    )
    logger.info(f"\n  Completed in {elapsed_min:.1f} minutes")
    logger.info(f"{'=' * 70}")
