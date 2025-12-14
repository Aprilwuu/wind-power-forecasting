from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from src.data.load import load_raw_data
from src.data.make_dataset import make_dataset
from src.data.split import train_valid_split
from src.features.build_features import make_ml_features, select_ml_xy
from src.models.lgbm import create_lgbm_model
from src.metrics import evaluate_rmse

logger = logging.getLogger(__name__)
    
def run_lgbm_pipeline(data_path:str | Path = "data/processed/gefcom_wind_all_zones.csv"):
    """End-to-end LightGBM training pipeline."""
    logger.info("Loading raw data...")
    df = load_raw_data(data_path)

    logger.info("Cleaning and imputing...")
    df = make_dataset(df)

    logger.info("Building ML features...")
    df = make_ml_features(df)

    logger.info("Splitting data (train/val/test)...")
    df_train, df_val, df_test = train_valid_split(df)

    logger.info("Preparing features and labels...")
    X_train, y_train = select_ml_xy(df_train)
    X_val, y_val = select_ml_xy(df_val)

    logger.info("Training LightGBM model...")
    model = create_lgbm_model()
    model.fit(X_train, y_train, X_val, y_val)

    logger.info("Evaluating...")
    y_pred = model.predict(X_val)
    rmse = evaluate_rmse(y_val, y_pred)
    print(f"Validation RMSE: {rmse:.4f}")

    return model, rmse

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_lgbm_pipeline()
