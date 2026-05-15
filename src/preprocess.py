"""
preprocess.py
-------------
Handles all data loading, cleaning, encoding, feature selection,
and scaling for the IPL Score Prediction project.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


# Columns with text that need encoding
CATEGORICAL_COLS = ['bat_team', 'bowl_team', 'venue', 'batsman', 'bowler']

# Columns removed because they leak info or are highly correlated
DROP_COLS = ['date', 'mid', 'non_striker', 'runs_last_5', 'wickets_last_5']

# Final features used for training
FEATURE_COLS = [
    'bat_team', 'bowl_team', 'venue',
    'runs', 'wickets', 'overs',
    'striker', 'batsman', 'bowler'
]

TARGET_COL = 'total'


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw IPL CSV and return a DataFrame."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'.\n"
            "Please download ipl_dataset.csv from Kaggle and place it in data/"
        )
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def encode_categoricals(df: pd.DataFrame):
    """
    Label-encode all categorical columns.
    Returns the encoded DataFrame and a dict of LabelEncoder objects.
    """
    df_enc = df.copy()
    encoders = {}

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le
        print(f"[INFO] Encoded '{col}' — {len(le.classes_)} unique values")

    return df_enc, encoders


def select_features(df_enc: pd.DataFrame):
    """
    Drop identifier/correlated columns and return X, y.
    """
    X = df_enc[FEATURE_COLS].copy()
    y = df_enc[TARGET_COL].copy()
    print(f"[INFO] Feature matrix shape: {X.shape}")
    return X, y


def split_and_scale(X, y, test_size=0.3, random_state=42):
    """
    Split into train/test and apply Min-Max scaling.
    Scaler is fit on train only to prevent data leakage.
    Returns scaled arrays and the fitted scaler.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"[INFO] Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def save_artifacts(encoders: dict, scaler: MinMaxScaler, out_dir='models/'):
    """Save encoders and scaler to disk for reuse during inference."""
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(encoders, os.path.join(out_dir, 'label_encoders.pkl'))
    joblib.dump(scaler,   os.path.join(out_dir, 'scaler.pkl'))
    print(f"[INFO] Saved encoders and scaler to '{out_dir}'")


def load_artifacts(model_dir='models/'):
    """Load saved encoders and scaler from disk."""
    encoders = joblib.load(os.path.join(model_dir, 'label_encoders.pkl'))
    scaler   = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    return encoders, scaler


def run_pipeline(data_path='data/ipl_dataset.csv'):
    """
    Full preprocessing pipeline.
    Returns everything needed for training.
    """
    df = load_data(data_path)
    df_enc, encoders = encode_categoricals(df)
    X, y = select_features(df_enc)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    save_artifacts(encoders, scaler)

    return X_train, X_test, y_train, y_test, encoders, scaler


if __name__ == '__main__':
    run_pipeline()
