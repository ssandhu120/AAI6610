"""Script to train and save a tuned LightGBM model pipeline for deployment."""

import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo

SEED = 20
TEST_SIZE = 0.20

rng = np.random.default_rng(SEED)
random.seed(SEED)
np.random.seed(SEED)

OUT_DIR = Path("artifacts_deploy")
OUT_DIR.mkdir(exist_ok=True)


def load_credit_default_data():
    """
    Load the default of credit card clients dataset from local file.
    """
    # Load from local Excel file
    df = pd.read_excel("default of credit card clients.xls", header=1)
    
    # The target is "default payment next month"
    y = df["default payment next month"].astype(int)
    X = df.drop(columns=["ID", "default payment next month"], errors="ignore")
    
    return X, y


def split_features(X: pd.DataFrame):
    """Match your original feature split: categorical vs numeric."""
    cat_cols = [c for c in ["SEX", "EDUCATION", "MARRIAGE"] if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return num_cols, cat_cols


def build_preprocessor(X: pd.DataFrame):
    num_cols, cat_cols = split_features(X)
    numeric = Pipeline([("scaler", StandardScaler())])
    categorical = Pipeline(
        [("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return pre


def main():
    print("Loading data...")
    X, y = load_credit_default_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )

    pre = build_preprocessor(X_train)

    # Tuned LightGBM hyperparameters from your Module 8 work
    lgbm_tuned = LGBMClassifier(
        random_state=SEED,
        n_estimators=1150,
        learning_rate=0.0077,
        num_leaves=45,
        min_child_samples=80,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=4.64,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("pre", pre),
            ("clf", lgbm_tuned),
        ]
    )

    print("Training tuned LightGBM pipeline...")
    pipe.fit(X_train, y_train)

    # Save the full pipeline (preprocessing + model)
    out_path_artifacts = OUT_DIR / "model_lgbm_tuned.pkl"
    joblib.dump(pipe, out_path_artifacts)
    print(f"Saved tuned LightGBM pipeline to: {out_path_artifacts}")

    # Also save at repo root for Hugging Face Space
    root_path = Path("model_lgbm_tuned.pkl")
    joblib.dump(pipe, root_path)
    print(f"Saved model_lgbm_tuned.pkl at project root: {root_path.resolve()}")


if __name__ == "__main__":
    main()
