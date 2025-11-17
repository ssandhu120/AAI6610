# UI_app.py
import gradio as gr
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier

SEED = 20

# Features from the UCI "Default of Credit Card Clients" dataset
FEATURES = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]


def load_credit_default_data():
    """
    Load the Yeh & Lien (2009) credit default dataset from OpenML.
    This is the same as the UCI 'Default of Credit Card Clients' dataset.
    """
    ds = fetch_openml(data_id=42477, as_frame=True)
    df = ds.frame.copy()

    y = df["default.payment.next.month"].astype(int)
    X = df.drop(columns=["ID", "default.payment.next.month"], errors="ignore")
    return X, y


def build_pipeline(X):
    """Build the preprocessing + tuned LightGBM pipeline."""
    # Categorical vs numeric split (matches your earlier work)
    cat_cols = [c for c in ["SEX", "EDUCATION", "MARRIAGE"] if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    # Tuned LightGBM hyperparameters from your Module 8 model
    clf = LGBMClassifier(
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
            ("clf", clf),
        ]
    )
    return pipe


def train_model():
    """Train the tuned LightGBM pipeline once at startup."""
    print("Loading credit default dataset from OpenML...")
    X, y = load_credit_default_data()

    print("Building preprocessing + LightGBM pipeline...")
    pipe = build_pipeline(X)

    print("Training tuned LightGBM model (this may take a moment)...")
    pipe.fit(X, y)
    print("Training complete.")

    return pipe


# Train model once when the Space starts
model = train_model()


def predict_default(
    LIMIT_BAL,
    SEX,
    EDUCATION,
    MARRIAGE,
    AGE,
    PAY_0,
    PAY_2,
    PAY_3,
    PAY_4,
    PAY_5,
    PAY_6,
    BILL_AMT1,
    BILL_AMT2,
    BILL_AMT3,
    BILL_AMT4,
    BILL_AMT5,
    BILL_AMT6,
    PAY_AMT1,
    PAY_AMT2,
    PAY_AMT3,
    PAY_AMT4,
    PAY_AMT5,
    PAY_AMT6,
):
    values = [
        LIMIT_BAL,
        SEX,
        EDUCATION,
        MARRIAGE,
        AGE,
        PAY_0,
        PAY_2,
        PAY_3,
        PAY_4,
        PAY_5,
        PAY_6,
        BILL_AMT1,
        BILL_AMT2,
        BILL_AMT3,
        BILL_AMT4,
        BILL_AMT5,
        BILL_AMT6,
        PAY_AMT1,
        PAY_AMT2,
        PAY_AMT3,
        PAY_AMT4,
        PAY_AMT5,
        PAY_AMT6,
    ]

    df = pd.DataFrame([values], columns=FEATURES)
    proba_default = float(model.predict_proba(df)[0, 1])
    pred_class = int(model.predict(df)[0])

    label = "Will DEFAULT" if pred_class == 1 else "Will NOT default"

    return {
        "Predicted class": label,
        "Probability of default": round(proba_default, 3),
    }


inputs = [
    gr.Number(label="LIMIT_BAL (NT credit limit)"),
    gr.Number(label="SEX (1 = male, 2 = female)"),
    gr.Number(label="EDUCATION (1–4 categories)"),
    gr.Number(label="MARRIAGE (1 = married, 2 = single, 3 = others)"),
    gr.Number(label="AGE"),
    gr.Number(label="PAY_0 (Sep repayment status)"),
    gr.Number(label="PAY_2 (Aug repayment status)"),
    gr.Number(label="PAY_3 (Jul repayment status)"),
    gr.Number(label="PAY_4 (Jun repayment status)"),
    gr.Number(label="PAY_5 (May repayment status)"),
    gr.Number(label="PAY_6 (Apr repayment status)"),
    gr.Number(label="BILL_AMT1 (Sep bill amount)"),
    gr.Number(label="BILL_AMT2 (Aug bill amount)"),
    gr.Number(label="BILL_AMT3 (Jul bill amount)"),
    gr.Number(label="BILL_AMT4 (Jun bill amount)"),
    gr.Number(label="BILL_AMT5 (May bill amount)"),
    gr.Number(label="BILL_AMT6 (Apr bill amount)"),
    gr.Number(label="PAY_AMT1 (Sep payment)"),
    gr.Number(label="PAY_AMT2 (Aug payment)"),
    gr.Number(label="PAY_AMT3 (Jul payment)"),
    gr.Number(label="PAY_AMT4 (Jun payment)"),
    gr.Number(label="PAY_AMT5 (May payment)"),
    gr.Number(label="PAY_AMT6 (Apr payment)"),
]

demo = gr.Interface(
    fn=predict_default,
    inputs=inputs,
    outputs=gr.JSON(label="Prediction"),
    title="Credit Default Risk – Tuned LightGBM",
    description="Enter features from the credit-card default dataset to estimate probability of default next month.",
)

if __name__ == "__main__":
    demo.launch()
