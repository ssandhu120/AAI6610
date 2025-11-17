# app.py: FastAPI service for credit default risk prediction
# Exposes both a JSON API and a tiny HTML form UI.

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd

from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

SEED = 20

# The 23 features used by the model
FEATURES = [
    "LIMIT_BAL",   # X1
    "SEX",         # X2
    "EDUCATION",   # X3
    "MARRIAGE",    # X4
    "AGE",         # X5
    "PAY_0",       # X6
    "PAY_2",       # X7
    "PAY_3",       # X8
    "PAY_4",       # X9
    "PAY_5",       # X10
    "PAY_6",       # X11
    "BILL_AMT1",   # X12
    "BILL_AMT2",   # X13
    "BILL_AMT3",   # X14
    "BILL_AMT4",   # X15
    "BILL_AMT5",   # X16
    "BILL_AMT6",   # X17
    "PAY_AMT1",    # X18
    "PAY_AMT2",    # X19
    "PAY_AMT3",    # X20
    "PAY_AMT4",    # X21
    "PAY_AMT5",    # X22
    "PAY_AMT6",    # X23
]


def load_credit_default_data():
    """
    Load the Yeh (2009) 'Default of Credit Card Clients' dataset from UCI.
    Returns X, y with human-readable feature names.
    """
    dataset = fetch_ucirepo(id=350)  # Default of Credit Card Clients dataset

    X_raw = dataset.data.features.copy()
    y = dataset.data.targets.squeeze().astype(int)

    # Drop ID column if present
    X = X_raw.copy()
    if "ID" in X.columns:
        X = X.drop(columns=["ID"])

    # Map to our FEATURE names if shape matches (23 features)
    if X.shape[1] == len(FEATURES):
        X.columns = FEATURES
    else:
        # Fallback: keep original names (but we expect 23)
        print(f"Warning: expected {len(FEATURES)} features, got {X.shape[1]}")

    return X, y


def build_pipeline(X):
    """
    Build the preprocessing + LightGBM classifier pipeline.
    All features are treated as numeric and standardized.
    """
    num_cols = list(X.columns)

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

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
    """
    Train the tuned LightGBM pipeline once at startup.
    """
    print("Loading credit default dataset from UCI (ucimlrepo)...")
    X, y = load_credit_default_data()

    print("Building pipeline...")
    pipe = build_pipeline(X)

    print("Training tuned LightGBM model...")
    pipe.fit(X, y)
    print("Training complete.")

    return pipe


# Train at import time (Space startup)
print("=== Starting up model service ===")
model = train_model()

# ---------- FastAPI app and templates ----------

app = FastAPI(
    title="Credit Default Risk API",
    description="Predict probability of default next month for credit card clients.",
    version="1.0.0",
)

templates = Jinja2Templates(directory="templates")


class CreditFeatures(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: CreditFeatures):
    """
    JSON API: accepts all 23 features and returns class + probability.
    """
    data = pd.DataFrame([[getattr(features, col) for col in FEATURES]], columns=FEATURES)

    proba_default = float(model.predict_proba(data)[0, 1])
    pred_class = int(model.predict(data)[0])

    return {
        "predicted_class": int(pred_class),
        "predicted_label": "Will DEFAULT" if pred_class == 1 else "Will NOT default",
        "probability_of_default": round(proba_default, 3),
    }


# ---------- Tiny HTML form UI ----------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """
    Render a simple HTML form to collect feature inputs.
    """
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": None,
            "probability": None,
        },
    )


@app.post("/predict-form", response_class=HTMLResponse)
def predict_form(
    request: Request,
    LIMIT_BAL: float = Form(...),
    SEX: int = Form(...),
    EDUCATION: int = Form(...),
    MARRIAGE: int = Form(...),
    AGE: int = Form(...),
    PAY_0: int = Form(...),
    PAY_2: int = Form(...),
    PAY_3: int = Form(...),
    PAY_4: int = Form(...),
    PAY_5: int = Form(...),
    PAY_6: int = Form(...),
    BILL_AMT1: float = Form(...),
    BILL_AMT2: float = Form(...),
    BILL_AMT3: float = Form(...),
    BILL_AMT4: float = Form(...),
    BILL_AMT5: float = Form(...),
    BILL_AMT6: float = Form(...),
    PAY_AMT1: float = Form(...),
    PAY_AMT2: float = Form(...),
    PAY_AMT3: float = Form(...),
    PAY_AMT4: float = Form(...),
    PAY_AMT5: float = Form(...),
    PAY_AMT6: float = Form(...),
):
    """
    Handle form submission, run prediction, and re-render the form with results.
    """
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

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": label,
            "probability": round(proba_default, 3),
        },
    )
