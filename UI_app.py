# app.py
import joblib
import pandas as pd
import gradio as gr

# Load the trained pipeline (preprocessor + tuned LightGBM)
model = joblib.load("model_lgbm_tuned.pkl")

# Feature names in the same order as in the training data
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
    gr.Number(label="LIMIT_BAL (NT dollar credit limit)"),
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
    description=(
        "Enter a single customer's features from the credit-card dataset "
        "to estimate probability of default next month."
    ),
)

if __name__ == "__main__":
    demo.launch()
