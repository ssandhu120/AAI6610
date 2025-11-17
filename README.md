---
title: Credit Default API
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Credit Default Prediction API (FastAPI + Docker)

This Space hosts a simple FastAPI service that serves predictions for the
“Default of Credit Card Clients” dataset. The model is trained at build time using
a scikit-learn pipeline and a tuned LightGBM classifier.  
There is **no UI**; interaction is done through HTTP requests.

## Endpoints

### `GET /health`
Returns API health status.

### `POST /predict`
Send a JSON body containing the required feature values to receive:
- Predicted class (0/1)
- Probability of default
- Explanation label

Example:

```json
{
  "LIMIT_BAL": 20000,
  "SEX": 2,
  "EDUCATION": 2,
  "MARRIAGE": 1,
  "AGE": 35,
  "PAY_0": 0,
  "PAY_2": 0,
  "PAY_3": 0,
  "PAY_4": 0,
  "PAY_5": 0,
  "PAY_6": 0,
  "BILL_AMT1": 3913,
  "BILL_AMT2": 3102,
  "BILL_AMT3": 689,
  "BILL_AMT4": 0,
  "BILL_AMT5": 0,
  "BILL_AMT6": 0,
  "PAY_AMT1": 0,
  "PAY_AMT2": 689,
  "PAY_AMT3": 0,
  "PAY_AMT4": 0,
  "PAY_AMT5": 0,
  "PAY_AMT6": 0
}
