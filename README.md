---
title: Credit Default Risk – AAI6610
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: "5.0.0"
app_file: UI_app.py
pinned: false
---

# Credit Default Risk – LightGBM (AAI6610)

This Space hosts a tuned LightGBM model trained on the credit card default dataset.
The app takes a single user's features as input and returns:

- Predicted default / no default
- Probability of default

Deployed as part of AAI6610 Assignment 10 using:

- Hugging Face Spaces (CPU Basic)
- Gradio for the UI
- GitHub Actions for CI/CD (auto-sync from repo → Space)
