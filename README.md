# NEXUSLEND Churn Prediction Deployment Pipeline

This package provides a deployment wrapper around the NEXUSLEND churn prediction model.

## 🚀 Quick Start

1. Install the required Python packages:

```bash
pip install pandas numpy scikit-learn joblib
```

2. Run the deployment pipeline:

```bash
python deployment_pipeline.py
```

3. The deployment artifact will be created at:

```text
models/inference_pipeline.pkl
```

## ⚠️ Critical Ethical Constraints

Do NOT target high-strain customers with predatory offers

- Human review required for debt_to_income > 0.6.
- Never deny service based solely on model output.
- Monthly fairness audits are mandatory.
- High-risk predictions must be reviewed by a human.
- Model predictions must not be treated as automatic financial decisions.

## 🔍 Monitoring Commands

Check the deployment artifact:

```bash
python -c "import joblib; x=joblib.load('models/inference_pipeline.pkl'); print(x['metadata'])"
```

Run the deployment script again to regenerate the inference artifact:

```bash
python deployment_pipeline.py
```

Retrain if township recall drops >10%

## Circuit Breaker

The circuit breaker activates when the prediction error rate exceeds 5%.
When activated, the pipeline returns safe default predictions instead of continuing normal inference.

## Input Validation

- Validates that the input is a pandas DataFrame.
- Checks required input fields.
- Converts numeric fields safely.
- Handles unknown region categories.
- Rejects missing model features.

## Model Metadata

- Version: 1.2
- Training date: 2026-09-13
- Ethical constraints: 3
- Source model: models/churn_pipeline.pkl

## Stakeholder Guidance

Per CFO: 'Every 1% churn reduction = R420k saved'

The model is a decision-support tool and must not replace human judgement.
