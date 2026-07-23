# Heart Failure Risk Prediction & False Alarm Reduction

[![CI](https://github.com/BioCode67/hf-risk-prediction-sw/actions/workflows/ci.yml/badge.svg)](https://github.com/BioCode67/hf-risk-prediction-sw/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end machine learning and FastAPI backend pipeline for **acute heart
failure risk prediction**, optimized to **reduce false alarms** in clinical
monitoring settings. The project pairs a cost-sensitive gradient-boosted model
with **SHAP** explainability for clinical transparency and an **OMOP CDM v5.4**
transformation layer for interoperability with standardized health-data
platforms.

---

## Highlights

- **Imbalance-aware modeling** — LightGBM & XGBoost with `scale_pos_weight`
  cost-sensitive weighting to suppress false alarms.
- **AUPRC-driven tuning** — [Optuna](https://optuna.org/) hyperparameter search
  targeting the area under the precision–recall curve.
- **Clinical transparency** — SHAP `TreeExplainer` surfaces the top 3 risk
  factors behind every prediction.
- **Standards-based interoperability** — `to_omop_cdm()` maps patient features
  to OMOP CDM v5.4 `Person` and `Measurement` tables.
- **Production-style API** — FastAPI endpoints for real-time scoring and OMOP
  conversion, with automatic OpenAPI docs.

---

## Project Structure

```
heart-failure-risk-prediction/
├── src/
│   ├── data_loader.py      # Load, preprocess, split, OMOP CDM mapping
│   ├── train.py            # LightGBM/XGBoost + Optuna (AUPRC) training
│   ├── explainability.py   # SHAP TreeExplainer (global + per-patient)
│   └── main.py             # FastAPI server (/predict, /convert-omop)
├── tests/
│   └── test_pipeline.py    # pytest unit & integration tests
├── data/                   # Datasets (git-ignored)
├── models/                 # Trained artifacts (git-ignored)
├── requirements.txt
├── LICENSE                 # MIT
└── README.md
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Place the source archives in `data/` (they are excluded from version control):

- `Heart Failure Clinical Records Dataset.zip` → `heart_failure_clinical_records_dataset.csv`
- `Cardiovascular Disease dataset.zip` → `cardio_train.csv`

The loader safely unzips archives on first use.

---

## Usage

### 1. Inspect & preprocess data

```bash
python src/data_loader.py
```

Standardizes continuous lab features, engineers derived signals
(`creatinine_ef_ratio`, `age_group`), and performs a **stratified**
train/test split (`test_size=0.2`, `random_state=42`).

### 2. Train the model

```bash
python src/train.py
```

Runs Optuna tuning (maximizing AUPRC), trains a cost-sensitive LightGBM plus an
XGBoost baseline, and saves the best artifact to `models/best_model.pkl`
alongside `models/training_metrics.json`.

### 3. Generate SHAP explanations

```bash
python src/explainability.py
```

Prints global top-5 feature importances and a sample patient's top-3 risk
factors, and writes `models/shap_summary.png`.

### 4. Serve the API

```bash
uvicorn main:app --app-dir src --reload
# or: python src/main.py
```

Interactive docs at **http://localhost:8000/docs**.

---

## API Reference

| Method | Endpoint         | Description                                             |
| ------ | ---------------- | ------------------------------------------------------- |
| `GET`  | `/`              | Service metadata                                        |
| `GET`  | `/health`        | Readiness + model-loaded status                         |
| `POST` | `/predict`       | Risk probability, label, and top-3 SHAP risk factors    |
| `POST` | `/convert-omop`  | Transform patient records into OMOP CDM v5.4 tables      |

### Example — risk prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 65, "anaemia": 0, "creatinine_phosphokinase": 146,
    "diabetes": 0, "ejection_fraction": 20, "high_blood_pressure": 1,
    "platelets": 162000, "serum_creatinine": 1.3, "serum_sodium": 129,
    "sex": 1, "smoking": 1
  }'
```

```json
{
  "risk_probability": 0.82,
  "risk_label": 1,
  "threshold": 0.5,
  "top_risk_factors": [
    {"feature": "ejection_fraction", "shap_value": 0.41, "feature_value": -1.6},
    {"feature": "serum_creatinine", "shap_value": 0.29, "feature_value": 1.1},
    {"feature": "age", "shap_value": 0.18, "feature_value": 0.7}
  ]
}
```

---

## Testing

```bash
pytest -q
```

Covers data loading, preprocessing, stratified splitting, OMOP mapping,
training utilities, SHAP explanations, and all FastAPI endpoints.

---

## Modeling Notes

- **Why AUPRC?** With rare acute events, precision–recall better reflects
  alarm quality than ROC-AUC.
- **False-alarm reduction.** `scale_pos_weight` and a reported
  *sensitivity @ 95% specificity* metric make the precision trade-off explicit.
- **Explainability.** Per-patient SHAP values keep predictions auditable for
  clinical review.

> **Disclaimer:** Research and educational software only. Not a medical device
> and not intended for clinical decision-making.

---

## License

Released under the [MIT License](LICENSE).
