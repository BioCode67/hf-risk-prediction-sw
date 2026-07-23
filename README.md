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
│   ├── data_loader.py      # [static] Load, preprocess, split, OMOP CDM mapping
│   ├── train.py            # [static] LightGBM/XGBoost + Optuna (AUPRC) training
│   ├── explainability.py   # [static] SHAP TreeExplainer (global + per-patient)
│   ├── main.py             # [static] FastAPI server (/predict, /convert-omop)
│   ├── vitals_data.py      # [time-series] synthetic + KHTH + MIMIC adapters, windows, split
│   ├── vitals_train.py     # [time-series] XGBoost vs NEWS early-warning + false-alarm metrics
│   ├── vitals_explain.py   # [time-series] SHAP drivers for early-warning windows
│   ├── vitals_report.py    # [time-series] PR-curve / trajectory / lead-time figures
│   ├── vitals_phenotype.py # [time-series] cardiac-arrest phenotype clustering + heatmap
│   ├── mimic_explore.py    # [time-series] explore/model real MIMIC-IV (--scan-arrest/--model)
│   └── omop_explore.py     # [OMOP] explore any OMOP CDM CSV folder (Eunomia/competition)
├── tests/
│   ├── test_pipeline.py    # static heart-failure pipeline tests
│   ├── test_vitals.py      # vital-sign early-warning tests (synthetic, always run)
│   ├── test_mimic.py       # MIMIC-IV explorer/adapter tests (network-free)
│   ├── test_omop.py        # OMOP CDM explorer tests (network-free)
│   └── test_report.py      # visual-report rendering tests
├── docs/
│   ├── competition-strategy.md  # K-Health 공모전 전략 & 제안서 설계
│   └── proposal-draft.md        # 예선 제안서 30장 골격 초안
├── data/                   # Datasets (git-ignored)
├── models/                 # Trained artifacts (git-ignored)
├── requirements.txt
├── LICENSE                 # MIT
└── README.md
```

> This repository hosts **two complementary tracks**. The **static** track
> (`data_loader`/`train`/`explainability`/`main`) predicts heart-failure risk
> from one-shot clinical records. The **time-series** track (`vitals_*`) is the
> cardiac-arrest early-warning system built for the *2026 K-Health 미개방
> 의료데이터 경진대회* (경북대병원 활력징후 데이터) — see
> [`docs/competition-strategy.md`](docs/competition-strategy.md).

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

## Cardiac-Arrest Early Warning (time-series track)

Vital-sign time-series pipeline for **in-hospital cardiac-arrest early warning**,
the differentiator of which is **false-alarm reduction + explainability**, not
raw accuracy. It runs end-to-end on a built-in **synthetic** cohort (no
restricted data required); real `KHTH_PINFO`/`KHTH_VITAL` tables plug in through
`vitals_data.cohort_from_khth`.

```bash
python src/vitals_data.py     # build synthetic cohort → windows → patient-level split
python src/vitals_train.py    # train XGBoost, compare against the NEWS baseline
python src/vitals_explain.py  # SHAP drivers + models/vitals_shap_summary.png
python src/vitals_report.py   # PR-curve, deterioration trajectory, lead-time figures
python src/vitals_phenotype.py # discover cardiac-arrest phenotypes + heatmap
```

**Method.** Hourly vitals (pulse, systolic/diastolic BP, temperature, SpO₂,
respiratory rate) → sliding-window features (mean/std/min/max/last/**slope**/delta
per vital + shock index) → predict an arrest within the next hour. Splitting is
**patient-level** (no patient in both train and test), and missing values are
imputed with train-set medians to avoid leakage.

**Personalized baseline (the differentiator).** `add_personalized_features`
adds, per vital, the window's deviation from *that patient's own* early-stable
baseline. A value that is normal for the ward can be a large personal deviation —
this personalizes the alarm and, on the synthetic demo, lifts AUPRC from ≈0.76 to
≈0.84. It also turns the case-only cohort's within-patient structure into the
method itself.

**Data sources (one pipeline).** The same windowing/labelling runs on three
sources via thin adapters: `generate_synthetic_cohort` (CI/demo, no data needed),
`cohort_from_mimic` (real MIMIC-IV ICU data — supplies the *control* patients the
competition cohort lacks, for honest false-alarm measurement), and
`cohort_from_khth` (the 안심존 competition tables, labelled by the exact `CARDT`
arrest time).

**Develop on real MIMIC-IV.** The MIMIC-IV **Demo** (100 patients) is downloadable
without credentialing; `mimic_explore.py` drives the whole loop:

```bash
python src/mimic_explore.py <mimic-demo-dir>                 # structure, vital coverage, value distribution
python src/mimic_explore.py <mimic-demo-dir> --scan-arrest   # candidate cardiac-arrest itemids (d_items)
python src/mimic_explore.py <mimic-demo-dir> --arrest-counts # how many stays actually arrested
python src/mimic_explore.py <mimic-demo-dir> --model         # XGBoost vs NEWS + personalized features
```

The demo has almost no documented arrests (it is for structure/quality checks);
the full MIMIC-IV (free CITI credentialing) runs the same `--model` command with
real positives. In-hospital cardiac arrest is defined from `procedureevents`
(`ARREST_ITEMIDS` — 225466 "Cardiac Arrest" primary, + respiratory-arrest /
defibrillation for sensitivity).

**Why it matters.** On the synthetic demo, XGBoost and NEWS look near-identical
on ROC-AUC (~0.99) yet diverge sharply on **AUPRC** (≈0.78 vs ≈0.61) — exactly
the alarm-quality gap ROC-AUC hides. SHAP consistently ranks the **trends**
(respiratory-rate, SpO₂ and pulse *slopes*) as the top risk drivers, validating
the sliding-window trend features.

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
