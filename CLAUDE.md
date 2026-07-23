# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## What this is

End-to-end **ML + FastAPI** pipeline for **acute heart failure risk prediction**
with **false-alarm reduction**. Cost-sensitive gradient boosting + SHAP
explainability + an **OMOP CDM v5.4** transformation layer.

Repo: `github.com/BioCode67/hf-risk-prediction-sw` · default branch `main`.

## ⚠️ Read first — data & models are NOT in git

`.gitignore` excludes `data/`, `*.zip`, `*.csv`, `models/*.pkl`, `models/*.png`,
`models/*.json` (to avoid committing PHI / large binaries). A **fresh clone
(including a claude.ai/code cloud session) has NO datasets and NO trained model.**

Consequences on a fresh checkout:
- `python src/train.py`, `data_loader.py`, `explainability.py` will raise
  `FileNotFoundError` until the source archives are placed in `data/`.
- The `/predict` endpoint returns **503** until `models/best_model.pkl` exists
  (produced by training).
- `pytest` still **passes green**: data-dependent tests **skip automatically**
  when datasets are absent (see `tests/conftest.py` `loader` fixture); only the
  pure-function tests run. This is intentional and is what CI relies on.

To run the full pipeline, drop these archives into `data/` (the loader unzips
them on first use):
- `Heart Failure Clinical Records Dataset.zip` → `heart_failure_clinical_records_dataset.csv` (299 rows, target `DEATH_EVENT`)
- `Cardiovascular Disease dataset.zip` → `cardio_train.csv` (70k rows, `;`-separated)

## Layout

```
# static heart-failure track (one row per patient)
src/data_loader.py    # load/unzip, StandardScaler, stratified split, to_omop_cdm()
src/train.py          # LightGBM (Optuna, AUPRC) + XGBoost baseline -> models/best_model.pkl
src/explainability.py # SHAP TreeExplainer: global top-5 + per-patient top-3
src/main.py           # FastAPI app: /predict, /convert-omop, /health

# time-series cardiac-arrest early-warning track (K-Health 공모전) — the active entry
src/vitals_data.py     # synthetic + KHTH + MIMIC adapters, sanitation, windows, personalized
                       #   + static(age/sex) features, patient-level split, lead-time
src/vitals_train.py    # cost-sensitive XGBoost vs NEWS; AUPRC/sens@spec/alarm-burden/lead-time
src/vitals_explain.py  # SHAP drivers (global + per-window)
src/vitals_report.py   # figures: PR-curve, trajectory, lead-time, alarm-burden (render_report)
src/vitals_phenotype.py# unsupervised cardiac-arrest phenotype clustering + heatmap
src/mimic_explore.py   # real MIMIC-IV: explore + --scan-arrest/--arrest-counts/--model
src/omop_explore.py    # explore any OMOP CDM CSV folder (Eunomia / competition sample)

docs/competition-strategy.md  # 공모전 우승 전략 & 제안서 설계 (rubric 정렬)
docs/proposal-draft.md        # 예선 제안서 30장 골격 초안
tests/  # test_pipeline.py (static) + test_vitals/_mimic/_omop/_report.py (time-series)
.github/workflows/ci.yml  # imports check + pytest on Python 3.11 & 3.12
```

Two independent pipelines share the repo. The **time-series track** is the
2026 K-Health 미개방 의료데이터 경진대회 entry (경북대병원 활력징후 →
in-hospital cardiac-arrest early warning). It is **case-only** (arrest patients
only, no controls) so labelling is *within-patient*; it ships a synthetic
generator so it runs with no restricted data, and real `KHTH_PINFO`/`KHTH_VITAL`
plug in via `vitals_data.cohort_from_khth`. Keep it lightweight
(numpy/pandas/sklearn/xgboost/shap only) — the 안심존 is an offline closed
network with pre-declared packages. `tests/test_vitals.py` uses synthetic data
so it always runs (it does not skip like the data-dependent static tests).

## Commands

```bash
pip install -r requirements.txt

python src/data_loader.py       # inspect/preprocess (needs data/)
python src/train.py             # train + save models/best_model.pkl (needs data/)
python src/explainability.py    # SHAP report + models/shap_summary.png (needs model)

uvicorn main:app --app-dir src --reload   # serve API; docs at /docs
pytest -q                                  # test suite
```

## Conventions & gotchas

- **`src/` is not an installed package.** Modules import each other by bare name
  (`from data_loader import ...`). Scripts run via `python src/<file>.py`; the
  API via `--app-dir src`; tests via `conftest.py` inserting `src/` on `sys.path`.
  Don't switch these to `from src.x import ...` without adjusting all entry points.
- **Model artifact name is `models/best_model.pkl`** — used consistently by
  `train.py`, `explainability.py`, and `main.py`. Keep them in sync if renamed.
- **Imbalance handling:** `scale_pos_weight = negatives/positives`; tuning
  **maximizes AUPRC** (not ROC-AUC) because acute events are rare. A
  *sensitivity @ 95% specificity* metric is reported to make the false-alarm
  trade-off explicit.
- **Split:** stratified, `test_size=0.2`, `random_state=42` (deterministic).
- **OMOP CDM v5.4:** `to_omop_cdm()` emits `Person` + `Measurement` tables;
  gender concepts `8507` (male) / `8532` (female); measurement `concept_id`s
  defined in `data_loader.py`.
- **Commits:** Conventional Commits (`feat(...)`, `fix(...)`, `ci:`, `docs:`,
  `chore:`). End commit messages with the `Co-Authored-By` trailer already in use.

## Status (as of 2026-07-23)

- **Static heart-failure track**: built, verified end-to-end (13 tests pass,
  live server smoke-tested), pushed to `origin/main`. CI green on 3.11 & 3.12.
- **Time-series early-warning track** (the K-Health competition entry, branch
  `claude/cardiac-arrest-early-warning-07fq9e`): full pipeline on synthetic /
  MIMIC-IV / KHTH — sanitation, sliding windows, personalized-baseline + age/sex
  features, XGBoost vs NEWS with AUPRC / sensitivity@specificity / **alarm burden
  at matched sensitivity** / **lead-time**, SHAP, a figure set (`vitals_report`)
  and **cardiac-arrest phenotype clustering** (`vitals_phenotype`). `mimic_explore
  --model` runs the whole thing on real MIMIC-IV in one command (arrest =
  procedureevents 225466). 40 tests pass. Docs: `competition-strategy.md`,
  `proposal-draft.md`. Remaining gate: full MIMIC-IV (CITI) for real positives.
