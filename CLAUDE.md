# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## What this is

Vital-sign time series → in-hospital cardiac-arrest early warning, whose
differentiator is false-alarm reduction + explainability, not raw accuracy.
Cost-sensitive gradient boosting + personalized-baseline features + SHAP. The
earlier static heart-failure pipeline (FastAPI + OMOP CDM v5.4) is archived in
`legacy/`.

Repo: `github.com/BioCode67/hf-risk-prediction-sw` · default branch `main`.

## Read first — data & models are NOT in git

`.gitignore` excludes `data/`, `*.zip`, `*.csv`, `models/*.pkl`, `models/*.png`,
`models/*.json` (to avoid committing PHI / large binaries). A fresh clone
(including a claude.ai/code cloud session) has NO datasets and NO trained model.

Consequences on a fresh checkout:
- The active time-series track runs fine with no data (built-in synthetic
  cohort). Only `legacy/` needs the archives.
- `python legacy/train.py`, `legacy/data_loader.py`, `legacy/explainability.py`
  raise `FileNotFoundError` until the source archives are placed in `data/`.
- The `/predict` endpoint returns 503 until `models/best_model.pkl` exists
  (produced by training).
- `pytest` still passes green: data-dependent tests skip automatically
  when datasets are absent (see `legacy/tests/conftest.py` `loader` fixture);
  everything else runs. This is intentional and is what CI relies on.

To run the full pipeline, drop these archives into `data/` (the loader unzips
them on first use):
- `Heart Failure Clinical Records Dataset.zip` → `heart_failure_clinical_records_dataset.csv` (299 rows, target `DEATH_EVENT`)
- `Cardiovascular Disease dataset.zip` → `cardio_train.csv` (70k rows, `;`-separated)

## Layout

```
# time-series cardiac-arrest early-warning track (K-Health 공모전) — the active entry
src/vitals_data.py     # synthetic + KHTH + MIMIC adapters, sanitation, windows, personalized
                       #   + static(age/sex) features, patient-level split, lead-time
src/vitals_train.py    # cost-sensitive XGBoost vs NEWS; AUPRC/sens@spec/alarm-burden/lead-time
src/vitals_explain.py  # SHAP drivers (global + per-window)
src/vitals_report.py   # figures: PR-curve, trajectory, lead-time, alarm-burden (render_report)
src/vitals_phenotype.py# unsupervised cardiac-arrest phenotype clustering + heatmap
src/mimic_explore.py   # real MIMIC-IV: explore + --scan-arrest/--arrest-counts/--model
src/sepsis_explore.py  # PhysioNet Challenge 2019 sepsis (open): --horizon/--tune/--gpu
src/mortality_explore.py # PhysioNet Challenge 2012 ICU mortality (open, 4k/set): --horizon/--outcomes
src/omop_explore.py    # explore any OMOP CDM CSV folder (Eunomia / competition sample)

# deep-learning benchmark (torch; requirements-torch.txt, excluded from CI import check)
src/utils.py           # forward-fill -> mean/zero impute -> z-score
src/dataset.py         # variable-length sequence batching (pad + mask)
src/model.py           # LSTM/GRU classifier
train_dl.py            # (repo root) DL training entry point

notebooks/01_baseline_pipeline.ipynb       # 4-stage walkthrough of this project's pipeline
notebooks/02_learning_project.ipynb        # standard tabular-ML workflow on Pima (teaching)
notebooks/03_challenge2019_realdata.ipynb  # real Challenge-2019 data walkthrough
notebooks/04_challenge2012_timeseries.ipynb# time-series pipeline taught on Challenge-2012
                                           #   (self-generates a demo cohort if data is absent)

README.md      # Korean; structure map + command table + how to read the metrics
src/README.md  # Korean; module map + why src/ must stay flat
docs/README.md # Korean; index of the docs below
docs/competition-strategy.md  # 공모전 우승 전략 & 제안서 설계 (rubric 정렬)
docs/proposal-draft.md        # 예선 제안서 30장 골격 초안
docs/differentiation.md       # 본선 발표·Q&A 대비
docs/STATUS.md                # 현재 진행 상황
docs/server-runbook.md        # GPU 서버 구축·실행 (conda, VS Code, 함정 모음)
tests/  # test_vitals/_mimic/_omop/_report/_sepsis/_mortality/_torch (active, time-series)
conftest.py  # repo root: puts BOTH src/ and legacy/ on sys.path for pytest

# archived static heart-failure track (one row per patient) — NOT the competition entry
legacy/data_loader.py    # load/unzip, StandardScaler, stratified split, to_omop_cdm()
legacy/train.py          # LightGBM (Optuna, AUPRC) + XGBoost baseline -> models/best_model.pkl
legacy/explainability.py # SHAP TreeExplainer: global top-5 + per-patient top-3
legacy/main.py           # FastAPI app: /predict, /convert-omop, /health
legacy/tests/            # test_pipeline.py + its data-dependent fixtures
legacy/README.md         # Korean; why it is kept (FastAPI + OMOP), how to run it
.github/workflows/ci.yml  # imports check + pytest on Python 3.11 & 3.12
```

Docs are written in Korean for the user; code, docstrings and commit
messages stay in English. Keep it that way when editing.

Two independent pipelines share the repo. The time-series track is the
2026 K-Health 미개방 의료데이터 경진대회 entry (경북대병원 활력징후 →
in-hospital cardiac-arrest early warning). It is case-only (arrest patients
only, no controls) so labelling is *within-patient*; it ships a synthetic
generator so it runs with no restricted data, and real `KHTH_PINFO`/`KHTH_VITAL`
plug in via `vitals_data.cohort_from_khth`. Keep it lightweight
(numpy/pandas/sklearn/xgboost/shap only) — the 안심존 is an offline closed
network with pre-declared packages. `tests/test_vitals.py` uses synthetic data
so it always runs (it does not skip like the data-dependent static tests).

## Commands

```bash
pip install -r requirements.txt

# active track — no data required (synthetic cohort built in)
python src/vitals_train.py      # XGBoost vs NEWS + false-alarm metrics
python src/vitals_explain.py    # SHAP drivers
python src/vitals_report.py     # figures -> models/
python src/vitals_phenotype.py  # arrest phenotype clustering

python src/sepsis_explore.py <dir> --horizon=6    # real Challenge-2019 data (sepsis)
python src/mortality_explore.py <dir> --horizon=6 # real Challenge-2012 data (ICU mortality)
python src/mimic_explore.py <dir> --model --gpu  # real MIMIC-IV

pytest -q                       # test suite (55 pass, 11 skip on a fresh clone)

# archived static track (needs data/)
python legacy/train.py
uvicorn main:app --app-dir legacy --reload   # serve API; docs at /docs
```

## Conventions & gotchas

- `src/` and `legacy/` are import roots, not installed packages. Modules
  import siblings by bare name (`from vitals_data import ...`). Scripts run via
  `python src/<file>.py`; the legacy API via `--app-dir legacy`; tests via the
  repo-root `conftest.py`, which puts both directories on `sys.path`. Don't
  switch to `from src.x import ...` without adjusting every entry point — and
  prefer not to require `pip install -e .`, since the 본선 안심존 is offline.
- `legacy/` is archived, not active. The competition entry is the
  time-series track in `src/`. Keep new work out of `legacy/`.
- Model artifact name is `models/best_model.pkl` — used consistently by
  `legacy/train.py`, `legacy/explainability.py`, and `legacy/main.py`. Keep them
  in sync if renamed. The time-series track writes `models/vitals_ews_model.pkl`.
- Imbalance handling: `scale_pos_weight = negatives/positives`; tuning
  maximizes AUPRC (not ROC-AUC) because acute events are rare. A
  *sensitivity @ 95% specificity* metric is reported to make the false-alarm
  trade-off explicit.
- Split: stratified, `test_size=0.2`, `random_state=42` (deterministic).
- OMOP CDM v5.4: `to_omop_cdm()` emits `Person` + `Measurement` tables;
  gender concepts `8507` (male) / `8532` (female); measurement `concept_id`s
  defined in `legacy/data_loader.py`.
- Commits: Conventional Commits (`feat(...)`, `fix(...)`, `ci:`, `docs:`,
  `chore:`). End commit messages with the `Co-Authored-By` trailer already in use.

## Status (as of 2026-07-28)

- Static heart-failure track: complete and archived in `legacy/`. Kept for
  the FastAPI server and the OMOP CDM converter; not part of the competition
  entry. Do not add new work there.
- Time-series early-warning track (the K-Health competition entry, branch
  `claude/cardiac-arrest-early-warning-07fq9e`): full pipeline on synthetic /
  MIMIC-IV / KHTH — sanitation, sliding windows, personalized-baseline + age/sex
  features, XGBoost vs NEWS with AUPRC / sensitivity@specificity / alarm burden
  at matched sensitivity / lead-time, SHAP, a figure set (`vitals_report`)
  and cardiac-arrest phenotype clustering (`vitals_phenotype`). `mimic_explore
  --model` runs the whole thing on real MIMIC-IV in one command (arrest =
  procedureevents 225466). 40 tests pass. Docs: `competition-strategy.md`,
  `proposal-draft.md`. Remaining gate: full MIMIC-IV (CITI) for real positives.
