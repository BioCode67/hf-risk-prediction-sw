#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PRODROME(전조) — K-Health 본선 안심존 원커맨드 분석.

    python run_khth.py <KHTH 데이터 폴더 또는 파일>          # 실데이터 — 설정 3가지를 한 번에(자동)
    python run_khth.py --selftest                            # 합성 데이터로 동작 확인(안심존 첫 실행 전)
    옵션: --bucket-min=10 --horizons=3,6 (설정 하나만 돌릴 때)  --tune --trials=30  --out=results
    심정지 이후 측정은 버린다(이 데이터는 측정의 대부분이 심정지 이후).

자동 모드(기본): results/ 아래에 설정별 폴더를 만든다 — 하나가 실패해도 나머지는 계속.
  A_10min  심정지 전 측정, 10분 단위, 30·60분 앞 예측
  B_5min   심정지 전 측정, 5분 단위, 15·30분 앞 예측 (심정지 전 기록이 짧은 입원까지 포함)
  C_60min  심정지 전 측정, 1시간 단위, 1·4시간 앞 예측 (10/27 첫 분석과 같은 설정)
  00_data_profile.csv / 00_config_summary.csv — 데이터 특성과 설정별 핵심 성능(집계)

입력: KHTH_PINFO, KHTH_VITAL (csv / xlsx / parquet / sas7bdat 자동 탐색, 인코딩 utf-8 → cp949 자동)
출력(results/): 반출용 **집계값만** — 환자 단위 행·식별자·원값은 쓰지 않는다.
  00_README.txt            무엇이 들어 있는지 · 반출 심의용 설명
  01_cohort_summary.csv    코호트 규모·결측률·관찰시간(집계)
  02_model_comparison.csv  XGBoost(+개인 기저선) / XGBoost(기저선 없음) / NEWS — AUPRC·ROC·민감도@95%특이도
  03_alarm_burden.csv      같은 민감도(50·70·80·90%)에서 경보 부담
  04_lead_time.csv         선행시간 요약(중앙값·사분위·검출률)
  05_shap_top.csv          전역 SHAP 상위 15 특징
  06_phenotypes.csv        표현형 군집 중심(변화량) · 군집 크기(5 미만은 '<5' 마스킹)
  07_horizon_sensitivity.csv  예측 지평별 성능
  08_temporal_validation.csv  시간적 검증(입원 시점 앞 80% 학습 → 뒤 20% 평가)
  09_subgroup.csv          연령(65세)·성별 하위집단 AUPRC (n<5 마스킹)
  fig_*.png                PR 곡선 · 경보부담 · 선행시간 · SHAP · 표현형 · 심정지 전 평균 궤적 · 보정 곡선
  02 에는 NEWS·MEWS·로지스틱·랜덤포레스트·HistGB 비교와 환자 단위 부트스트랩 95% CI, NEWS 대비 ΔAUPRC 포함
  metrics.json             위 수치 전체
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
import pandas as pd

MASK_N = 5  # 반출 소표본 마스킹 기준


# ─────────────────────────────────────────────── 입력
def _read_any(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path, dtype=str)
    if suf == ".sas7bdat":
        return pd.read_sas(path, encoding="utf-8")
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, dtype=str, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"읽기 실패: {path}")


PINFO_COLS = ("PATID", "AGE", "SEX", "INDD", "OUDD", "OUTDD", "CARDT", "DEATHDT")
VITAL_COLS = ("PATID", "INDD", "VSDT", "VS_GBN", "VS_RSLT")
DATA_EXT = (".csv", ".txt", ".xlsx", ".xls", ".parquet", ".pq", ".sas7bdat")


def split_merged(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One long table with patient columns repeated on every vital row
    (e.g. KNUH_DATA.csv) -> (pinfo, vital)."""
    df.columns = [c.strip().upper() for c in df.columns]
    pinfo = df[[c for c in PINFO_COLS if c in df.columns]].drop_duplicates(["PATID", "INDD"]).reset_index(drop=True)
    vital = df[[c for c in VITAL_COLS if c in df.columns]]
    print(f"  통합 파일 분리: 입원 {len(pinfo):,}건 / 측정 {len(vital):,}행")
    return pinfo, vital


def find_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
    data_files = [p for p in files if p.suffix.lower() in DATA_EXT and "results" not in p.parts]
    # 통합 파일(한 파일에 VS_GBN 과 CARDT 가 함께 있음)을 먼저 찾는다.
    for p in sorted(data_files, key=lambda p: p.stat().st_size, reverse=True):
        try:
            head = pd.read_csv(p, nrows=0, encoding="utf-8-sig") if p.suffix.lower() in (".csv", ".txt") else None
        except UnicodeDecodeError:
            head = pd.read_csv(p, nrows=0, encoding="cp949")
        except Exception:
            head = None
        cols = {c.strip().upper() for c in head.columns} if head is not None else set()
        if {"VS_GBN", "CARDT", "PATID"} <= cols:
            print(f"  통합 데이터 파일: {p}")
            return split_merged(_read_any(p))
    pick = {}
    for key in ("PINFO", "VITAL"):
        cands = [p for p in files if key in p.name.upper()
                 and p.suffix.lower() in (".csv", ".txt", ".xlsx", ".xls", ".parquet", ".pq", ".sas7bdat")]
        if not cands:
            raise SystemExit(f"[오류] {root} 아래에서 KHTH_{key} 파일을 찾지 못했습니다. 파일 경로를 직접 주세요.")
        pick[key] = sorted(cands, key=lambda p: p.stat().st_size, reverse=True)[0]
        print(f"  KHTH_{key}: {pick[key]}")
    pinfo, vital = _read_any(pick["PINFO"]), _read_any(pick["VITAL"])
    pinfo.columns = [c.strip().upper() for c in pinfo.columns]
    vital.columns = [c.strip().upper() for c in vital.columns]
    return pinfo, vital


def normalize(pinfo: pd.DataFrame, vital: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """설명서 스키마에 맞춘다: AGE는 '60대' 같은 연령대 문자열일 수 있다, 날짜는 YYYYMMDDHHMI."""
    pinfo = pinfo.copy(); vital = vital.copy()
    if "AGE" in pinfo.columns:
        pinfo["AGE"] = pinfo["AGE"].astype(str).str.extract(r"(\d+)")[0]
    if "SEX" in pinfo.columns:
        s = pinfo["SEX"].astype(str).str.strip().str.upper()
        pinfo["SEX"] = s.replace({"1": "M", "2": "F", "남": "M", "여": "F", "MALE": "M", "FEMALE": "F"})
    for df, cols in ((pinfo, ("INDD", "OUDD", "CARDT", "DEATHDT")), (vital, ("INDD", "VSDT"))):
        for c in cols:
            if c in df.columns:
                v = df[c].astype(str).str.replace(r"[^0-9]", "", regex=True)
                # 날짜만(YYYYMMDD) 있는 열은 00:00으로 채운다. 조인키(INDD)는 원문 유지 후 숫자만.
                df[c] = v.where(v.str.len() != 8, v + "0000").str[:12]
    if "VS_GBN" in vital.columns:
        vital["VS_GBN"] = vital["VS_GBN"].astype(str).str.strip().str.upper().replace({"PR": "HR", "PULSE": "HR", "SPO₂": "SPO2"})
    return pinfo, vital


# ─────────────────────────────────────────────── 합성 KHTH (selftest)
def fake_khth(n=160, seed=0) -> tuple[pd.DataFrame, pd.DataFrame]:
    from vitals_data import generate_synthetic_cohort
    c = generate_synthetic_cohort(n_patients=n, arrest_fraction=1.0, seed=seed)
    base = pd.Timestamp("2024-03-01 08:00")
    rows = []
    code = {"pulse": "HR", "sbp": "SBP", "dbp": "DBP", "temperature": "BT", "spo2": "SPO2", "resp_rate": "RR"}
    ids = {pid: f"P{i:04d}" for i, pid in enumerate(c.vitals["patient_id"].unique())}
    for r in c.vitals.itertuples(index=False):
        t = base + pd.Timedelta(hours=float(r.hour)) + pd.Timedelta(minutes=7)
        for v, g in code.items():
            val = getattr(r, v)
            if pd.notna(val):
                rows.append((ids[r.patient_id], "20240301", t.strftime("%Y%m%d%H%M"), g, f"{val:.1f}"))
    vital = pd.DataFrame(rows, columns=["PATID", "INDD", "VSDT", "VS_GBN", "VS_RSLT"])
    ev = c.events.set_index("patient_id")["arrest_hour"]
    rng = np.random.default_rng(seed)
    pinfo = pd.DataFrame({
        "PATID": [ids[p] for p in ids], "INDD": "20240301",
        "AGE": [f"{rng.choice([20,30,40,50,60,70,80])}대" for _ in ids],
        "SEX": rng.choice(["M", "F"], len(ids)),
        "CARDT": [(base + pd.Timedelta(hours=float(ev[p]))).strftime("%Y%m%d%H%M") for p in ids],
    })
    return pinfo, vital


# ─────────────────────────────────────────────── 시간 단위·심정지 전 필터
UNIT_MIN = 60  # 파이프라인의 '1시간' 한 칸이 실제 몇 분인지 (--bucket-min 으로 바꿈)


def _dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype(str).str.replace(r"[^0-9]", "", regex=True).str[:12],
                          format="%Y%m%d%H%M", errors="coerce")


def prearrest_and_rescale(pinfo: pd.DataFrame, vital: pd.DataFrame, bucket_min: int):
    """심정지 이후 측정을 버리고, 시간축을 bucket_min 분 = 1 칸으로 늘린다.

    파이프라인은 '시간' 단위로 창을 만든다. 측정이 심정지 직전 몇 시간에 몰린
    데이터에서는 10분 단위가 맞으므로, 심정지 시각을 기준으로 (측정-심정지)를
    60/bucket_min 배 늘려 같은 코드를 분 단위로 돌린다. 출력에서 다시 분으로 환산한다.
    """
    v = vital.copy()
    card = pinfo[["PATID", "INDD", "CARDT"]].drop_duplicates(["PATID", "INDD"])
    v = v.drop(columns=[c for c in ("CARDT",) if c in v.columns]).merge(card, on=["PATID", "INDD"], how="left")
    t, c = _dt(v["VSDT"]), _dt(v["CARDT"])
    rel_h = (t - c) / pd.Timedelta("1h")
    prof = {
        "측정 중 심정지 이후 비율(%)": round(100 * float((rel_h >= 0).mean()), 1),
        "측정 중 심정지 6h 이전 비율(%)": round(100 * float((rel_h < -6).mean()), 1),
        "측정 중 심정지 24h 이전 비율(%)": round(100 * float((rel_h < -24).mean()), 1),
    }
    keep = rel_h < 0
    v = v[keep].copy()
    k = 60.0 / bucket_min
    v["VSDT"] = (c[keep] + (t[keep] - c[keep]) * k).dt.strftime("%Y%m%d%H%M")
    print(f"  심정지 전 측정만 사용: {int(keep.sum()):,}/{len(keep):,}행, 단위 {bucket_min}분")
    return v[[col for col in ("PATID", "INDD", "VSDT", "VS_GBN", "VS_RSLT") if col in v.columns]], prof


def data_profile(pinfo: pd.DataFrame, vital: pd.DataFrame) -> pd.DataFrame:
    """입원별 '심정지 전 기록 길이' 분포 — 어떤 시간 단위가 맞는지 판단하는 근거(집계만)."""
    v = vital.merge(pinfo[["PATID", "INDD", "CARDT"]].drop_duplicates(["PATID", "INDD"]), on=["PATID", "INDD"], how="left")
    t, c = _dt(v["VSDT"]), _dt(v["CARDT"])
    v = v.assign(rel_min=(t - c) / pd.Timedelta("1min"), stay=v["PATID"].astype(str) + "_" + v["INDD"].astype(str))
    pre = v[v["rel_min"] < 0]
    span = (-pre.groupby("stay")["rel_min"].min()).reindex(v["stay"].unique()).fillna(0)
    rows = [("입원 수", len(span)),
            ("심정지 전 측정이 있는 입원 수", int((span > 0).sum())),
            ("심정지 전 기록 길이 중앙값(분)", round(float(span[span > 0].median()), 1) if (span > 0).any() else np.nan)]
    for m in (10, 20, 30, 60, 120, 360, 1440):
        rows.append((f"심정지 전 기록 {m}분 이상 입원 수", mask(int((span >= m).sum()))))
    rows += [("측정 중 심정지 이후 비율(%)", round(100 * float((v["rel_min"] >= 0).mean()), 1)),
             ("측정 중 심정지 1h 이전 비율(%)", round(100 * float((v["rel_min"] < -60).mean()), 1)),
             ("측정 중 심정지 6h 이전 비율(%)", round(100 * float((v["rel_min"] < -360).mean()), 1))]
    return pd.DataFrame(rows, columns=["항목", "값"])


CONFIGS = (("A_10min", 10, (3, 6)), ("B_5min", 5, (3, 6)), ("C_60min", 60, (1, 4)))


def run_auto(pinfo, vital, out: Path, tune=False, trials=30, gpu=False) -> None:
    """설정 3가지를 차례로 돌린다. 하나가 실패해도 나머지는 계속하고, 요약표를 남긴다."""
    import traceback
    out.mkdir(parents=True, exist_ok=True)
    prof = data_profile(pinfo, vital)
    prof.to_csv(out / "00_data_profile.csv", index=False, encoding="utf-8-sig")
    print("\n[데이터 특성]\n" + prof.to_string(index=False))
    summary = []
    for name, bucket, hz in CONFIGS:
        print(f"\n==================== {name} (단위 {bucket}분, 예측 {[h * bucket for h in hz]}분 앞) ====================")
        row = {"설정": name, "단위(분)": bucket, "예측 지평(분)": "·".join(str(h * bucket) for h in hz)}
        try:
            v2, p2 = prearrest_and_rescale(pinfo, vital, bucket)
            M = run(pinfo, v2, out / name, horizons=hz, tune=tune, trials=trials, gpu=gpu, bucket_min=bucket, profile=p2)
            hr = (M.get("horizons") or [{}])[0]
            mc = {m["모델"]: m for m in M.get("model_comparison", [])}
            x, n = mc.get("XGBoost(개인 기저선 포함)", {}), mc.get("NEWS", {})
            row.update({"창 수": hr.get("windows"), "양성 창 수": hr.get("positive_windows"),
                        "테스트 입원 수": mask(int(hr.get("test_patients", 0) or 0)),
                        "XGB AUPRC": x.get("AUPRC"), "NEWS AUPRC": n.get("AUPRC"),
                        "XGB ROC": x.get("ROC_AUC"), "NEWS ROC": n.get("ROC_AUC"),
                        "XGB 민감도@특이도95%": x.get("민감도@특이도95%"), "상태": "완료" if x else "양성 부족(학습 생략)"})
        except Exception as e:
            traceback.print_exc()
            row["상태"] = f"실패: {type(e).__name__}"
        summary.append(row)
        pd.DataFrame(summary).to_csv(out / "00_config_summary.csv", index=False, encoding="utf-8-sig")
    (out / "00_README.txt").write_text(README_OUT + README_AUTO, encoding="utf-8")
    s = pd.DataFrame(summary)
    print("\n[설정별 요약]\n" + s.to_string(index=False))
    done = s[s["상태"] == "완료"]
    if len(done):
        best = done.sort_values("XGB AUPRC", ascending=False).iloc[0]
        print(f"\n→ AUPRC 기준 가장 좋은 설정: {best['설정']} (XGB {best['XGB AUPRC']:.3f} vs NEWS {best['NEWS AUPRC']:.3f})")
    print(f"\n반출 대상: {out.resolve()} 폴더 전체 (집계값만)")


# ─────────────────────────────────────────────── 모델 비교·신뢰구간
def benchmark_models(split, evaluate) -> list:
    """XGBoost만 쓰지 않았다는 근거 — 같은 환자 분할에서 대표 모델들을 함께 학습·평가한다.

    로지스틱 회귀(해석 가능한 기준선), 랜덤 포레스트, HistGradientBoosting.
    전부 scikit-learn(안심존 기설치)만 쓰고, 불균형은 class_weight 로 맞춘다.
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    models = {
        "로지스틱 회귀": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  LogisticRegression(max_iter=2000, class_weight="balanced")),
        "랜덤 포레스트": make_pipeline(SimpleImputer(strategy="median"),
                                  RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                                         n_jobs=-1, random_state=42)),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                               class_weight="balanced", random_state=42),
    }
    out = []
    for name, m in models.items():
        try:
            m.fit(split.X_train, split.y_train)
            s = m.predict_proba(split.X_test)[:, 1]
            out.append((evaluate(name, split.y_test, s), s))
        except Exception as e:
            print(f"({name} 생략: {e})")
    return out


def mews_scores(X: pd.DataFrame) -> np.ndarray:
    """MEWS(Subbe 2001) — 의식(AVPU) 항목은 데이터에 없어 제외한 4항목 합.

    딥카스(DEWS) 원 논문이 비교군으로 쓴 점수라, NEWS와 함께 기준선으로 둔다.
    """
    def band(v, edges, pts):
        v = np.asarray(v, dtype=float); out = np.zeros_like(v)
        for (lo, hi), p in zip(edges, pts):
            out = np.where((v >= lo) & (v < hi), p, out)
        return np.where(np.isnan(v), 0, out)
    sbp = band(X["sbp_last"], [(-1e9, 71), (71, 81), (81, 101), (101, 200), (200, 1e9)], [3, 2, 1, 0, 2])
    hr = band(X["pulse_last"], [(-1e9, 41), (41, 51), (51, 101), (101, 111), (111, 130), (130, 1e9)], [2, 1, 0, 1, 2, 3])
    rr = band(X["resp_rate_last"], [(-1e9, 9), (9, 15), (15, 21), (21, 30), (30, 1e9)], [2, 0, 1, 2, 3])
    bt = band(X["temperature_last"], [(-1e9, 35), (35, 38.5), (38.5, 1e9)], [2, 0, 2])
    return (sbp + hr + rr + bt).astype(float)


def cluster_bootstrap(y, s_by_model: dict, groups, n=500, seed=0) -> dict:
    """환자 단위 부트스트랩 — 같은 환자의 창끼리 상관을 고려한 AUPRC 95% CI와,
    기준선(NEWS) 대비 AUPRC 차이(Δ)의 95% CI·양(+)일 확률. 집계값만 반환한다."""
    from sklearn.metrics import average_precision_score
    y = np.asarray(y); groups = np.asarray(groups)
    pats = np.unique(groups); idx_by = {p: np.where(groups == p)[0] for p in pats}
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in s_by_model}
    for _ in range(n):
        pick = rng.choice(pats, len(pats), replace=True)
        i = np.concatenate([idx_by[p] for p in pick])
        if not (0 < y[i].sum() < len(i)):
            continue
        for k, s in s_by_model.items():
            draws[k].append(average_precision_score(y[i], np.asarray(s)[i]))
    res = {}
    if not draws or len(next(iter(draws.values()))) < 100:
        return res
    base = np.array(draws.get("NEWS", []))
    for k, v in draws.items():
        v = np.array(v)
        r = {"lo": float(np.quantile(v, .025)), "hi": float(np.quantile(v, .975))}
        if len(base) and k != "NEWS":
            d = v - base
            r.update({"d_lo": float(np.quantile(d, .025)), "d_hi": float(np.quantile(d, .975)), "p_better": float((d > 0).mean())})
        res[k] = r
    return res


def calibration_outputs(y, s, path_png: Path) -> dict:
    """보정(calibration) — Brier 점수와 10분위 보정 곡선(집계)."""
    from sklearn.metrics import brier_score_loss
    from sklearn.calibration import calibration_curve
    out = {"brier": float(brier_score_loss(y, s))}
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        _font()
        fy, fx = calibration_curve(y, s, n_bins=10, strategy="quantile")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], ls="--", color="#9AA6B2", lw=1)
        ax.plot(fx, fy, marker="o", color="#0EA5B7", lw=2)
        ax.set_xlabel("예측 위험도(10분위 평균)"); ax.set_ylabel("실제 양성 비율")
        ax.set_title(f"보정 곡선 — Brier {out['brier']:.3f}")
        fig.tight_layout(); fig.savefig(path_png, dpi=150); plt.close(fig)
    except Exception as e:
        print(f"(보정 그림 생략: {e})")
    return out


def temporal_split(w):
    """시간적 검증 — 입원 시점(INDD) 순으로 앞 80% 입원으로 학습, 뒤 20%로 평가.
    무작위 분할보다 엄격하다(미래 환자에 대한 성능). 날짜 자체는 출력하지 않는다."""
    from vitals_data import PatientSplit
    g = w.groups.astype(str)
    pats = pd.Series(g.unique())
    order = pats.str.rsplit("_", n=1).str[-1]
    pats = pats.iloc[np.argsort(order.to_numpy(), kind="stable")].to_numpy()
    cut = int(len(pats) * 0.8)
    tr, te = g.isin(pats[:cut]).to_numpy(), g.isin(pats[cut:]).to_numpy()
    Xtr, Xte = w.features[tr].reset_index(drop=True), w.features[te].reset_index(drop=True)
    med = Xtr.median(numeric_only=True)
    tta = w.time_to_arrest[te].to_numpy() if getattr(w, "time_to_arrest", None) is not None else None
    return PatientSplit(X_train=Xtr.fillna(med), X_test=Xte.fillna(med), y_train=w.labels[tr].to_numpy(),
                        y_test=w.labels[te].to_numpy(), groups_train=g[tr].to_numpy(), groups_test=g[te].to_numpy(),
                        feature_names=list(Xtr.columns), imputation_medians=med, time_to_arrest_test=tta)


def subgroup_table(split, score, news) -> pd.DataFrame:
    """하위집단 공정성 — 연령(65세 기준)·성별별 AUPRC. 양성 창·환자 수가 5 미만이면 마스킹."""
    from sklearn.metrics import average_precision_score
    X = split.X_test; y = np.asarray(split.y_test); g = np.asarray(split.groups_test)
    defs = []
    if "static_age" in X.columns:
        defs += [("연령 65세 미만", (X["static_age"] < 65).to_numpy()), ("연령 65세 이상", (X["static_age"] >= 65).to_numpy())]
    if "static_sex" in X.columns:
        defs += [("남성", (X["static_sex"] == 1).to_numpy()), ("여성", (X["static_sex"] == 0).to_numpy())]
    rows = []
    for name, m in defs:
        npat, npos = len(set(g[m])), int(y[m].sum())
        ok = npat >= MASK_N and npos >= MASK_N and npos < m.sum()
        rows.append({"하위집단": name, "입원 수": mask(npat), "양성 창 수": mask(npos),
                     "XGBoost AUPRC": average_precision_score(y[m], np.asarray(score)[m]) if ok else np.nan,
                     "NEWS AUPRC": average_precision_score(y[m], np.asarray(news)[m]) if ok else np.nan})
    return pd.DataFrame(rows)


def bootstrap_auprc_ci(y, s, n=300, seed=0):
    """AUPRC 95% 부트스트랩 신뢰구간(창 단위 재표집). 양성이 한쪽에 몰린 표본은 건너뛴다."""
    from sklearn.metrics import average_precision_score
    y = np.asarray(y); s = np.asarray(s)
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            vals.append(average_precision_score(y[i], s[i]))
    if len(vals) < 50:
        return (np.nan, np.nan)
    return (float(np.quantile(vals, .025)), float(np.quantile(vals, .975)))


# ─────────────────────────────────────────────── 분석
def mask(n: int) -> str:
    return f"<{MASK_N}" if n < MASK_N else str(int(n))


def run(pinfo, vital, out: Path, horizons=(1,), tune=False, trials=30, gpu=False, bucket_min=60, profile=None) -> dict:
    from vitals_data import (VITALS, add_personalized_features, add_static_features,
                             build_windows, cohort_from_khth, patient_level_split)
    from vitals_train import (alarm_burden, compute_news_scores, evaluate, lead_time_summary,
                              threshold_at_specificity, train_xgboost, tune_xgboost)
    from vitals_report import plot_alarm_burden_curve, plot_lead_time_distribution, plot_pr_curves

    out.mkdir(parents=True, exist_ok=True)
    M: dict = {"generated": time.strftime("%Y-%m-%d %H:%M"), "mask_threshold": MASK_N, "bucket_min": bucket_min}
    u = bucket_min / 60.0  # 한 칸 → 시간
    t0 = time.time()

    cohort = cohort_from_khth(vital, pinfo)
    v = cohort.vitals
    stays = v["patient_id"].nunique()
    arrest_known = int(cohort.events["arrest_hour"].notna().sum())
    obs_h = v.groupby("patient_id")["hour"].max() * u
    ah = cohort.events["arrest_hour"].dropna() * u
    cs = pd.DataFrame([
        ("입원(PATID+INDD) 수", stays),
        ("환자(PATID) 수", pinfo["PATID"].nunique() if "PATID" in pinfo else np.nan),
        ("심정지 시각(CARDT) 확인 입원 수", arrest_known),
        ("활력징후 원 측정 행 수", len(vital)),
        (f"{bucket_min}분 단위 행 수", len(v)),
        ("관찰시간 중앙값(h)", float(obs_h.median())),
        ("관찰시간 IQR(h)", f"{obs_h.quantile(.25):.0f}–{obs_h.quantile(.75):.0f}"),
        ("첫 측정→심정지 중앙값(h)", float(ah.median()) if len(ah) else np.nan),
    ] + list((profile or {}).items()), columns=["항목", "값"])
    miss = (1 - v[list(VITALS)].notna().mean()).mul(100).round(1)
    for k_, m_ in miss.items():
        cs.loc[len(cs)] = (f"칸별 결측률 {k_}(%)", m_)
    cs.to_csv(out / "01_cohort_summary.csv", index=False, encoding="utf-8-sig")
    M["cohort"] = {r["항목"]: (r["값"] if not isinstance(r["값"], (np.floating, np.integer)) else float(r["값"])) for _, r in cs.iterrows()}
    print(cs.to_string(index=False))

    # 심정지 전 평균 궤적(집계) — 개별 환자 그림 대신
    try:
        _mean_trajectory(cohort, out / "fig_6_prearrest_mean_trajectory.png", VITALS, bucket_min)
    except Exception as e:
        print(f"(평균 궤적 그림 생략: {e})")

    comparison, burden_rows, horizon_rows = [], [], []
    for h in horizons:
        base_w = build_windows(cohort, prediction_horizon_hours=h, min_valid_fraction=0.34)
        w_full = add_static_features(add_personalized_features(base_w, cohort), cohort)
        pos = int(w_full.labels.sum())
        print(f"\n[horizon {h*bucket_min}분] windows={len(w_full.labels):,} positives={pos:,}")
        if pos < 10:
            print("  양성 창이 10개 미만 — 건너뜀")
            horizon_rows.append({"horizon_min": h * bucket_min, "windows": len(w_full.labels),
                                 "positive_windows": mask(pos), "test_patients": 0, "note": "양성 창 10개 미만 — 학습 생략"})
            continue
        split = patient_level_split(w_full)
        params = tune_xgboost(split, n_trials=trials, use_gpu=gpu) if tune else {}
        model, m_full = train_xgboost(split, use_gpu=gpu, **params)
        score = model.predict_proba(split.X_test)[:, 1]
        news = compute_news_scores(split.X_test)
        m_news = evaluate("NEWS", split.y_test, news)

        # 기저선 피처 제거(ablation) — 같은 환자 분할 유지
        drop = [c for c in split.X_train.columns if c.endswith("_dev")]
        from dataclasses import replace
        split_nb = replace(split, X_train=split.X_train.drop(columns=drop), X_test=split.X_test.drop(columns=drop),
                           feature_names=[f for f in split.feature_names if f not in drop])
        model_nb, m_nb = train_xgboost(split_nb, use_gpu=gpu, **params)
        m_nb.model_name = "XGBoost(기저선 피처 제외)"
        m_full.model_name = "XGBoost(개인 기저선 포함)"

        horizon_rows.append({"horizon_min": h * bucket_min, "windows": len(w_full.labels), "positive_windows": pos,
                             "test_patients": int(len(set(split.groups_test))),
                             "XGB_AUPRC": m_full.auprc, "NEWS_AUPRC": m_news.auprc,
                             "XGB_ROC": m_full.roc_auc, "NEWS_ROC": m_news.roc_auc,
                             "prevalence": m_full.prevalence})
        if h != horizons[0]:
            continue

        score_nb = model_nb.predict_proba(split_nb.X_test)[:, 1]
        bench = benchmark_models(split, evaluate)
        mews = mews_scores(split.X_test)
        m_mews = evaluate("MEWS(의식 제외)", split.y_test, mews)
        rows_ = [(m_full, score), (m_nb, score_nb), (m_news, news), (m_mews, mews)] + bench
        boot = cluster_bootstrap(split.y_test, {("NEWS" if m.model_name == "NEWS" else m.model_name): s_ for m, s_ in rows_},
                                 split.groups_test)
        M["calibration_xgb"] = calibration_outputs(split.y_test, score, out / "fig_7_calibration.png")
        # 하위집단(연령·성별) 성능
        try:
            sg = subgroup_table(split, score, news)
            if len(sg):
                sg.to_csv(out / "09_subgroup.csv", index=False, encoding="utf-8-sig"); M["subgroup"] = sg.to_dict("records")
        except Exception as e:
            print(f"(하위집단 분석 생략: {e})")
        # 시간적 검증(앞 80% 입원 학습 → 뒤 20% 평가)
        try:
            st = temporal_split(w_full)
            if 0 < st.y_test.sum() < len(st.y_test) and st.y_train.sum() >= MASK_N:
                mt, m_t = train_xgboost(st, use_gpu=gpu, **params)
                st_news = compute_news_scores(st.X_test)
                n_t = evaluate("NEWS", st.y_test, st_news)
                tv = pd.DataFrame([{"검증 방식": "시간적 검증(입원 시점 앞 80% 학습 → 뒤 20% 평가)",
                                    "평가 입원 수": mask(len(set(st.groups_test))), "평가 양성 창 수": mask(int(st.y_test.sum())),
                                    "XGBoost AUPRC": m_t.auprc, "NEWS AUPRC": n_t.auprc,
                                    "XGBoost ROC": m_t.roc_auc, "NEWS ROC": n_t.roc_auc}])
                tv.to_csv(out / "08_temporal_validation.csv", index=False, encoding="utf-8-sig"); M["temporal"] = tv.to_dict("records")
                print(f"  시간적 검증: XGB AUPRC {m_t.auprc:.3f} vs NEWS {n_t.auprc:.3f}")
        except Exception as e:
            print(f"(시간적 검증 생략: {e})")
        for m, s_ in rows_:
            b_ = boot.get(m.model_name, {})
            lo, hi = b_.get("lo", np.nan), b_.get("hi", np.nan)
            comparison.append({"모델": m.model_name, "AUPRC": m.auprc, "AUPRC 95%CI 하한(환자 부트스트랩)": lo,
                               "AUPRC 95%CI 상한(환자 부트스트랩)": hi,
                               "NEWS 대비 ΔAUPRC 하한": b_.get("d_lo", np.nan), "NEWS 대비 ΔAUPRC 상한": b_.get("d_hi", np.nan),
                               "NEWS보다 나을 확률": b_.get("p_better", np.nan),
                               "ROC_AUC": m.roc_auc,
                               "민감도@특이도95%": m.sensitivity_at_95_specificity,
                               "정밀도@특이도95%": m.precision_at_threshold,
                               "경보/100창@특이도95%": m.alarms_per_100_windows, "양성률": m.prevalence})
        for name, s in (("XGBoost", score), ("NEWS", news)):
            for sens in (0.5, 0.7, 0.8, 0.9):
                b = alarm_burden(split.y_test, s, target_sensitivity=sens)
                burden_rows.append({"모델": name, "목표 민감도": sens, "실제 민감도": b["sensitivity"],
                                    "특이도": b["specificity"], "오경보 비율": b["false_alarm_rate"],
                                    "경보/100창": b["alarms_per_100_windows"]})

        thr = threshold_at_specificity(split.y_test, score)
        lead = lead_time_summary(split, score, thr)
        frame = pd.DataFrame({"p": split.groups_test, "tta": split.time_to_arrest_test, "s": score})
        arr = frame[frame["tta"].notna() & (frame["tta"] >= 0)]
        lts = [u * float(g[(g.s >= thr) & (g.tta <= 48)].tta.max()) for _, g in arr.groupby("p") if ((g.s >= thr) & (g.tta <= 48)).any()]
        lt = pd.DataFrame([{
            "기준": f"XGBoost @ 특이도 95% 임계값, 심정지 전 {48*bucket_min/60:g}h 이내 첫 경보",
            "평가 심정지 입원 수": mask(int(lead.get("arrest_patients", 0))),
            "검출 수": mask(int(lead.get("detected", 0))),
            "검출률": lead.get("detection_rate", np.nan),
            "선행시간 중앙값(h)": float(np.median(lts)) if len(lts) >= MASK_N else np.nan,
            "선행시간 Q1(h)": float(np.quantile(lts, .25)) if len(lts) >= MASK_N else np.nan,
            "선행시간 Q3(h)": float(np.quantile(lts, .75)) if len(lts) >= MASK_N else np.nan,
        }])
        lt.to_csv(out / "04_lead_time.csv", index=False, encoding="utf-8-sig")
        M["lead_time"] = lt.iloc[0].to_dict()

        try:
            plot_pr_curves(split.y_test, {"XGBoost": score, "NEWS": news}, out / "fig_1_pr_curve.png")
            plot_alarm_burden_curve(split.y_test, {"XGBoost": score, "NEWS": news}, out / "fig_2_alarm_burden.png",
                                    sensitivities=np.linspace(0.5, 0.95, 10))
            if len(lts) >= MASK_N:
                plot_lead_time_distribution(lts, out / "fig_3_lead_time.png")
        except Exception as e:
            print(f"(그림 일부 생략: {e})")

        # SHAP — 전역 중요도(집계)만. 개별 창 설명은 반출하지 않는다.
        try:
            from vitals_explain import global_top_features
            Xs = split.X_test.sample(min(5000, len(split.X_test)), random_state=0)
            top = pd.DataFrame(global_top_features(model, Xs, list(split.X_test.columns), top_k=15))
            top.to_csv(out / "05_shap_top.csv", index=False, encoding="utf-8-sig")
            M["shap_top"] = top.to_dict("records")
            _bar(top, out / "fig_4_shap_top.png")
        except Exception as e:
            print(f"(SHAP 생략: {e})")

        M["params"] = params

    pd.DataFrame(comparison).to_csv(out / "02_model_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(burden_rows).to_csv(out / "03_alarm_burden.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(horizon_rows).to_csv(out / "07_horizon_sensitivity.csv", index=False, encoding="utf-8-sig")
    M["model_comparison"], M["alarm_burden"], M["horizons"] = comparison, burden_rows, horizon_rows

    # 표현형
    try:
        from vitals_phenotype import cluster_phenotypes, patient_prearrest_signatures, plot_phenotypes
        sig = patient_prearrest_signatures(cohort)
        if len(sig) >= 3 * MASK_N:
            labels, cent = cluster_phenotypes(sig, k=3)
            sizes = pd.Series(labels).value_counts().sort_index()
            ph = cent.round(2).copy()
            ph["군집 크기"] = [mask(int(n)) for n in sizes]
            ph.to_csv(out / "06_phenotypes.csv", encoding="utf-8-sig")
            M["phenotypes"] = ph.reset_index().astype(str).to_dict("records")
            if sizes.min() >= MASK_N:
                plot_phenotypes(cent, sizes, out / "fig_5_phenotypes.png")
    except Exception as e:
        print(f"(표현형 생략: {e})")

    M["elapsed_sec"] = round(time.time() - t0, 1)
    (out / "metrics.json").write_text(json.dumps(M, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "00_README.txt").write_text(README_OUT, encoding="utf-8")
    print(f"\n완료 — {out.resolve()}  ({M['elapsed_sec']}초)")
    for p in sorted(out.iterdir()):
        print("  ", p.name)
    return M


def _font():
    import matplotlib
    from matplotlib import font_manager
    for f in sorted((HERE / "fonts").glob("*.ttf")):
        font_manager.fontManager.addfont(str(f))
    names = {f.name for f in font_manager.fontManager.ttflist}
    for n in ("Pretendard", "NanumGothic", "Malgun Gothic", "AppleGothic"):
        if n in names:
            matplotlib.rcParams["font.family"] = n
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


def _bar(top: pd.DataFrame, path: Path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    _font()
    t = top.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(t["feature"], t["mean_abs_shap"], color="#0EA5B7")
    ax.set_xlabel("mean |SHAP|"); ax.set_title("전역 특징 중요도 (상위 15)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _mean_trajectory(cohort, path: Path, VITALS, bucket_min=60):
    """심정지 시각 기준 -48h~0h, 시간대별 중앙값·IQR (각 시점 n≥MASK_N만)."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    _font()
    ev = cohort.events.dropna(subset=["arrest_hour"]).set_index("patient_id")["arrest_hour"]
    v = cohort.vitals[cohort.vitals["patient_id"].isin(ev.index)].copy()
    v["rel"] = np.floor(v["hour"] - v["patient_id"].map(ev))
    v = v[(v["rel"] >= -48) & (v["rel"] < 0)]
    if v.empty:
        return
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, vit in zip(axes.ravel(), VITALS):
        g = v.groupby("rel")[vit]
        q = g.quantile([.25, .5, .75]).unstack()
        n = g.count()
        q = q[n >= MASK_N]
        ax.fill_between(q.index, q[.25], q[.75], alpha=.25, color="#0EA5B7")
        ax.plot(q.index, q[.5], color="#0A7C8A", lw=2)
        ax.axvline(0, color="#D97B1A", ls="--", lw=1)
        ax.set_title(vit); ax.set_xlabel(f"심정지까지 ({bucket_min}분 단위 칸)")
    fig.suptitle("심정지 전 48시간 활력징후 — 시간대별 중앙값과 IQR (집계)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


README_AUTO = """
- 폴더 구성: A_10min · B_5min · C_60min = 시간 단위가 다른 같은 분석. 00_config_summary.csv 에 설정별 핵심 성능.
- 00_data_profile.csv: 입원별 '심정지 전 기록 길이' 분포(입원 수 집계, n<5 마스킹).
"""

README_OUT = """PRODROME(전조) 분석 결과물 — 반출 신청용 설명
==================================================
- 모든 파일은 집계 통계(평균·중앙값·비율·모델 성능 지표)와 그 그림만 담는다.
- 환자 식별자(PATID), 날짜·시각, 개별 측정값, 환자 단위 예측값은 포함하지 않는다.
- 소표본(셀 n<5)은 '<5'로 마스킹했고, n<5인 시점·군집은 그림에서 제외했다.
- 모델 파일(가중치)은 반출 대상이 아니다.
"""


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opt = {a.split("=")[0]: (a.split("=", 1)[1] if "=" in a else True) for a in sys.argv[1:] if a.startswith("--")}
    bucket = int(opt.get("--bucket-min", 60 if "--selftest" in opt else 10))
    horizons = tuple(int(x) for x in str(opt.get("--horizons", "1,4" if bucket >= 60 else "3,6")).split(","))
    out = Path(opt.get("--out", "results"))
    if "--selftest" in opt:
        print("[selftest] 합성 KHTH 데이터로 전체 흐름을 확인합니다(실데이터 아님).")
        pinfo, vital = fake_khth()
        out = Path(opt.get("--out", "results_selftest"))
    else:
        if not args:
            print(__doc__); return
        print("[입력 탐색]")
        pinfo, vital = find_tables(Path(args[0]))
        print(f"  PINFO 열: {list(pinfo.columns)}\n  VITAL 열: {list(vital.columns)}")
        print(f"  VS_GBN 값: {sorted(vital['VS_GBN'].astype(str).str.upper().unique())[:20]}")
    pinfo, vital = normalize(pinfo, vital)
    kw = dict(tune="--tune" in opt, trials=int(opt.get("--trials", 30)), gpu="--gpu" in opt)
    if "--selftest" not in opt and "--bucket-min" not in opt and "--horizons" not in opt:
        run_auto(pinfo, vital, out, **kw)       # 실데이터 기본: 설정 3가지
        return
    vital, prof = prearrest_and_rescale(pinfo, vital, bucket)
    run(pinfo, vital, out, horizons=horizons, bucket_min=bucket, profile=prof, **kw)


if __name__ == "__main__":
    main()
