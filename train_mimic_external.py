#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PRODROME — MIMIC-IV로 학습한 외부 모델 만들기 (안심존 외부검증용) + MIMIC 자체 성능.

본인 PC/서버에서만 실행한다(MIMIC-IV DUA: 자격 인증자 본인만 데이터 접근, 제3자 공유 금지).
결과물 중 안심존에 가져갈 것은 model.json + features.json 뿐이고(가중치 = 사전학습 모델, 반입 허용 항목),
나머지는 집계표다. 환자 단위 행·식별자는 어떤 출력에도 쓰지 않는다.

    python train_mimic_external.py <MIMIC-IV 폴더(hosp/ icu/ 포함)> --density-only     # 1단계: 기록 밀도만(수 분)
    python train_mimic_external.py <MIMIC-IV 폴더> --out=mimic_ext                     # 2단계: 학습(5·10분 단위)

    옵션: --buckets=5,10       학습할 시간 단위(분). 경북대 run_khth 설정 B_5min·A_10min 과 맞춘다
          --horizon-buckets=3  예측 지평(칸) — 경북대 본 분석과 같은 3칸(5분 단위 15분 앞, 10분 단위 30분 앞)
          --controls=3000      심정지 없는 ICU 체류 표본 수(오경보 평가용, 0이면 생략)
          --control-hours=12   대조군은 ICU 퇴실 전 마지막 몇 시간을 쓸지
          --gpu                CUDA XGBoost

출력(<out>/):
    00_density.csv                심정지 전 60분 기록 밀도(집계) — 외부검증이 성립하는지 판단 근거
    b5/model.json, b5/features.json   ← 안심존 반입 대상 (b10/ 도 동일)
    b5/mimic_internal.csv         MIMIC 안에서의 성능(XGBoost vs NEWS, AUPRC 95%CI)
    b5/mimic_controls.csv         심정지 없는 환자에서의 경보 빈도(같은 임계값) — 오경보 근거
    00_summary.txt                위 내용 요약(저에게 공유할 것은 이 파일과 csv 집계뿐)

경북대 실행: py run_khth.py C:\\데이터톤 --external=<반입한 mimic_ext 폴더>
  → 단위가 맞는 설정(B_5min ← b5, A_10min ← b10)에서 10_external_validation.csv 가 생긴다.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

# 경북대 분석과 '똑같은' 전처리·특징을 쓰기 위해 run_khth 의 함수를 그대로 가져온다.
from run_khth import (add_clinical_score_features, bootstrap_auprc_ci, normalize, prearrest_and_rescale,
                      safe_patient_split)

ITEM2CODE = {220045: "HR", 220050: "SBP", 220179: "SBP", 220051: "DBP", 220180: "DBP",
             223762: "BT", 223761: "BT", 220277: "SPO2", 220210: "RR"}
FAHRENHEIT = {223761}
ARREST_ITEMS = (225466,)          # procedureevents "Cardiac Arrest" — 가장 깨끗한 정의
KNOWN = {"--density-only", "--out", "--buckets", "--horizon-buckets", "--controls", "--control-hours", "--gpu",
         "--arrest-items", "--seed"}


def _read(root: Path, mod: str, name: str, **kw) -> pd.DataFrame:
    for base in (root / mod, root):
        for suf in (".csv.gz", ".csv"):
            p = base / f"{name}{suf}"
            if p.exists():
                return pd.read_csv(p, **kw)
    raise FileNotFoundError(f"{mod}/{name}.csv(.gz) 를 {root} 아래에서 찾지 못했습니다")


def _chart_path(root: Path) -> Path:
    for p in (root / "icu" / "chartevents.csv.gz", root / "icu" / "chartevents.csv",
              root / "chartevents.csv.gz", root / "chartevents.csv"):
        if p.exists():
            return p
    raise FileNotFoundError("icu/chartevents.csv(.gz) 가 없습니다")


def load_stays(root: Path, arrest_items, n_controls: int, seed: int):
    print("[1/4] icustays · patients · procedureevents 읽는 중")
    st = _read(root, "icu", "icustays", usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"])
    pt = _read(root, "hosp", "patients", usecols=["subject_id", "gender", "anchor_age"])
    pr = _read(root, "icu", "procedureevents", usecols=["stay_id", "starttime", "itemid"])
    ar = pr[pr["itemid"].isin(arrest_items)].copy()
    ar["starttime"] = pd.to_datetime(ar["starttime"], errors="coerce")
    ar = ar.groupby("stay_id")["starttime"].min().rename("arrest_time").reset_index()
    st = st.merge(pt, on="subject_id", how="left").merge(ar, on="stay_id", how="left")
    st["intime"] = pd.to_datetime(st["intime"], errors="coerce")
    st["outtime"] = pd.to_datetime(st["outtime"], errors="coerce")
    st = st[st["arrest_time"].isna() | (st["arrest_time"] > st["intime"])]      # 입실 전 기록 제외
    arrest = st[st["arrest_time"].notna()]
    ctrl_pool = st[st["arrest_time"].isna() & st["outtime"].notna()]
    ctrl = ctrl_pool.sample(n=min(n_controls, len(ctrl_pool)), random_state=seed) if n_controls > 0 else ctrl_pool.iloc[:0]
    print(f"  ICU 체류 {len(st):,} · 심정지 체류 {len(arrest):,} · 대조군 표본 {len(ctrl):,}")
    return arrest, ctrl


def load_vitals(root: Path, stay_ids: set) -> pd.DataFrame:
    path = _chart_path(root)
    print(f"[2/4] {path.name} 에서 활력징후 추출 중(큰 파일 — 수십 분 걸릴 수 있음)")
    keep, n = [], 0
    t0 = time.time()
    for ch in pd.read_csv(path, usecols=["stay_id", "charttime", "itemid", "valuenum"], chunksize=2_000_000):
        ch = ch[ch["itemid"].isin(ITEM2CODE.keys()) & ch["stay_id"].isin(stay_ids)]
        if len(ch):
            keep.append(ch)
        n += 1
        if n % 25 == 0:
            print(f"  … {n * 2:,}M 행 처리 ({time.time() - t0:.0f}s)")
    v = pd.concat(keep, ignore_index=True) if keep else pd.DataFrame(columns=["stay_id", "charttime", "itemid", "valuenum"])
    v["valuenum"] = pd.to_numeric(v["valuenum"], errors="coerce")
    f = v["itemid"].isin(FAHRENHEIT)
    v.loc[f, "valuenum"] = (v.loc[f, "valuenum"] - 32.0) * 5.0 / 9.0
    v["charttime"] = pd.to_datetime(v["charttime"], errors="coerce")
    v["VS_GBN"] = v["itemid"].map(ITEM2CODE)
    print(f"  활력징후 {len(v):,}행 ({time.time() - t0:.0f}s)")
    return v.dropna(subset=["charttime", "valuenum"])


def density_table(v: pd.DataFrame, arrest: pd.DataFrame) -> pd.DataFrame:
    """심정지 전 60분 동안 기록이 얼마나 촘촘한지 — 집계만."""
    a = v.merge(arrest[["stay_id", "arrest_time"]], on="stay_id")
    a["rel_min"] = (a["charttime"] - a["arrest_time"]) / pd.Timedelta("1min")
    pre = a[(a["rel_min"] < 0) & (a["rel_min"] >= -60)]
    n_st = arrest["stay_id"].nunique()
    rows = [("심정지 체류 수", n_st)]
    for code in ("HR", "SBP", "SPO2", "RR"):
        s = pre[pre["VS_GBN"] == code].sort_values(["stay_id", "charttime"])
        cnt = s.groupby("stay_id").size().reindex(arrest["stay_id"], fill_value=0)
        gap = s.groupby("stay_id")["charttime"].diff().dt.total_seconds().div(60).dropna()
        rows += [(f"{code}: 심정지 전 60분 측정 횟수 중앙값", float(cnt.median())),
                 (f"{code}: 심정지 전 60분 측정 간격 중앙값(분)", float(gap.median()) if len(gap) else np.nan),
                 (f"{code}: 심정지 전 60분에 3회 이상 측정된 체류 비율(%)", round(100 * float((cnt >= 3).mean()), 1))]
    for w in (10, 30, 60):
        has = pre[pre["rel_min"] >= -w].groupby("stay_id").size().reindex(arrest["stay_id"], fill_value=0)
        rows.append((f"심정지 전 {w}분 안에 측정이 1회 이상 있는 체류 비율(%)", round(100 * float((has > 0).mean()), 1)))
    return pd.DataFrame(rows, columns=["항목", "값"])


def to_khth(v: pd.DataFrame, stays: pd.DataFrame, anchor_col: str | None, hours_before: float | None):
    """MIMIC → 경북대(KHTH) 표 형식. anchor 이전 측정만(대조군은 퇴실 전 hours_before 시간만)."""
    s = stays.copy()
    s["PATID"] = s["subject_id"].astype(str)
    s["INDD"] = s["stay_id"].astype(str)
    m = v.merge(s[["stay_id", "PATID", "INDD"] + ([anchor_col] if anchor_col else [])], on="stay_id")
    if anchor_col and hours_before:
        rel = (m["charttime"] - m[anchor_col]) / pd.Timedelta("1h")
        m = m[(rel < 0) & (rel >= -hours_before)]
    vital = pd.DataFrame({"PATID": m["PATID"], "INDD": m["INDD"], "VSDT": m["charttime"].dt.strftime("%Y%m%d%H%M"),
                          "VS_GBN": m["VS_GBN"], "VS_RSLT": m["valuenum"].round(2).astype(str)})
    pinfo = pd.DataFrame({"PATID": s["PATID"], "INDD": s["INDD"], "AGE": s["anchor_age"].astype(str),
                          "SEX": s["gender"].astype(str),
                          "CARDT": s["arrest_time"].dt.strftime("%Y%m%d%H%M") if "arrest_time" in s else np.nan})
    return pinfo, vital


def rescale_controls(pinfo, vital, stays, bucket_min):
    """대조군: 퇴실 시각을 기준점으로 경북대와 같은 비율로 시간축을 늘린다(심정지 시각 대신)."""
    s = stays.assign(INDD=stays["stay_id"].astype(str))[["INDD", "outtime"]]
    v = vital.merge(s, on="INDD")
    t = pd.to_datetime(v["VSDT"], format="%Y%m%d%H%M")
    k = 60.0 / bucket_min
    v["VSDT"] = (v["outtime"] + (t - v["outtime"]) * k).dt.strftime("%Y%m%d%H%M")
    return v.drop(columns=["outtime"])


def windows(pinfo, vital, horizon_buckets):
    from vitals_data import add_personalized_features, add_static_features, build_windows, cohort_from_khth
    cohort = cohort_from_khth(vital, pinfo)
    w = build_windows(cohort, prediction_horizon_hours=horizon_buckets, min_valid_fraction=0.34)
    return add_static_features(add_personalized_features(w, cohort), cohort)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opt = {a.split("=")[0]: (a.split("=", 1)[1] if "=" in a else True) for a in sys.argv[1:] if a.startswith("--")}
    bad = [k for k in opt if k not in KNOWN]
    if bad or not args:
        print(__doc__ if not args else f"[오류] 알 수 없는 옵션 {bad} — 사용 가능: {sorted(KNOWN)}")
        sys.exit(2)
    root = Path(args[0])
    out = Path(opt.get("--out", "mimic_ext")); out.mkdir(parents=True, exist_ok=True)
    buckets = [int(b) for b in str(opt.get("--buckets", "5,10")).split(",")]
    horizon_b = int(opt.get("--horizon-buckets", 3))
    n_ctrl = int(opt.get("--controls", 3000))
    ctrl_h = float(opt.get("--control-hours", 12))
    seed = int(opt.get("--seed", 42))
    items = tuple(int(x) for x in str(opt.get("--arrest-items", ",".join(map(str, ARREST_ITEMS)))).split(","))
    gpu = "--gpu" in opt

    arrest, ctrl = load_stays(root, items, 0 if "--density-only" in opt else n_ctrl, seed)
    v = load_vitals(root, set(arrest["stay_id"]) | set(ctrl["stay_id"]))
    dens = density_table(v, arrest)
    dens.to_csv(out / "00_density.csv", index=False, encoding="utf-8-sig")
    print("\n[기록 밀도 — 심정지 전 60분]\n" + dens.to_string(index=False))
    summary = ["PRODROME — MIMIC-IV 외부 모델 요약(집계)", "=" * 50, dens.to_string(index=False), ""]
    if "--density-only" in opt:
        (out / "00_summary.txt").write_text("\n".join(summary), encoding="utf-8")
        print(f"\n저장: {out / '00_density.csv'}  — 이 표를 공유해 주세요.")
        return

    import xgboost as xgb
    from vitals_train import compute_news_scores, evaluate, threshold_at_specificity, train_xgboost
    from run_khth import mews_scores

    a_pinfo, a_vital = to_khth(v, arrest, "arrest_time", None)
    a_pinfo, a_vital = normalize(a_pinfo, a_vital)
    c_pinfo, c_vital = to_khth(v, ctrl, "outtime", ctrl_h) if len(ctrl) else (None, None)
    for b in buckets:
        hb = horizon_b   # 경북대 본 분석과 같은 칸 수(3칸: 5분 단위 15분 앞, 10분 단위 30분 앞)
        print(f"\n[3/4] 단위 {b}분 · {hb * b}분 앞 예측 — MIMIC 심정지 체류로 학습")
        d = out / f"b{b}"; d.mkdir(exist_ok=True)
        pv, _ = prearrest_and_rescale(a_pinfo, a_vital, b)
        w = windows(a_pinfo, pv, hb)
        pos = int(w.labels.sum())
        print(f"  창 {len(w.labels):,} · 양성 창 {pos:,} · 체류 {len(set(w.groups)):,}")
        if pos < 20:
            msg = f"단위 {b}분: 양성 창 {pos}개 — 기록이 너무 성겨 학습 생략(외부검증 불가 근거)"
            print("  " + msg); summary.append(msg); continue
        split = add_clinical_score_features(safe_patient_split(w, seed=seed))
        model, m = train_xgboost(split, use_gpu=gpu)
        s = model.predict_proba(split.X_test)[:, 1]
        news = compute_news_scores(split.X_test)
        rows = []
        for name, sc in (("XGBoost(MIMIC 학습·MIMIC 평가)", s), ("NEWS", news), ("MEWS(의식 제외)", mews_scores(split.X_test))):
            mm = evaluate(name, split.y_test, sc)
            lo, hi = bootstrap_auprc_ci(split.y_test, sc)
            rows.append({"모델": name, "AUPRC": mm.auprc, "AUPRC 95%CI 하한": lo, "AUPRC 95%CI 상한": hi,
                         "ROC_AUC": mm.roc_auc, "민감도@특이도95%": mm.sensitivity_at_95_specificity})
        internal = pd.DataFrame(rows)
        internal.insert(0, "단위(분)", b); internal.insert(1, "예측 지평(분)", hb * b)
        internal["평가 체류 수"] = len(set(split.groups_test)); internal["평가 양성 창 수"] = int(split.y_test.sum())
        internal.to_csv(d / "mimic_internal.csv", index=False, encoding="utf-8-sig")
        print(internal.to_string(index=False))
        summary += [f"[단위 {b}분 · {hb * b}분 앞] MIMIC 내부 성능", internal.to_string(index=False), ""]

        # 오경보: 심정지 없는 환자에서 같은 임계값(심정지 체류 평가셋 특이도 95%)으로 경보가 얼마나 울리나
        if c_pinfo is not None:
            thr_x = threshold_at_specificity(split.y_test, s)
            thr_n = threshold_at_specificity(split.y_test, news)
            cv = rescale_controls(c_pinfo, c_vital, ctrl, b)
            cp, cv = normalize(c_pinfo.assign(CARDT=np.nan), cv)
            try:
                wc = windows(cp, cv, hb)
                Xc = wc.features.copy()
            except Exception as e:
                Xc = None; print(f"  (대조군 창 생성 실패: {e})")
            if Xc is not None and not len(Xc):
                msg = (f"  대조군: {b}분 단위로는 평가 창이 0개 — 일반 병상 기록(약 1시간 간격)이 이 해상도에 비해 너무 성김"
                       f"(대조군 오경보 평가는 이 단위에서 불가)")
                print(msg); summary.append(msg.strip())
            if Xc is not None and len(Xc):
                Xc["clin_news"] = compute_news_scores(Xc); Xc["clin_mews"] = mews_scores(Xc)
                Xc = Xc.reindex(columns=split.X_train.columns)
                sx, sn = model.predict_proba(Xc)[:, 1], compute_news_scores(Xc)
                g = pd.Series(np.asarray(wc.groups))
                ctl = pd.DataFrame([
                    {"모델": "XGBoost", "대조군 체류 수": g.nunique(), "대조군 창 수": len(Xc),
                     "경보/100창": 100 * float((sx >= thr_x).mean()),
                     "경보가 1회 이상 울린 체류 비율(%)": 100 * float(pd.Series(sx >= thr_x).groupby(g).any().mean())},
                    {"모델": "NEWS", "대조군 체류 수": g.nunique(), "대조군 창 수": len(Xc),
                     "경보/100창": 100 * float((sn >= thr_n).mean()),
                     "경보가 1회 이상 울린 체류 비율(%)": 100 * float(pd.Series(sn >= thr_n).groupby(g).any().mean())}])
                ctl.to_csv(d / "mimic_controls.csv", index=False, encoding="utf-8-sig")
                print("  심정지 없는 환자에서의 경보(같은 임계값):\n" + ctl.to_string(index=False))
                summary += ["심정지 없는 환자(퇴실 전 마지막 구간)에서의 경보", ctl.to_string(index=False), ""]

        # 반입용 최종 모델: 평가가 끝난 뒤 MIMIC 전체(학습+평가)로 다시 학습
        from dataclasses import replace
        full = replace(split, X_train=pd.concat([split.X_train, split.X_test]),
                       y_train=np.concatenate([split.y_train, split.y_test]))
        final, _ = train_xgboost(full, use_gpu=gpu)
        final.save_model(str(d / "model.json"))
        meta = {"features": list(split.X_train.columns), "bucket_min": b, "horizon_min": hb * b,
                "source": "MIMIC-IV", "arrest_definition": f"procedureevents itemid {list(items)}",
                "train_stays": int(len(set(w.groups))), "train_positive_windows": pos,
                "generated": time.strftime("%Y-%m-%d %H:%M")}
        (d / "features.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  반입 대상 저장: {d / 'model.json'} · {d / 'features.json'}")

    (out / "00_summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print(f"\n[4/4] 완료 — 안심존 반입: {out}/b*/model.json, features.json  · 공유: {out}/00_summary.txt")


if __name__ == "__main__":
    main()
