"""Natural-language explanation of a single early-warning alarm.

The model already knows *why* it fired — SHAP contributions plus the window's
own vital values. This module turns that into a sentence a clinician can read.

Two steps, deliberately separate:

1. :func:`build_evidence` — deterministic. Collects the top SHAP drivers, the
   current vital values, the patient's personal baseline (recovered from the
   ``_dev`` features) and the NEWS score into a plain dict.
2. :func:`narrate` — sends that dict to an LLM (Groq, OpenAI-compatible API)
   and gets Korean prose back. The LLM only *words* the evidence; it does not
   judge. Every number it may use is already in the dict.

Step 1 stands alone: it needs no network and no API key, and printing its
output is a perfectly good explanation on its own.

Restricted data (MIMIC-IV, 경북대 KHTH) must not be sent to a third-party API —
PhysioNet's DUA forbids sharing credentialed data. Pass ``source=`` to
:func:`build_evidence` and :func:`narrate` refuses those sources.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Hosted model names expire. Groq retired ``llama-3.3-70b-versatile`` — what
# used to be here — on 2026-06-17 for free and developer tiers, so it now fails
# with a 400 at call time. Kept in step with ``MODEL`` in
# ``web/src/app/api/explain/route.ts``; change both together.
DEFAULT_MODEL = "openai/gpt-oss-120b"

# Cohorts under a data-use agreement that forbids third-party sharing.
RESTRICTED_SOURCES = frozenset({"mimic", "khth"})

VITAL_KO: dict[str, tuple[str, str]] = {
    "pulse": ("맥박", "회/분"),
    "sbp": ("수축기혈압", "mmHg"),
    "dbp": ("이완기혈압", "mmHg"),
    "temperature": ("체온", "°C"),
    "spo2": ("SpO2", "%"),
    "resp_rate": ("호흡수", "회/분"),
}
STAT_KO: dict[str, str] = {
    "mean": "8시간 평균",
    "std": "8시간 변동폭",
    "min": "8시간 최저",
    "max": "8시간 최고",
    "last": "현재값",
    "slope": "시간당 변화 추세",
    "delta": "윈도우 시작 대비 변화",
}
DERIVED_KO: dict[str, tuple[str, str]] = {
    "shock_index_last": ("쇼크지수 (맥박/수축기혈압)", ""),
    "pulse_pressure_last": ("맥압 (수축기-이완기)", "mmHg"),
    "static_age": ("나이", "세"),
    "static_sex": ("성별 (1=남)", ""),
}


def describe_feature(feature: str) -> tuple[str, str]:
    """Map a feature column name to (Korean description, unit)."""
    if feature in DERIVED_KO:
        return DERIVED_KO[feature]
    personal = feature.endswith("_dev")
    stem = feature[: -len("_dev")] if personal else feature
    vital, _, stat = stem.rpartition("_")
    if vital not in VITAL_KO or stat not in STAT_KO:
        return feature, ""
    name, unit = VITAL_KO[vital]
    if personal:
        return f"{name} {STAT_KO[stat]}의 개인 기저선 대비 편차", unit
    return f"{name} {STAT_KO[stat]}", unit


def build_evidence(
    model: Any,
    X: pd.DataFrame,
    row: int,
    *,
    threshold: float = 0.5,
    news_threshold: float = 5.0,
    top_k: int = 3,
    source: str = "unknown",
    patient_id: Any = None,
    hours_to_arrest: float | None = None,
) -> dict[str, Any]:
    """Collect everything needed to explain one window's alarm decision.

    Args:
        model: Trained tree model with ``predict_proba``.
        X: Window feature frame (as fed to the model).
        row: Positional index of the window to explain.
        threshold: Risk probability above which the model alarms.
        news_threshold: NEWS score at or above which the clinical baseline alarms.
        top_k: How many SHAP drivers to report.
        source: Cohort origin — ``"mimic"``/``"khth"`` block :func:`narrate`.
        patient_id: Optional label for display.
        hours_to_arrest: Optional ground truth, for demo/report context only.

    Returns:
        A JSON-serialisable dict; see :func:`format_evidence` for a readable view.
    """
    from vitals_explain import window_risk_factors
    from vitals_train import compute_news_scores

    window = X.iloc[[row]]
    names = list(X.columns)
    risk = float(model.predict_proba(window)[0, 1])
    news = float(compute_news_scores(window)[0])

    factors = []
    for item in window_risk_factors(model, window, names, top_k=top_k):
        feature = item["feature"]
        description, unit = describe_feature(feature)
        factors.append(
            {
                "feature": feature,
                "description": description,
                "unit": unit,
                "value": round(item["feature_value"], 2),
                "shap": round(item["shap_value"], 4),
                "effect": "위험을 높임" if item["shap_value"] > 0 else "위험을 낮춤",
            }
        )

    now: dict[str, float] = {}
    baseline: dict[str, float] = {}
    for vital, (name_ko, _) in VITAL_KO.items():
        last = f"{vital}_last"
        if last not in window:
            continue
        current = float(window[last].iloc[0])
        if not np.isfinite(current):
            continue
        now[name_ko] = round(current, 1)
        dev = f"{vital}_last_dev"
        if dev in window and np.isfinite(window[dev].iloc[0]):
            baseline[name_ko] = round(current - float(window[dev].iloc[0]), 1)

    return {
        "patient_id": None if patient_id is None else str(patient_id),
        "source": source,
        "risk": round(risk, 3),
        "threshold": round(float(threshold), 3),
        "model_alarm": risk >= threshold,
        "news_score": news,
        "news_threshold": news_threshold,
        "news_alarm": news >= news_threshold,
        "top_factors": factors,
        "current_vitals": now,
        "personal_baseline": baseline,
        "hours_to_arrest": hours_to_arrest,
    }


def format_evidence(evidence: dict[str, Any]) -> str:
    """Render the evidence dict as a compact Korean text block.

    Doubles as the LLM prompt body and as a no-network fallback display.
    """
    lines = []
    if evidence.get("patient_id"):
        lines.append(f"환자: {evidence['patient_id']}")
    lines.append(
        f"모델 위험도: {evidence['risk']:.3f} (경보 기준 {evidence['threshold']:.3f}) "
        f"→ {'경보' if evidence['model_alarm'] else '경보 없음'}"
    )
    lines.append(
        f"NEWS 점수: {evidence['news_score']:.0f} (경보 기준 {evidence['news_threshold']:.0f}) "
        f"→ {'경보' if evidence['news_alarm'] else '경보 없음'}"
    )
    lines.append("기여도 상위 요인:")
    for rank, factor in enumerate(evidence["top_factors"], start=1):
        unit = f" {factor['unit']}" if factor["unit"] else ""
        lines.append(
            f"  {rank}. {factor['description']} = {factor['value']}{unit} "
            f"(SHAP {factor['shap']:+.3f}, {factor['effect']})"
        )
    if evidence["current_vitals"]:
        lines.append("현재 활력징후:")
        for name, value in evidence["current_vitals"].items():
            base = evidence["personal_baseline"].get(name)
            suffix = f" (이 환자 기저선 {base})" if base is not None else ""
            lines.append(f"  {name} {value}{suffix}")
    return "\n".join(lines)


SYSTEM_PROMPT = """\
당신은 병동 조기경보 시스템의 설명 생성기입니다. 모델이 계산한 근거를 임상\
의료진이 읽을 한국어 문장으로 옮기는 것이 유일한 역할입니다.

규칙:
- 제공된 숫자만 사용하십시오. 주어지지 않은 수치를 만들어내지 마십시오.
- 진단명을 붙이지 마십시오 (예: "패혈증 의심" 금지).
- 처치나 처방을 권고하지 마십시오. 확인이 필요한 지점까지만 언급하십시오.
- 기저선 값이 함께 주어진 항목만 "이 환자 평소 대비"로 표현하십시오. 추세(slope)\
 처럼 기저선이 없는 항목에 그 표현을 쓰지 마십시오.
- 경보가 없는 경우에는 왜 울리지 않았는지를 같은 방식으로 설명하십시오.

형식: 3문장 이내, 불릿 없이 이어지는 산문. 활력징후를 전부 나열하지 말고 상위\
 요인과 그와 직접 관련된 값만 언급하십시오. 위험도·NEWS 숫자를 그대로 되읽지\
 말고, 무엇이 어떻게 변하고 있는지를 먼저 쓰십시오."""


def _api_key(explicit: str | None = None) -> str:
    """Resolve the Groq key from the argument, the environment, or ``.env``."""
    if explicit:
        return explicit
    if os.environ.get("GROQ_API_KEY"):
        return os.environ["GROQ_API_KEY"]
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "GROQ_API_KEY":
                return value.strip().strip("\"'")
    raise RuntimeError("GROQ_API_KEY not set (environment variable or .env file)")


def narrate(
    evidence: dict[str, Any],
    *,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    temperature: float = 0.2,
    timeout: int = 30,
) -> str:
    """Turn an evidence dict into Korean prose via the Groq API.

    Raises:
        PermissionError: When ``evidence["source"]`` is under a data-use
            agreement that forbids sending it to a third-party service.
    """
    import requests

    source = evidence.get("source", "unknown")
    if source in RESTRICTED_SOURCES:
        raise PermissionError(
            f"cohort source {source!r} is credentialed data and must not be sent to an "
            "external API; use format_evidence() instead"
        )

    response = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {_api_key(api_key)}"},
        json={
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": format_evidence(evidence)},
            ],
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def main() -> None:
    """Explain the riskiest window of a synthetic run, with and without the LLM."""
    from vitals_data import (
        add_personalized_features,
        build_windows,
        generate_synthetic_cohort,
        patient_level_split,
    )
    from vitals_train import train_xgboost

    cohort = generate_synthetic_cohort()
    windowed = add_personalized_features(build_windows(cohort), cohort)
    split = patient_level_split(windowed)
    model, _ = train_xgboost(split)

    scores = model.predict_proba(split.X_test)[:, 1]
    evidence = build_evidence(model, split.X_test, int(np.argmax(scores)), source="synthetic")

    print("=== 근거 (결정론적) ===")
    print(format_evidence(evidence))
    print("\n=== 설명 (LLM) ===")
    try:
        print(narrate(evidence))
    except Exception as error:  # noqa: BLE001 - demo should not die on a network hiccup
        print(f"(건너뜀: {error})")
    print("\n" + json.dumps(evidence, ensure_ascii=False, indent=2)[:400] + " ...")


if __name__ == "__main__":
    main()
