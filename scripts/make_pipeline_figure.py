"""Draw the pipeline overview used in the proposal (§6.1).

§6 is the feasibility section and carried only tables, so what the system
actually does end to end had to be assembled from a numbered list. This lays the
four stages out left to right with the counts that matter — 58 features, of
which the 12 personalized ones are the differentiator — and marks where the two
novelty claims sit, so a reviewer can find them without reading §4 first.

Writes models/vitals_pipeline_concept.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "models" / "vitals_pipeline_concept.png"

INK = "#16222E"
MUTED = "#5A6472"
ACCENT = "#1F4E79"
STAR = "#B8860B"

# stage: (title, lines, fill, edge, title colour)
STAGES = [
    (
        "① 데이터 준비",
        ["활력징후 6종 적재", "센서 아티팩트·단위 보정", "8시간 슬라이딩 윈도우"],
        "#EEF2F6", "#AAB4BE", INK,
    ),
    (
        "② 피처 생성  ★",
        ["통계 피처 44", "개인 기저선 이탈 12", "정적(연령·성별) 2", "─────────  합계 58"],
        "#E8F3E9", "#0F7A12", "#0F7A12",
    ),
    (
        "③ 학습  ★",
        ["within-patient 라벨링", "cost-sensitive XGBoost", "환자 단위 분할", "비교군: NEWS"],
        "#EDF3FB", "#2A78D6", "#2A78D6",
    ),
    (
        "④ 평가·설명",
        ["AUPRC · 알람 부담", "lead-time", "SHAP 기여 요인"],
        "#FDF2EC", "#EB6834", "#C4521F",
    ),
]

BOX_W, BOX_H, GAP = 2.55, 1.72, 0.42


def _korean_font() -> str:
    available = {f.name for f in fm.fontManager.ttflist}
    for name in ("NanumGothic", "NanumBarunGothic", "Malgun Gothic", "AppleGothic"):
        if name in available:
            return name
    print("WARNING: no Korean font found — labels will render as boxes.")
    return plt.rcParams["font.family"][0]


def main() -> Path:
    font = _korean_font()
    plt.rcParams["axes.unicode_minus"] = False

    total_w = len(STAGES) * BOX_W + (len(STAGES) - 1) * GAP
    fig, ax = plt.subplots(figsize=(11.6, 3.5))

    for i, (title, lines, fill, edge, title_colour) in enumerate(STAGES):
        x = i * (BOX_W + GAP)
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, 0), BOX_W, BOX_H,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            facecolor=fill, edgecolor=edge, lw=1.5, zorder=2,
        ))
        ax.text(
            x + BOX_W / 2, BOX_H - 0.24, title, ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=title_colour,
            fontfamily=font, zorder=3,
        )
        for j, line in enumerate(lines):
            bold = line.startswith("─")
            ax.text(
                x + BOX_W / 2, BOX_H - 0.60 - j * 0.27, line,
                ha="center", va="center", fontsize=9,
                fontweight="bold" if bold else "normal",
                color=INK if bold else MUTED, fontfamily=font, zorder=3,
            )
        if i < len(STAGES) - 1:
            ax.annotate(
                "", xy=(x + BOX_W + GAP - 0.06, BOX_H / 2),
                xytext=(x + BOX_W + 0.06, BOX_H / 2),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=2), zorder=3,
            )

    ax.text(
        total_w / 2, -0.30,
        "★ 본 제안의 차별 지점 — 개인 기저선 이탈 피처(②)와 within-patient 라벨링(③)",
        ha="center", va="center", fontsize=10, fontweight="bold",
        color=STAR, fontfamily=font,
    )
    ax.text(
        total_w / 2, -0.62,
        "전 과정이 numpy · pandas · scikit-learn · xgboost · shap · matplotlib 6종으로 동작 — 폐쇄망 사전 통보 대상",
        ha="center", va="center", fontsize=9, color=MUTED, fontfamily=font,
    )

    ax.set_xlim(-0.25, total_w + 0.25)
    ax.set_ylim(-0.85, BOX_H + 0.18)
    ax.axis("off")
    fig.tight_layout(pad=0.35)
    fig.savefig(OUT, dpi=150, facecolor="white")
    plt.close(fig)
    print("Pipeline concept:", OUT)
    return OUT


if __name__ == "__main__":
    main()
