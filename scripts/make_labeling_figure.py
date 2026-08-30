"""Draw the within-patient labelling diagram used in the proposal (§4.3).

The competition cohort is case-only: every patient arrested, so there is no
control group to draw negatives from. The design turns that into the method —
negatives come from the same patient's earlier stable hours, positives from the
hours just before their arrest. That is hard to follow in prose and obvious as
a timeline, which is what this draws.

Parameters mirror vitals_data: 8h observation window, 1h prediction horizon,
6h personal baseline.

Writes models/vitals_labeling_concept.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "models" / "vitals_labeling_concept.png"

INK = "#16222E"
MUTED = "#5A6472"
NEG_BLUE = "#5FA8F0"
POS_RED = "#C02626"
BASE_GREEN = "#0F7A12"
ARREST = "#7A1020"
LINE = "#C8D2DC"

ARREST_HOUR = 30
WINDOW_H = 8          # observation window
HORIZON_H = 1         # prediction horizon
BASELINE_H = 6        # personal baseline stretch


def _korean_font() -> str:
    """Delegate to the shared resolver, which registers the bundled TTFs."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from plot_fonts import ensure_korean_font

    return ensure_korean_font()


def _window(ax, *, start, y, colour, label, font, hatch=None):
    """One observation window drawn as a bar on the patient's timeline."""
    ax.add_patch(mpatches.FancyBboxPatch(
        (start, y - 0.16), WINDOW_H, 0.32,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=colour, edgecolor="white", lw=1.4, alpha=0.9,
        hatch=hatch, zorder=3,
    ))
    ax.text(
        start + WINDOW_H / 2, y, label, ha="center", va="center",
        fontsize=10, fontweight="bold", color="white", fontfamily=font, zorder=4,
    )


def main() -> Path:
    font = _korean_font()
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(11.5, 4.6))

    # ── patient timeline
    ax.plot([0, ARREST_HOUR + 2], [0, 0], color=LINE, lw=3, zorder=1, solid_capstyle="round")

    # personal baseline stretch
    ax.add_patch(mpatches.Rectangle(
        (0, -0.10), BASELINE_H, 0.20, facecolor=BASE_GREEN, alpha=0.22,
        edgecolor=BASE_GREEN, lw=1.2, zorder=2,
    ))
    ax.text(
        BASELINE_H / 2, -0.42, f"개인 기저선 구간\n(초기 {BASELINE_H}시간)",
        ha="center", va="top", fontsize=9.5, color=BASE_GREEN,
        fontweight="bold", linespacing=1.5, fontfamily=font,
    )

    # arrest marker
    ax.axvline(ARREST_HOUR, color=ARREST, lw=2.2, zorder=5)
    ax.plot([ARREST_HOUR], [0], marker="X", ms=15, color=ARREST, zorder=6)
    ax.text(
        ARREST_HOUR + 0.5, 1.28, "심정지 발생\n(CARDT)", ha="left", va="top",
        fontsize=10.5, fontweight="bold", color=ARREST,
        linespacing=1.5, fontfamily=font,
    )

    # ── negative windows: the same patient's earlier, stable hours
    for start in (2, 11):
        _window(ax, start=start, y=0.72, colour=NEG_BLUE, label="음성", font=font)
    ax.text(
        2, 1.13, "음성 윈도우 — 같은 환자의 안정 구간",
        fontsize=10.5, fontweight="bold", color=NEG_BLUE, fontfamily=font,
    )

    # ── positive window: ends within the horizon before the arrest
    pos_start = ARREST_HOUR - HORIZON_H - WINDOW_H
    _window(ax, start=pos_start, y=-0.72, colour=POS_RED, label="양성", font=font)
    ax.annotate(
        "",
        xy=(ARREST_HOUR, -0.72), xytext=(pos_start + WINDOW_H, -0.72),
        arrowprops=dict(arrowstyle="->", color=POS_RED, lw=1.6), zorder=4,
    )
    ax.text(
        pos_start + WINDOW_H + HORIZON_H / 2, -0.56,
        f"예측 지평 {HORIZON_H}h", ha="center", va="bottom",
        fontsize=9.5, color=POS_RED, fontfamily=font,
    )
    ax.text(
        pos_start - 7.5, -1.16,
        "양성 윈도우 — 심정지 직전, 같은 환자에게서",
        fontsize=10.5, fontweight="bold", color=POS_RED, fontfamily=font,
    )

    # window length callout
    ax.annotate(
        "", xy=(2, 0.34), xytext=(2 + WINDOW_H, 0.34),
        arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.2), zorder=4,
    )
    ax.text(
        2 + WINDOW_H / 2, 0.27, f"관찰 윈도우 {WINDOW_H}h", ha="center", va="top",
        fontsize=9.5, color=MUTED, fontfamily=font,
    )

    ax.set_xlim(-1.5, ARREST_HOUR + 8)
    ax.set_ylim(-1.55, 1.55)
    ax.set_xlabel("입원 후 경과 시간 (h)", fontsize=10, color=MUTED, fontfamily=font)
    ax.set_yticks([])
    ax.set_xticks(range(0, ARREST_HOUR + 1, 6))
    ax.tick_params(labelsize=9.5)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    fig.suptitle(
        "대조군이 없는 코호트에서, 라벨은 환자 내부에서 만들어진다",
        fontsize=13, fontweight="bold", color=INK, fontfamily=font, y=0.97,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT, dpi=150, facecolor="white")
    plt.close(fig)
    print("Labeling concept:", OUT)
    return OUT


if __name__ == "__main__":
    main()
