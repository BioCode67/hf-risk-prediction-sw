"""Draw the personalized-baseline concept figure used in the proposal (§4.1).

The differentiator is hard to state in prose and instant as a picture: two
patients reach the same absolute respiratory rate, a ward-wide threshold fires
on both, and only one of them is actually deviating from their own norm. That
second alarm is the false alarm this project is built to remove.

Writes models/vitals_baseline_concept.png. Korean labels need a Korean face —
install fonts-nanum if the fallback warning appears.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "models" / "vitals_baseline_concept.png"

MODEL_BLUE = "#2A78D6"
ALARM_RED = "#C02626"
CALM_GREEN = "#0F7A12"
INK = "#16222E"
MUTED = "#5A6472"
BAND = "#E8EEF5"

WARD_THRESHOLD = 20.0  # ward-wide respiratory-rate cutoff, breaths/min


def _korean_font() -> str:
    """Pick an installed Korean face, or fall back to the default."""
    available = {f.name for f in fm.fontManager.ttflist}
    for name in ("NanumGothic", "NanumBarunGothic", "Malgun Gothic", "AppleGothic"):
        if name in available:
            return name
    print("WARNING: no Korean font found — labels will render as boxes.")
    return plt.rcParams["font.family"][0]


def _patient_series(baseline: float, final: float, seed: int) -> np.ndarray:
    """A flat stretch at `baseline` that ramps to `final` over the last third."""
    rng = np.random.default_rng(seed)
    hours = np.arange(24)
    stable = np.full(16, baseline)
    ramp = np.linspace(baseline, final, 8)
    series = np.concatenate([stable, ramp])
    return series + rng.normal(0, 0.35, hours.size)


def _panel(ax, *, hours, series, baseline, title, deviating, font) -> None:
    ax.plot(hours, series, color=MODEL_BLUE, lw=2.2, zorder=3)
    ax.scatter(hours[-1], series[-1], s=70, color=MODEL_BLUE, zorder=4)

    # The two reference lines can sit within a breath of each other (patient B),
    # so anchor their labels on opposite sides and at opposite ends of the axis.
    ax.axhspan(baseline - 1.5, baseline + 1.5, color=BAND, zorder=0)
    ax.axhline(baseline, color=CALM_GREEN, lw=1.6, ls="--", zorder=2)
    ax.text(
        0.3, baseline + 0.45, f"개인 기저선 {baseline:.0f}",
        color=CALM_GREEN, fontsize=10, fontfamily=font, va="bottom", zorder=5,
    )

    ax.axhline(WARD_THRESHOLD, color=ALARM_RED, lw=1.6, ls=":", zorder=2)
    ax.text(
        0.3, WARD_THRESHOLD - 0.5, f"병동 공통 임계값 {WARD_THRESHOLD:.0f}",
        color=ALARM_RED, fontsize=10, fontfamily=font, va="top", zorder=5,
    )

    # A one-breath deviation makes for an arrow too short to read, so the
    # label sits clear of the lines and points at it instead of straddling it.
    deviation = series[-1] - baseline
    arrow_x = hours[-1] - 2.6
    if deviation >= 2:
        ax.annotate(
            "",
            xy=(arrow_x, series[-1]), xytext=(arrow_x, baseline),
            arrowprops=dict(arrowstyle="<->", color=INK, lw=1.6),
            zorder=5,
        )
    ax.annotate(
        f"이탈 +{deviation:.0f}",
        xy=(arrow_x, (series[-1] + baseline) / 2),
        xytext=(arrow_x - 4.2, 24.2),
        arrowprops=dict(arrowstyle="-", color=INK, lw=0.9, shrinkB=2),
        color=INK, fontsize=11.5, fontweight="bold",
        ha="center", va="center", fontfamily=font, zorder=6,
    )

    ax.set_title(title, fontsize=12, fontfamily=font, color=INK, pad=10)
    ax.set_ylabel("호흡수 (회/분)", fontsize=10, fontfamily=font, color=MUTED)
    ax.tick_params(labelsize=9.5)
    ax.set_ylim(10, 26)
    ax.set_xlim(-0.5, 24)
    ax.grid(axis="y", color="#DDE3EA", lw=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    verdict = "경보가 맞다 — 실제 악화" if deviating else "오경보 — 이 환자의 평소 상태"
    ax.text(
        0.5, -0.20, verdict, transform=ax.transAxes, ha="center",
        fontsize=12, fontweight="bold", fontfamily=font,
        color=ALARM_RED if deviating else CALM_GREEN,
    )


def main() -> Path:
    font = _korean_font()
    plt.rcParams["axes.unicode_minus"] = False

    hours = np.arange(24)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    _panel(
        axes[0], hours=hours,
        series=_patient_series(baseline=14, final=22, seed=3),
        baseline=14, title="환자 A — 평소 호흡수가 낮은 환자",
        deviating=True, font=font,
    )
    _panel(
        axes[1], hours=hours,
        series=_patient_series(baseline=21, final=22, seed=7),
        baseline=21, title="환자 B — 평소 호흡수가 높은 환자",
        deviating=False, font=font,
    )

    fig.suptitle(
        "같은 절대값(22회/분), 다른 의미 — 병동 공통 임계값은 두 환자 모두에게 경보한다",
        fontsize=13, fontweight="bold", fontfamily=font, color=INK, y=0.985,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.91), w_pad=3.0)
    fig.savefig(OUT, dpi=150, facecolor="white")
    plt.close(fig)
    print("Baseline concept:", OUT)
    return OUT


if __name__ == "__main__":
    main()
