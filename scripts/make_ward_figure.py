"""Draw the ward-scale alarm burden used in the proposal (§8.1).

"40% fewer alarms" is a ratio, and ratios do not land. The impact section
carried the ward arithmetic as a table; this is the same numbers as a picture,
because the point is how the saved alarms grow with ward size while detection
stays fixed.

Numbers come from the measured burden at matched 50% detection in §7-1 —
24.8 alarms per 100 windows for the model, 41.3 for NEWS — scaled to wards
evaluated once an hour.

Writes models/vitals_ward_burden.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "models" / "vitals_ward_burden.png"

INK = "#16222E"
MUTED = "#5A6472"
MODEL = "#2A78D6"
NEWS = "#EB6834"
SAVED = "#0F7A12"

# Alarms per 100 windows at matched 50% detection (§7-1, test 3,972 patients).
PER100_MODEL, PER100_NEWS = 24.8, 41.3
BEDS = (20, 30, 50)
EVALS_PER_BED_PER_DAY = 24  # hourly evaluation


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

    windows = np.array(BEDS) * EVALS_PER_BED_PER_DAY
    news = windows * PER100_NEWS / 100
    model = windows * PER100_MODEL / 100

    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    x = np.arange(len(BEDS))
    w = 0.34

    ax.bar(x - w / 2, news, w, label="NEWS (임상 표준)", color=NEWS, zorder=3)
    ax.bar(x + w / 2, model, w, label="본 모델", color=MODEL, zorder=3)

    for i, (n, m) in enumerate(zip(news, model)):
        ax.text(i - w / 2, n + 8, f"{n:.0f}", ha="center", va="bottom",
                fontsize=10, color=NEWS, fontweight="bold", fontfamily=font)
        ax.text(i + w / 2, m + 8, f"{m:.0f}", ha="center", va="bottom",
                fontsize=10, color=MODEL, fontweight="bold", fontfamily=font)
        # the gap is the point, so name it
        ax.annotate(
            "", xy=(i + w / 2, m), xytext=(i - w / 2, n),
            arrowprops=dict(arrowstyle="<->", color=SAVED, lw=1.4, ls=(0, (4, 2))),
            zorder=4,
        )
        ax.text(i, (n + m) / 2, f"{n - m:.0f}건 감소", ha="center", va="center",
                fontsize=11, color=SAVED, fontweight="bold", fontfamily=font,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=SAVED, lw=1.1),
                zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}병상" for b in BEDS], fontsize=11, fontfamily=font)
    ax.set_ylabel("일일 알람 건수", fontsize=10.5, color=MUTED, fontfamily=font)
    ax.set_ylim(0, max(news) * 1.22)
    ax.grid(axis="y", color="#DDE3EA", lw=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=9.5)
    ax.legend(prop={"family": font, "size": 10}, frameon=False, loc="upper left")

    fig.suptitle(
        "같은 검출률(50%)에서, 병동이 하루에 받는 알람",
        fontsize=13, fontweight="bold", color=INK, fontfamily=font, y=0.98,
    )
    ax.set_title(
        "검출률을 낮춰 줄인 것이 아니라, 같은 수의 위험 환자를 잡아내면서 줄어든 건수",
        fontsize=9.5, color=MUTED, fontfamily=font, pad=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT, dpi=150, facecolor="white")
    plt.close(fig)
    print("Ward burden:", OUT)
    return OUT


if __name__ == "__main__":
    main()
