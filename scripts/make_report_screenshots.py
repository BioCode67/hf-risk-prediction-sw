"""Render the weekly-report screenshots (code cards, console cards, figures).

The weekly progress report needs a couple of images per week, and re-cropping a
terminal by hand every week is the sort of thing that quietly stops happening.
This bakes the whole set from the repo itself: source lines are syntax-
highlighted into a code card, scripts are executed and their real stdout framed
in a console card, and the figures already produced by ``vitals_report`` /
``vitals_explain`` are copied in.

    pip install playwright pygments        # not in requirements.txt (dev-only)
    python scripts/make_report_screenshots.py

Everything runs on the built-in synthetic cohort, so no restricted data is
touched and nothing here can leak PHI. Output lands in
``reports/screenshots/``.

The dashboard shot is separate — it needs the Next.js app running — and is
taken by ``--dashboard`` once ``npm run dev`` is up on :3000 (see web/README.md
for the export step that fills ``web/public/data``).
"""

from __future__ import annotations

import argparse
import html as html_mod
import shutil
import subprocess
import sys
from pathlib import Path

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "reports" / "screenshots"

# Cohort-summary program, run for the week-1 console card. Kept inline rather
# than as a second file: it exists only to be screenshotted.
EXPLORE_SOURCE = '''
import sys
sys.path.insert(0, "SRC_DIR")
from vitals_data import (
    VITALS, add_personalized_features, build_windows,
    generate_synthetic_cohort, patient_level_split, sanitize_vitals,
)

cohort = generate_synthetic_cohort(seed=42)
v = cohort.vitals
bar = "=" * 64

print(bar); print(" 1. Cohort overview"); print(bar)
print(f"  patients            : {v['patient_id'].nunique()}")
print(f"  arrest events       : {len(cohort.events)}")
print(f"  vital-sign rows     : {len(v):,}")
print(f"  hours per patient   : {v.groupby('patient_id').size().median():.0f} (median)")
print()

print(bar); print(" 2. Missingness / plausibility per vital"); print(bar)
clean = sanitize_vitals(v)
print(f"  {'vital':<14}{'missing%':>10}{'artefact%':>11}{'mean':>9}{'std':>8}")
for name in VITALS:
    miss = v[name].isna().mean() * 100
    art = (clean[name].isna().mean() - v[name].isna().mean()) * 100
    print(f"  {name:<14}{miss:>9.1f}%{art:>10.1f}%{clean[name].mean():>9.1f}{clean[name].std():>8.1f}")
print()

print(bar); print(" 3. Sliding windows (6h observation / 1h gap / 2h horizon)"); print(bar)
w = add_personalized_features(build_windows(cohort), cohort)
pos = int(w.labels.sum())
print(f"  windows             : {len(w.labels):,}")
print(f"  features per window : {len(w.feature_names)}")
print(f"  positives           : {pos} ({pos / len(w.labels) * 100:.2f}%)")
print(f"  negatives           : {len(w.labels) - pos}")
print(f"  class imbalance     : 1 : {(len(w.labels) - pos) / pos:.0f}  -> scale_pos_weight")
print()

print(bar); print(" 4. Patient-level split (no patient in both sides)"); print(bar)
s = patient_level_split(w, test_size=0.2, seed=42)
print(f"  train windows       : {len(s.y_train):,}  (positives {int(s.y_train.sum())})")
print(f"  test  windows       : {len(s.y_test):,}  (positives {int(s.y_test.sum())})")
print(f"  patient overlap     : {len(set(s.groups_train) & set(s.groups_test))}  (leakage check)")
'''

CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; padding: 0; background: #0b1020;
       font-family: ui-sans-serif, system-ui, "Noto Sans KR", sans-serif; }
#wrap { display: inline-block; padding: 28px; background: #0b1020; }
.card { background: #141a2e; border: 1px solid #263050; border-radius: 12px;
        overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,.45); }
.bar { display: flex; align-items: center; gap: 8px; padding: 10px 14px;
       background: #1b2340; border-bottom: 1px solid #263050; }
.dot { width: 11px; height: 11px; border-radius: 50%; }
.name { margin-left: 8px; color: #cbd5f5; font-size: 13px;
        font-family: ui-monospace, Menlo, monospace; }
.tag { margin-left: auto; padding-left: 32px; color: #8fa3d8; font-size: 12px; }
.body { padding: 16px 18px; }
pre { margin: 0; font-family: ui-monospace, Menlo, Consolas, monospace;
      font-size: 13.5px; line-height: 1.55; color: #e6ecff; white-space: pre; }
.caption { margin-top: 14px; color: #93a4d4; font-size: 12.5px; text-align: right; }
.hll { background: #2a3355 }
.c, .c1, .cm, .cs, .ch, .cpf { color: #7c89b8; font-style: italic }
.k, .kn, .kc, .kd, .kp, .kr, .kt { color: #c792ea }
.s, .s1, .s2, .sd, .sb, .sc, .se, .sh, .si, .sx, .sr, .ss { color: #c3e88d }
.nb, .nf, .bp { color: #82aaff }
.nc { color: #ffcb6b }
.nd, .mi, .mf, .mh, .mo, .il { color: #f78c6c }
.o, .ow { color: #89ddff }
.n, .p { color: #e6ecff }
"""


def chromium_path() -> str | None:
    """Locate the pre-installed Chromium, letting Playwright resolve it if absent."""
    roots = sorted(Path("/opt/pw-browsers").glob("chromium-*/chrome-linux/chrome"))
    return str(roots[-1]) if roots else None


def _card(name: str, tag: str, inner: str, caption: str) -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head>
<body><div id="wrap"><div class="card">
  <div class="bar">
    <span class="dot" style="background:#ff5f57"></span>
    <span class="dot" style="background:#febc2e"></span>
    <span class="dot" style="background:#28c840"></span>
    <span class="name">{html_mod.escape(name)}</span>
    <span class="tag">{html_mod.escape(tag)}</span>
  </div>
  <div class="body">{inner}</div>
</div>
<div class="caption">{html_mod.escape(caption)}</div></div>
</body></html>"""


def code_card(path: str, start: int, end: int, tag: str, caption: str) -> str:
    """Syntax-highlight ``path`` lines ``[start, end]`` into a card, with gutter."""
    code = "\n".join((REPO / path).read_text().splitlines()[start - 1 : end])
    body = highlight(code, PythonLexer(), HtmlFormatter(nowrap=True))
    gutter = [
        f'<span style="color:#4a5680">{i:>4}</span>  {line}'
        for i, line in enumerate(body.split("\n"), start=start)
    ]
    return _card(path, tag, "<pre>" + "\n".join(gutter) + "</pre>", caption)


def console_card(command: str, text: str, tag: str, caption: str) -> str:
    return _card(command, tag, f'<pre>{html_mod.escape(text)}</pre>', caption)


def run(*args: str) -> str:
    """Run a project script from the repo root and return its stdout."""
    done = subprocess.run(args, cwd=REPO, capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"{' '.join(args)} failed:\n{done.stderr}")
    return done.stdout.rstrip()


def render(pages: list[tuple[str, str]]) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chromium_path())
        page = browser.new_page(viewport={"width": 1200, "height": 900}, device_scale_factor=2)
        for filename, document in pages:
            page.set_content(document)
            page.wait_for_timeout(150)
            page.locator("#wrap").screenshot(path=str(OUT / filename))
            print("wrote", (OUT / filename).relative_to(REPO))
        browser.close()


def shoot_dashboard(url: str) -> None:
    """Screenshot the running dashboard with its first patient opened."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chromium_path())
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=2)
        # Not ``networkidle``: the dev server holds an HMR socket open forever.
        page.goto(url, wait_until="domcontentloaded", timeout=120_000)
        page.wait_for_timeout(6000)
        rows = page.locator("button, [role='button']")
        for i in range(rows.count()):
            if rows.nth(i).inner_text().strip().isdigit():
                rows.nth(i).click()
                break
        page.wait_for_timeout(5000)
        page.screenshot(path=str(OUT / "week4_2_dashboard.png"), full_page=True)
        print("wrote", (OUT / "week4_2_dashboard.png").relative_to(REPO))
        browser.close()


def build_static() -> None:
    explore_py = OUT / "_explore.py"
    explore_py.write_text(EXPLORE_SOURCE.replace("SRC_DIR", str(REPO / "src")))
    try:
        explore = run(sys.executable, str(explore_py))
    finally:
        explore_py.unlink()

    train = run(sys.executable, "src/vitals_train.py")
    report = run(sys.executable, "src/vitals_report.py")
    explain = run(sys.executable, "src/vitals_explain.py")
    # Drop the four "figure written to ..." lines; keep the burden/lead-time numbers.
    evaluation = train + "\n\n" + "\n".join(report.splitlines()[4:])

    render(
        [
            (
                "week1_1_preprocessing_code.png",
                code_card("src/vitals_data.py", 271, 302, "1주차 · 전처리",
                          "생체신호 이상치(센서 단선·화씨 오기록) 정제 — src/vitals_data.py"),
            ),
            (
                "week1_2_data_exploration.png",
                console_card("$ python explore_summary.py", explore, "1주차 · 데이터 탐색",
                             "코호트 규모·결측 패턴·클래스 불균형·환자 단위 분할 누출 검증"),
            ),
            (
                "week2_1_rnn_model_code.png",
                code_card("src/model.py", 14, 65, "2주차 · 모델 설계",
                          "LSTM/GRU 시퀀스 분류기 (가변 길이 패딩 마스킹) — src/model.py"),
            ),
            (
                "week2_2_xgboost_code.png",
                code_card("src/vitals_train.py", 231, 259, "2주차 · 베이스라인 구현",
                          "비용민감 XGBoost 설정 (scale_pos_weight, AUPRC 최적화) — src/vitals_train.py"),
            ),
            (
                "week3_1_evaluation_output.png",
                console_card("$ python src/vitals_train.py", evaluation, "3주차 · 성능 평가",
                             "XGBoost vs NEWS: AUPRC / 특이도 95% 민감도 / 동일 민감도 알람 부담"),
            ),
            (
                "extra_shap_console.png",
                console_card("$ python src/vitals_explain.py", explain, "설명가능성 (참고)",
                             "SHAP 전역 기여도 상위 5개 + 최고위험 윈도우의 개별 근거 3개"),
            ),
        ]
    )

    figures = {
        "vitals_pr_curve.png": "week3_2_pr_curve.png",
        "vitals_shap_summary.png": "week4_1_shap_summary.png",
        "vitals_alarm_burden.png": "extra_alarm_burden.png",
    }
    for source, target in figures.items():
        shutil.copyfile(REPO / "models" / source, OUT / target)
        print("wrote", (OUT / target).relative_to(REPO))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard", action="store_true",
                        help="also screenshot the running Next.js dashboard")
    parser.add_argument("--url", default="http://localhost:3000",
                        help="dashboard URL (default: %(default)s)")
    parser.add_argument("--only-dashboard", action="store_true",
                        help="skip the code/console/figure cards")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if not args.only_dashboard:
        build_static()
    if args.dashboard or args.only_dashboard:
        shoot_dashboard(args.url)


if __name__ == "__main__":
    main()
