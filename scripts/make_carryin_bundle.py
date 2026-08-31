"""Build the 안심존 carry-in code snapshot, exactly as the pre-declaration promises.

The 사전신청 회신 declared: a pure-Python code ZIP of about 1MB with every
network-calling module removed, plus the NanumGothic TTFs as a *separate*
carry-in item. This script produces that ZIP and proves it is self-sufficient:

1. Stage the active-track code, with ``vitals_narrate.py`` stripped of its LLM
   half (Groq URL, API-key handling, the ``requests`` call). A ``narrate()``
   stub stays behind so importers keep working — it preserves the DUA guard
   (PermissionError for mimic/khth) and otherwise explains the removal.
2. Leave the font TTFs out (declared separately, ~10MB) but keep their OFL
   license and a note saying where to place them; ``plot_fonts`` degrades
   gracefully until they arrive.
3. Run the full test suite *inside the staged copy* — it must be green with
   no data and no network, because that is exactly the first-visit check.
4. Zip it and enforce the declared size bound.

Usage: python scripts/make_carryin_bundle.py [--skip-tests]
Writes dist/PRODROME_carryin_code.zip (git-ignored).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
STAGING = DIST / "carryin"
ZIP_PATH = DIST / "PRODROME_carryin_code.zip"

# Declared "약 1MB 이내" — fail loudly before the declaration becomes false.
MAX_ZIP_BYTES = int(1.2 * 1024 * 1024)

# torch 2.13.0 is in the declared package list, so the DL benchmark arm
# (src/utils|dataset|model.py, train_dl.py, tests/test_torch.py) ships too —
# its tests importorskip torch, so the bundle stays green either way.
# Excluded scripts either call the network or build the proposal, which
# stays outside the zone.
SRC_EXCLUDE: set[str] = set()
TEST_EXCLUDE: set[str] = set()
SCRIPTS_INCLUDE = {"ablate_personalized.py"}

NARRATE_STUB = '''

def narrate(evidence, **_kwargs):
    """Carry-in stub — the LLM half is removed for the offline 안심존.

    The DUA guard is kept: credentialed cohorts must never reach a
    third-party API, offline or not.
    """
    source = evidence.get("source", "unknown")
    if source in RESTRICTED_SOURCES:
        raise PermissionError(
            f"cohort source {source!r} is credentialed data and must not be sent to an "
            "external API; use format_evidence() instead"
        )
    raise RuntimeError(
        "narrate()'s network half is not part of the carry-in copy; "
        "use format_evidence() for the deterministic Korean explanation"
    )
'''


def strip_narrate(source: str) -> str:
    """Remove the network half of vitals_narrate, keeping the offline half."""
    # The LLM configuration block sits between the imports and the
    # RESTRICTED_SOURCES constant.
    start = source.index("GROQ_URL")
    end = source.index("# Cohorts under")
    source = (
        source[:start]
        + "# (LLM configuration removed from the carry-in copy — offline zone.)\n"
        + source[end:]
    )

    # Everything from the LLM prompt onward is the network half; cut it and
    # append the stub so importers of `narrate` keep working.
    source = source[: source.index("\nSYSTEM_PROMPT")] + NARRATE_STUB

    for forbidden in ("requests", "groq", "GROQ", "api_key"):
        if forbidden in source:
            raise SystemExit(f"strip failed: {forbidden!r} still present in vitals_narrate")
    for required in ("def build_evidence", "def format_evidence", "def narrate", "RESTRICTED_SOURCES"):
        if required not in source:
            raise SystemExit(f"strip failed: {required!r} missing from stripped vitals_narrate")
    compile(source, "vitals_narrate.py", "exec")
    return source


def stage() -> None:
    if STAGING.exists():
        shutil.rmtree(STAGING)
    (STAGING / "src").mkdir(parents=True)
    (STAGING / "tests").mkdir()
    (STAGING / "scripts").mkdir()
    (STAGING / "assets" / "fonts").mkdir(parents=True)
    (STAGING / "models").mkdir()  # figure/artifact output target

    for path in sorted((ROOT / "src").glob("*.py")):
        if path.name in SRC_EXCLUDE:
            continue
        text = path.read_text(encoding="utf-8")
        if path.name == "vitals_narrate.py":
            text = strip_narrate(text)
        (STAGING / "src" / path.name).write_text(text, encoding="utf-8")
    shutil.copy(ROOT / "src" / "README.md", STAGING / "src" / "README.md")

    for path in sorted((ROOT / "tests").glob("*.py")):
        if path.name not in TEST_EXCLUDE:
            shutil.copy(path, STAGING / "tests" / path.name)

    for name in sorted(SCRIPTS_INCLUDE):
        shutil.copy(ROOT / "scripts" / name, STAGING / "scripts" / name)

    for name in ("conftest.py", "requirements.txt", "requirements-torch.txt", "README.md", "train_dl.py"):
        shutil.copy(ROOT / name, STAGING / name)

    shutil.copy(ROOT / "assets" / "fonts" / "LICENSE-OFL.txt", STAGING / "assets" / "fonts" / "LICENSE-OFL.txt")
    (STAGING / "assets" / "fonts" / "PLACE_FONTS_HERE.md").write_text(
        "나눔고딕 TTF(별도 반입 매체)를 이 폴더에 두면 plot_fonts가 자동 등록합니다.\n"
        "파일: NanumGothic.ttf, NanumGothicBold.ttf (SIL OFL — LICENSE-OFL.txt 참조)\n",
        encoding="utf-8",
    )

    (STAGING / "BUNDLE_README.md").write_text(
        "# PRODROME 반입 코드 스냅샷\n\n"
        "2026 K-Health 경진대회 본선 안심존 반입용 코드입니다. 사전신청서의 선언대로\n"
        "외부 네트워크를 호출하는 코드는 포함되어 있지 않습니다\n"
        "(`src/vitals_narrate.py`의 LLM 호출부 제거 — 결정론적 근거 생성부는 유지).\n\n"
        "## 반입 직후 검증 (데이터·네트워크 불필요)\n\n"
        "```bash\n"
        "python -m pytest -q   # 데이터 없이 green이어야 정상\n"
        "python src/vitals_train.py    # 내장 합성 코호트로 전체 파이프라인 확인\n"
        "```\n\n"
        "## 구성\n\n"
        "- `src/` — 시계열 조기경보 파이프라인 (numpy·pandas·scikit-learn·xgboost·shap·matplotlib)\n"
        "- `tests/` + `conftest.py` — 자동화 테스트 (데이터·torch 의존 테스트는 자동 skip)\n"
        "- `scripts/ablate_personalized.py` — 개인화 피처 통제 실험 (다중 시드)\n"
        "- `train_dl.py` + `src/utils|dataset|model.py` — 딥러닝(LSTM/GRU) 비교 실험\n"
        "  (사전 신고한 torch 2.13.0, CPU 모드 — `python train_dl.py`가 더미 데이터 자가검증)\n"
        "- `assets/fonts/` — 나눔고딕 TTF는 별도 매체로 반입하여 이 폴더에 배치\n"
        "- 사전학습 모델(XGBoost JSON + config JSON)은 별도 반입 —\n"
        "  `vitals_train.load_artifact(\"<모델>.json\")` 으로 적재\n",
        encoding="utf-8",
    )


def verify() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=STAGING,
        capture_output=True,
        text=True,
        timeout=900,
    )
    tail = "\n".join(result.stdout.strip().splitlines()[-3:])
    print(f"staged pytest:\n{tail}")
    if result.returncode != 0:
        print(result.stdout[-3000:])
        raise SystemExit("staged test suite failed — the bundle is not self-sufficient")


def build_zip() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(STAGING.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and ".pytest_cache" not in path.parts:
                zf.write(path, path.relative_to(STAGING))
    size = ZIP_PATH.stat().st_size
    print(f"bundle: {ZIP_PATH} ({size / 1024:.0f} KB)")
    if size > MAX_ZIP_BYTES:
        raise SystemExit(
            f"bundle is {size} bytes — over the declared ~1MB bound; trim before carrying in"
        )


def main() -> None:
    stage()
    if "--skip-tests" not in sys.argv:
        verify()
    build_zip()


if __name__ == "__main__":
    main()
