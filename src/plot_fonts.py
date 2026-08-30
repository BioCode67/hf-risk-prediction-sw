"""Korean font resolution for matplotlib figures, closed-network safe.

The 안심존 has no Korean system fonts and no way to install any, so the repo
snapshot carries NanumGothic in ``assets/fonts/`` (SIL OFL — see LICENSE-OFL.txt
there). ``ensure_korean_font()`` first looks for a Korean face matplotlib
already knows, then registers the bundled TTFs via ``fontManager.addfont`` —
a per-process, user-level registration needing no admin rights — and returns
the family name to pass as ``fontfamily``.
"""

from __future__ import annotations

from pathlib import Path

BUNDLED_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
KNOWN_FACES = ("NanumGothic", "NanumBarunGothic", "Malgun Gothic", "AppleGothic")


def ensure_korean_font() -> str:
    """Return a usable Korean font family name, registering bundled TTFs if needed."""
    import matplotlib.font_manager as fm

    available = {f.name for f in fm.fontManager.ttflist}
    for name in KNOWN_FACES:
        if name in available:
            return name

    if BUNDLED_DIR.is_dir():
        for ttf in sorted(BUNDLED_DIR.glob("*.ttf")):
            fm.fontManager.addfont(str(ttf))
        available = {f.name for f in fm.fontManager.ttflist}
        for name in KNOWN_FACES:
            if name in available:
                return name

    import matplotlib.pyplot as plt

    print("WARNING: no Korean font found — labels will render as boxes.")
    family = plt.rcParams["font.family"]
    return family[0] if isinstance(family, list) else family
