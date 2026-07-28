"""Pytest root configuration: put the import roots on ``sys.path``.

Neither ``src/`` nor ``legacy/`` is an installed package — modules inside each
import their siblings by bare name (``from vitals_data import ...``). Tests
therefore need both directories on the path before collection.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

for import_root in (PROJECT_ROOT / "src", PROJECT_ROOT / "legacy"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))
