# -*- coding: utf-8 -*-
"""Test fixtures: run every test from the repository root.

Several tests read the shipped ``configs/*.yaml`` by relative path, and the
package itself is imported from the root, so pytest must not depend on where
it was invoked from.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _run_from_repo_root(monkeypatch):
    monkeypatch.chdir(ROOT)


if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep the suite single-threaded: BLAS thread pools fight each other when
# dozens of small logistic regressions are fitted back to back.
os.environ.setdefault("OMP_NUM_THREADS", "1")
