"""Pytest configuration: expose the src package and shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture(scope="session")
def loader():
    """Session-scoped data loader pointed at the repository data directory.

    Skips dependent tests when the source datasets are not present (e.g. in CI,
    where the git-ignored ``data/`` archives are unavailable).
    """
    from data_loader import HeartFailureDataLoader

    instance = HeartFailureDataLoader(data_dir=PROJECT_ROOT / "data")
    try:
        instance.load_raw_data()
    except FileNotFoundError as exc:
        pytest.skip(f"Source datasets unavailable in data/: {exc}")
    return instance


@pytest.fixture(scope="session")
def split(loader):
    """Preprocessed stratified train/test split."""
    loader.preprocess()
    return loader.get_train_test_split()


@pytest.fixture(scope="session")
def trained_artifact(loader, split):
    """A lightweight LightGBM artifact for fast, self-contained API tests."""
    import lightgbm as lgb

    from train import _scale_pos_weight

    model = lgb.LGBMClassifier(
        n_estimators=60,
        learning_rate=0.1,
        num_leaves=16,
        random_state=42,
        verbosity=-1,
        scale_pos_weight=_scale_pos_weight(split.y_train),
    )
    model.fit(split.X_train, split.y_train)

    assert loader._processed is not None
    return {
        "model": model,
        "feature_names": split.feature_names,
        "scaler": loader._processed.scaler,
    }
