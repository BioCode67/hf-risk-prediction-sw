"""Fixtures for the legacy static heart-failure pipeline.

These all depend on the source archives in ``data/``, which are git-ignored, so
every test using them skips automatically on a fresh clone. Import roots are set
up by the repository-root ``conftest.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
