"""Data loading and preprocessing for heart failure risk prediction."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Dataset file patterns (case-insensitive).
HEART_FAILURE_ZIP: Final[str] = "Heart_failure_clinical_records_dataset.zip"
HEART_FAILURE_CSV: Final[str] = "heart_failure_clinical_records_dataset.csv"
CARDIO_ZIP: Final[str] = "Cardiovascular Disease dataset.zip"
CARDIO_CSV: Final[str] = "cardio_train.csv"

TARGET_COLUMN: Final[str] = "DEATH_EVENT"
CONTINUOUS_FEATURES: Final[tuple[str, ...]] = (
    "age",
    "creatinine_phosphokinase",
    "ejection_fraction",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "creatinine_ef_ratio",
)
BINARY_FEATURES: Final[tuple[str, ...]] = (
    "anaemia",
    "diabetes",
    "high_blood_pressure",
    "sex",
    "smoking",
)
CATEGORICAL_FEATURES: Final[tuple[str, ...]] = ("age_group",)

# OMOP CDM v5.4 concept identifiers.
GENDER_MALE_CONCEPT_ID: Final[int] = 8507
GENDER_FEMALE_CONCEPT_ID: Final[int] = 8532
OMOP_CREATININE_CONCEPT_ID: Final[int] = 3016723
OMOP_EJECTION_FRACTION_CONCEPT_ID: Final[int] = 4229813
OMOP_MEASUREMENT_CONCEPTS: Final[dict[str, int]] = {
    "creatinine_phosphokinase": 3007690,
    "ejection_fraction": OMOP_EJECTION_FRACTION_CONCEPT_ID,
    "platelets": 3000963,
    "serum_creatinine": OMOP_CREATININE_CONCEPT_ID,
    "serum_sodium": 3019550,
}


@dataclass
class OmopCdmTables:
    """OMOP CDM-compatible Person and Measurement tables."""

    person: pd.DataFrame
    measurement: pd.DataFrame


@dataclass
class ProcessedData:
    """Preprocessed features, target, and fitted scaler."""

    features: pd.DataFrame
    target: pd.Series
    feature_names: list[str]
    scaler: StandardScaler
    raw: pd.DataFrame = field(repr=False)


@dataclass
class SplitData:
    """Stratified train/test split container."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    scaler: StandardScaler


class HeartFailureDataLoader:
    """Load, preprocess, and export heart failure clinical data."""

    def __init__(self, data_dir: str | Path = "data") -> None:
        """Initialize loader with path to the data directory."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._heart_failure_raw: pd.DataFrame | None = None
        self._cardio_raw: pd.DataFrame | None = None
        self._processed: ProcessedData | None = None
        self._split: SplitData | None = None

    @staticmethod
    def _normalize(name: str) -> str:
        """Lowercase and collapse spaces/underscores for tolerant name matching."""
        return name.lower().replace(" ", "").replace("_", "").replace("-", "")

    def _find_file(self, name: str, suffix: str) -> Path | None:
        """Find a file by name (separator- and case-insensitive) and suffix."""
        target = self._normalize(name)
        for path in self.data_dir.rglob(f"*{suffix}"):
            candidate = self._normalize(path.name)
            if candidate == target or target in candidate or candidate in target:
                return path
        return None

    def _extract_zip(self, zip_path: Path) -> None:
        """Extract zip archive if CSV contents are not already present."""
        extract_dir = self.data_dir / zip_path.stem
        if extract_dir.exists() and any(extract_dir.rglob("*.csv")):
            return
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(extract_dir)

    def _load_csv_from_zip_or_disk(self, zip_name: str, csv_name: str, sep: str = ",") -> pd.DataFrame:
        """Resolve CSV from disk or extract from matching zip archive."""
        csv_path = self._find_file(csv_name, ".csv")
        if csv_path is None:
            zip_path = self._find_file(zip_name, ".zip")
            if zip_path is not None:
                self._extract_zip(zip_path)
                csv_path = self._find_file(csv_name, ".csv")
                if csv_path is None:
                    csv_path = next(self.data_dir.rglob("*.csv"), None)

        if csv_path is None:
            raise FileNotFoundError(
                f"Could not find '{csv_name}' or '{zip_name}' in '{self.data_dir}'."
            )
        return pd.read_csv(csv_path, sep=sep)

    def load_raw_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Safely unzip and load heart failure and cardiovascular datasets.

        Returns:
            Tuple of (heart_failure_df, cardiovascular_df).
        """
        hf_df = self._load_csv_from_zip_or_disk(HEART_FAILURE_ZIP, HEART_FAILURE_CSV)
        try:
            cardio_df = self._load_csv_from_zip_or_disk(CARDIO_ZIP, CARDIO_CSV, sep=";")
        except (pd.errors.ParserError, ValueError):
            cardio_df = self._load_csv_from_zip_or_disk(CARDIO_ZIP, CARDIO_CSV, sep=",")

        self._heart_failure_raw = hf_df
        self._cardio_raw = cardio_df
        return hf_df, cardio_df

    @staticmethod
    def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """Engineer derived features and handle missing values."""
        engineered = df.copy()

        # Impute missing values with column medians (continuous) or mode (binary).
        for column in engineered.columns:
            if engineered[column].isna().any():
                if engineered[column].dtype in ("float64", "int64", "float32", "int32"):
                    engineered[column] = engineered[column].fillna(engineered[column].median())
                else:
                    engineered[column] = engineered[column].fillna(engineered[column].mode().iloc[0])

        # creatinine_ef_ratio: serum creatinine relative to ejection fraction.
        ef_safe = engineered["ejection_fraction"].replace(0, np.nan).fillna(engineered["ejection_fraction"].median())
        engineered["creatinine_ef_ratio"] = engineered["serum_creatinine"] / ef_safe

        # age_group: binned age categories for non-linear age effects.
        engineered["age_group"] = pd.cut(
            engineered["age"],
            bins=[0, 50, 65, 80, 120],
            labels=["young", "middle", "senior", "elderly"],
            include_lowest=True,
        ).astype(str)

        return engineered

    def preprocess(self, df: pd.DataFrame | None = None) -> ProcessedData:
        """Preprocess heart failure data with scaling and feature engineering.

        Args:
            df: Optional raw dataframe; loads from disk when omitted.

        Returns:
            ProcessedData with scaled features and target series.
        """
        if df is None:
            if self._heart_failure_raw is None:
                df, _ = self.load_raw_data()
            else:
                df = self._heart_failure_raw
        else:
            self._heart_failure_raw = df

        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Missing target column '{TARGET_COLUMN}'.")

        engineered = self._engineer_features(df)
        target = engineered[TARGET_COLUMN].astype(int)

        feature_cols = list(CONTINUOUS_FEATURES) + list(BINARY_FEATURES) + list(CATEGORICAL_FEATURES)
        features = engineered[feature_cols].copy()

        # One-hot encode age_group; keep binary features unscaled.
        features = pd.get_dummies(features, columns=["age_group"], prefix="age_group", dtype=float)

        continuous_cols = [c for c in features.columns if c not in BINARY_FEATURES]
        scaler = StandardScaler()
        features[continuous_cols] = scaler.fit_transform(features[continuous_cols].astype(float))

        processed = ProcessedData(
            features=features,
            target=target,
            feature_names=list(features.columns),
            scaler=scaler,
            raw=engineered,
        )
        self._processed = processed
        return processed

    def get_train_test_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> SplitData:
        """Return stratified train/test split of preprocessed data."""
        if self._processed is None:
            self.preprocess()

        assert self._processed is not None
        X_train, X_test, y_train, y_test = train_test_split(
            self._processed.features.to_numpy(),
            self._processed.target.to_numpy(),
            test_size=test_size,
            random_state=random_state,
            stratify=self._processed.target,
        )

        split = SplitData(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=self._processed.feature_names,
            scaler=self._processed.scaler,
        )
        self._split = split
        return split

    def to_omop_cdm(self, df: pd.DataFrame | None = None) -> OmopCdmTables:
        """Map clinical records to OMOP CDM v5.4 Person and Measurement tables.

        Args:
            df: Raw heart failure dataframe; uses cached raw data when omitted.

        Returns:
            OmopCdmTables with Person and Measurement dataframes.
        """
        if df is None:
            if self._heart_failure_raw is None:
                df, _ = self.load_raw_data()
            else:
                df = self._heart_failure_raw

        person_rows: list[dict[str, Any]] = []
        measurement_rows: list[dict[str, Any]] = []
        measurement_datetime = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        reference_year = pd.Timestamp.today().year

        for idx, row in df.reset_index(drop=True).iterrows():
            person_id = int(idx + 1)
            gender_concept_id = GENDER_MALE_CONCEPT_ID if int(row["sex"]) == 1 else GENDER_FEMALE_CONCEPT_ID

            person_rows.append(
                {
                    "person_id": person_id,
                    "gender_concept_id": gender_concept_id,
                    "year_of_birth": int(reference_year - int(row["age"])),
                }
            )

            for feature_name, concept_id in OMOP_MEASUREMENT_CONCEPTS.items():
                value = row.get(feature_name)
                if pd.isna(value):
                    continue
                measurement_rows.append(
                    {
                        "person_id": person_id,
                        "measurement_concept_id": concept_id,
                        "measurement_datetime": measurement_datetime,
                        "value_as_number": float(value),
                    }
                )

        return OmopCdmTables(
            person=pd.DataFrame(person_rows),
            measurement=pd.DataFrame(measurement_rows),
        )


def _class_distribution(target: pd.Series | np.ndarray, label: str) -> str:
    """Format class distribution summary string."""
    series = pd.Series(target)
    counts = series.value_counts().sort_index()
    total = len(series)
    lines = [f"{label} (n={total})"]
    for cls, count in counts.items():
        pct = 100.0 * count / total
        lines.append(f"  class {cls}: {int(count)} ({pct:.1f}%)")
    return "\n".join(lines)


def main() -> None:
    """Run data loading, preprocessing, and split verification."""
    project_root = Path(__file__).resolve().parent.parent
    loader = HeartFailureDataLoader(data_dir=project_root / "data")

    hf_raw, cardio_raw = loader.load_raw_data()
    print(f"Heart failure raw shape: {hf_raw.shape}")
    print(f"Cardiovascular raw shape: {cardio_raw.shape}")

    processed = loader.preprocess(hf_raw)
    print(f"\nProcessed feature shape: {processed.features.shape}")
    print(f"Feature columns: {processed.feature_names}")
    print(_class_distribution(processed.target, "Full dataset target"))

    split = loader.get_train_test_split()
    print(f"\nTrain shape: {split.X_train.shape}, Test shape: {split.X_test.shape}")
    print(_class_distribution(split.y_train, "Train target"))
    print(_class_distribution(split.y_test, "Test target"))

    omop = loader.to_omop_cdm(hf_raw)
    print(f"\nOMOP Person rows: {len(omop.person)}")
    print(f"OMOP Measurement rows: {len(omop.measurement)}")
    print("\nData loader verification completed successfully.")


if __name__ == "__main__":
    main()
