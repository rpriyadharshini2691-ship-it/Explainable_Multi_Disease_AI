"""Leakage-safe preprocessing for the disease-specific feature schemas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_validation import (
    DatasetValidationError,
    clean_missing_markers,
    normalize_column_names,
)


@dataclass(frozen=True)
class FeatureSchema:
    """Explicit feature types, including numeric-looking coded categories."""

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]


@dataclass
class LoadedDataset:
    """Clean model inputs and a compact record of pre-split data cleaning."""

    features: pd.DataFrame
    target: pd.Series
    feature_name_mapping: dict[str, str]
    report: dict[str, Any]


FEATURE_SCHEMAS: dict[str, FeatureSchema] = {
    "diabetes": FeatureSchema(
        numeric=("age",),
        categorical=(
            "gender",
            "polyuria",
            "polydipsia",
            "sudden_weight_loss",
            "weakness",
            "polyphagia",
            "genital_thrush",
            "visual_blurring",
            "itching",
            "irritability",
            "delayed_healing",
            "partial_paresis",
            "muscle_stiffness",
            "alopecia",
            "obesity",
        ),
    ),
    "heart": FeatureSchema(
        numeric=("age", "trestbps", "chol", "thalach", "oldpeak"),
        categorical=("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"),
    ),
    "kidney": FeatureSchema(
        numeric=("age", "bp", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc"),
        categorical=(
            "sg",
            "al",
            "su",
            "rbc",
            "pc",
            "pcc",
            "ba",
            "htn",
            "dm",
            "cad",
            "appet",
            "pe",
            "ane",
        ),
    ),
}


def load_dataset(
    csv_path: str | Path,
    target_column: str = "target",
) -> LoadedDataset:
    """Load a dataset, normalize fields, remove exact duplicates, and validate target."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset CSV does not exist: {path}")

    source_frame = pd.read_csv(path)
    frame, name_mapping = normalize_column_names(source_frame)
    target_name = name_mapping.get(target_column, target_column)
    if target_name not in frame.columns:
        raise DatasetValidationError(
            f"Target column {target_column!r} is missing from {path}."
        )

    frame = clean_missing_markers(frame)
    duplicate_count = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().reset_index(drop=True)
    missing_target_count = int(frame[target_name].isna().sum())
    frame = frame.dropna(subset=[target_name]).reset_index(drop=True)

    target_values = pd.to_numeric(frame[target_name], errors="coerce")
    if target_values.isna().any():
        raise DatasetValidationError(
            f"Target column {target_name!r} contains non-numeric or invalid labels."
        )
    observed_targets = set(target_values.unique().tolist())
    if not observed_targets.issubset({0, 1}):
        raise DatasetValidationError(
            f"Target column must contain only binary 0/1 labels; found "
            f"{sorted(observed_targets)}."
        )
    if observed_targets != {0, 1}:
        raise DatasetValidationError(
            "Both binary target classes must remain after cleaning; found "
            f"{sorted(observed_targets)}."
        )

    target = target_values.astype("int64").rename(target_name)
    features = frame.drop(columns=[target_name])
    if features.empty or features.shape[1] == 0:
        raise DatasetValidationError("No feature columns remain after loading.")

    reverse_name_mapping = {
        normalized: original for original, normalized in name_mapping.items()
    }
    missing_counts = {column: int(features[column].isna().sum()) for column in features}
    report = {
        "source_path": str(path),
        "source_shape": [int(source_frame.shape[0]), int(source_frame.shape[1])],
        "cleaned_shape": [int(features.shape[0]), int(features.shape[1] + 1)],
        "target_column": target_name,
        "target_labels": {
            str(label): int(count)
            for label, count in target.value_counts(sort=False).items()
        },
        "duplicate_rows_removed": duplicate_count,
        "missing_target_rows_removed": missing_target_count,
        "feature_missing_value_counts": missing_counts,
    }
    return LoadedDataset(
        features=features,
        target=target,
        feature_name_mapping=reverse_name_mapping,
        report=report,
    )


def identify_feature_types(
    features: pd.DataFrame,
    disease: str | None = None,
) -> tuple[list[str], list[str]]:
    """Return numeric and categorical columns using a disease schema when known."""
    if disease is not None:
        if disease not in FEATURE_SCHEMAS:
            raise DatasetValidationError(f"Unknown disease schema: {disease!r}")
        schema = FEATURE_SCHEMAS[disease]
        expected = set(schema.numeric) | set(schema.categorical)
        actual = set(features.columns)
        if expected != actual:
            raise DatasetValidationError(
                f"Feature columns for {disease} do not match its schema. "
                f"Missing: {sorted(expected - actual)}; "
                f"unexpected: {sorted(actual - expected)}."
            )
        return list(schema.numeric), list(schema.categorical)

    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    return numeric, categorical


def build_preprocessor(
    features: pd.DataFrame,
    disease: str | None = None,
) -> ColumnTransformer:
    """Build imputing/scaling/encoding transformers for use inside an ML pipeline."""
    numeric_features, categorical_features = identify_feature_types(features, disease)

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )
    if not transformers:
        raise DatasetValidationError("No features were provided to the preprocessor.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
        sparse_threshold=0,
    )


def build_training_pipeline(
    estimator: Any,
    features: pd.DataFrame,
    disease: str | None = None,
) -> Pipeline:
    """Keep every learned transform inside the estimator passed to CV/model fitting."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(features, disease)),
            ("model", estimator),
        ]
    )


def transformed_feature_mapping(
    fitted_preprocessor: ColumnTransformer,
) -> dict[str, str]:
    """Map transformed feature names back to their original source feature."""
    feature_mapping: dict[str, str] = {}
    for transformer_name, transformer, columns in fitted_preprocessor.transformers_:
        if transformer_name == "remainder" or transformer == "drop":
            continue
        source_columns = [str(column) for column in columns]
        if transformer_name == "numeric":
            for column in source_columns:
                feature_mapping[f"numeric__{column}"] = column
            continue
        if transformer_name == "categorical":
            encoder = transformer.named_steps["onehot"]
            encoded_names = encoder.get_feature_names_out(source_columns)
            encoded_position = 0
            for source_column, categories in zip(
                source_columns, encoder.categories_, strict=True
            ):
                category_count = len(categories)
                for encoded_name in encoded_names[
                    encoded_position : encoded_position + category_count
                ]:
                    feature_mapping[f"categorical__{encoded_name}"] = source_column
                encoded_position += category_count
    return feature_mapping