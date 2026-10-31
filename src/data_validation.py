"""Shared validation and label-mapping helpers for UCI source datasets."""

from __future__ import annotations

import re

import pandas as pd


class DatasetValidationError(ValueError):
    """Raised when source data does not match the expected dataset contract."""


MISSING_MARKERS = {"", "?", "na", "n/a"}


def normalize_column_name(column_name: object) -> str:
    """Convert a source column label to a stable lowercase snake-case name."""
    normalized = re.sub(r"[^0-9a-z]+", "_", str(column_name).strip().casefold())
    return normalized.strip("_")


def normalize_column_names(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Normalize frame columns and return the original-to-normalized mapping."""
    original_names = [str(column) for column in frame.columns]
    normalized_names = [normalize_column_name(column) for column in frame.columns]
    if len(set(normalized_names)) != len(normalized_names):
        raise DatasetValidationError(
            "Column-name normalization produced duplicate names: "
            f"{normalized_names}"
        )

    normalized_frame = frame.copy()
    normalized_frame.columns = normalized_names
    name_mapping = dict(zip(original_names, normalized_names, strict=True))
    return normalized_frame, name_mapping


def clean_missing_markers(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace common textual missing markers with pandas missing values."""

    def clean_value(value: object) -> object:
        if isinstance(value, str):
            value = value.strip()
            if value.casefold() in MISSING_MARKERS:
                return pd.NA
        return value

    return frame.apply(lambda column: column.map(clean_value))


def map_binary_target(labels: pd.Series, disease: str) -> pd.Series:
    """Map verified source labels to 0/1 while retaining missing labels as NA."""
    mapped = pd.Series(pd.NA, index=labels.index, dtype="Int64", name="target")
    present = labels.notna()

    if disease == "heart":
        numeric_labels = pd.to_numeric(labels.loc[present], errors="coerce")
        invalid = numeric_labels.isna()
        if invalid.any():
            unexpected = labels.loc[present].loc[invalid].astype(str).unique().tolist()
            raise DatasetValidationError(
                f"Heart target contains non-numeric labels: {unexpected}"
            )
        allowed = numeric_labels.isin([0, 1, 2, 3, 4])
        if not allowed.all():
            unexpected = numeric_labels.loc[~allowed].unique().tolist()
            raise DatasetValidationError(
                f"Heart target labels must be in 0..4; found {unexpected}"
            )
        mapped.loc[present] = numeric_labels.gt(0).astype("int64")
        return mapped

    mappings = {
        "diabetes": {"positive": 1, "negative": 0},
        "kidney": {"ckd": 1, "notckd": 0},
    }
    if disease not in mappings:
        raise DatasetValidationError(f"Unknown target mapping for {disease!r}")

    normalized_labels = labels.loc[present].astype(str).str.strip().str.casefold()
    unexpected = sorted(set(normalized_labels) - set(mappings[disease]))
    if unexpected:
        raise DatasetValidationError(
            f"{disease.title()} target contains unexpected labels: {unexpected}"
        )
    mapped.loc[present] = normalized_labels.map(mappings[disease]).astype("int64")
    return mapped


def value_counts_as_text(values: pd.Series) -> dict[str, int]:
    """Return a JSON-safe distribution, preserving a visible missing bucket."""
    counts = values.value_counts(dropna=False, sort=False)
    return {
        "<missing>" if pd.isna(label) else str(label): int(count)
        for label, count in counts.items()
    }