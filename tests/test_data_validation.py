import pandas as pd
import pytest

from src.data_validation import (
    DatasetValidationError,
    clean_missing_markers,
    map_binary_target,
    normalize_column_names,
)


def test_diabetes_target_mapping_preserves_missing_labels():
    mapped = map_binary_target(
        pd.Series(["Positive", "Negative", pd.NA]),
        "diabetes",
    )

    assert mapped.tolist() == [1, 0, pd.NA]


def test_heart_target_mapping_collapses_grades_one_through_four():
    mapped = map_binary_target(
        pd.Series([0, 1, 2, 3, 4, pd.NA]),
        "heart",
    )

    assert mapped.tolist() == [0, 1, 1, 1, 1, pd.NA]


def test_kidney_target_mapping_trims_whitespace_and_preserves_missing():
    mapped = map_binary_target(
        pd.Series(["ckd", "notckd", " ckd\t", pd.NA]),
        "kidney",
    )

    assert mapped.tolist() == [1, 0, 1, pd.NA]


def test_missing_marker_cleanup_trims_values_and_normalizes_markers():
    frame = pd.DataFrame({"category": [" no ", "?", "N/A", ""]})

    cleaned = clean_missing_markers(frame)

    assert cleaned["category"].iloc[0] == "no"
    assert cleaned["category"].iloc[1:].isna().all()


def test_column_normalization_rejects_colliding_names():
    frame = pd.DataFrame([[1, 2]], columns=["Heart Rate", "heart-rate"])

    with pytest.raises(DatasetValidationError, match="duplicate names"):
        normalize_column_names(frame)


def test_target_mapping_rejects_unexpected_labels():
    with pytest.raises(DatasetValidationError, match="unexpected labels"):
        map_binary_target(pd.Series(["maybe"]), "diabetes")