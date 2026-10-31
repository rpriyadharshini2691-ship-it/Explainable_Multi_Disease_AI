import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data_validation import DatasetValidationError
from src.preprocessing import (
    FEATURE_SCHEMAS,
    build_preprocessor,
    build_training_pipeline,
    identify_feature_types,
    load_dataset,
    transformed_feature_mapping,
)


def test_load_dataset_normalizes_columns_and_removes_duplicates_and_missing_targets(
    tmp_path,
):
    source = pd.DataFrame(
        {
            "Age": [25, 25, 40, 50],
            "Family History": [" no ", " no ", "yes", "no"],
            "Target": [0, 0, pd.NA, 1],
        }
    )
    csv_path = tmp_path / "sample.csv"
    source.to_csv(csv_path, index=False)

    dataset = load_dataset(csv_path, target_column="Target")

    assert dataset.features.columns.tolist() == ["age", "family_history"]
    assert dataset.target.tolist() == [0, 1]
    assert dataset.features["family_history"].tolist() == ["no", "no"]
    assert dataset.report["duplicate_rows_removed"] == 1
    assert dataset.report["missing_target_rows_removed"] == 1
    assert dataset.feature_name_mapping["family_history"] == "Family History"


def test_load_dataset_rejects_non_binary_targets(tmp_path):
    csv_path = tmp_path / "invalid_target.csv"
    pd.DataFrame({"feature": [1, 2], "target": [0, 2]}).to_csv(
        csv_path, index=False
    )

    with pytest.raises(DatasetValidationError, match="binary 0/1"):
        load_dataset(csv_path)


def test_feature_schema_treats_numeric_codes_as_categories():
    heart_schema = FEATURE_SCHEMAS["heart"]
    features = pd.DataFrame(columns=[*heart_schema.numeric, *heart_schema.categorical])

    numeric, categorical = identify_feature_types(features, "heart")

    assert "age" in numeric
    assert "cp" in categorical
    assert "ca" in categorical


def test_preprocessor_imputes_scales_and_ignores_unseen_categories():
    features = pd.DataFrame(
        {
            "measurement": [1.0, np.nan, 3.0, 4.0],
            "category": ["a", "b", None, "new"],
        }
    )
    preprocessor = build_preprocessor(features.iloc[:3])
    train_matrix = preprocessor.fit_transform(features.iloc[:3])
    unseen_matrix = preprocessor.transform(features.iloc[3:])
    feature_mapping = transformed_feature_mapping(preprocessor)

    assert train_matrix.shape[1] == unseen_matrix.shape[1]
    assert train_matrix.shape[1] == len(feature_mapping)
    assert np.isfinite(train_matrix).all()
    assert np.isfinite(unseen_matrix).all()


def test_training_pipeline_contains_preprocessing_and_predicts_probabilities():
    features = pd.DataFrame(
        {
            "measurement": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "category": ["a", "a", "b", "b", "a", "b"],
        }
    )
    target = pd.Series([0, 0, 0, 1, 1, 1])
    pipeline = build_training_pipeline(LogisticRegression(), features)

    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps

    pipeline.fit(features, target)
    probabilities = pipeline.predict_proba(features)

    assert probabilities.shape == (len(features), 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()