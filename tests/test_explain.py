import numpy as np
import pytest

from src.explain import normalize_shap_outputs, select_model_from_cv


def test_normalize_shap_outputs_selects_positive_class_from_three_dimensional_values():
    raw_values = np.asarray(
        [
            [[1.0, 10.0], [2.0, 20.0]],
            [[3.0, 30.0], [4.0, 40.0]],
        ]
    )
    raw_base_values = np.asarray([0.25, 0.75])

    values, base_values = normalize_shap_outputs(
        raw_values,
        raw_base_values,
        classes=np.asarray([0, 1]),
        row_count=2,
        feature_count=2,
    )

    assert values.tolist() == [[10.0, 20.0], [30.0, 40.0]]
    assert base_values.tolist() == [0.75, 0.75]


def test_normalize_shap_outputs_supports_legacy_list_output():
    raw_values = [np.asarray([[1.0, 2.0]]), np.asarray([[3.0, 4.0]])]

    values, base_values = normalize_shap_outputs(
        raw_values,
        [0.1, 0.9],
        classes=np.asarray([0, 1]),
        row_count=1,
        feature_count=2,
    )

    assert values.tolist() == [[3.0, 4.0]]
    assert base_values.tolist() == [0.9]


def test_select_model_uses_training_cv_roc_auc_only():
    training_report = {
        "models": {
            "logistic_regression": {
                "cross_validation": {"metrics": {"roc_auc": {"mean": 0.9, "std": 0.01}}}
            },
            "xgboost": {
                "cross_validation": {"metrics": {"roc_auc": {"mean": 0.9, "std": 0.02}}}
            },
        }
    }

    selected, score = select_model_from_cv(training_report)

    assert selected == "logistic_regression"
    assert score == {"mean": 0.9, "std": 0.01}


def test_normalize_shap_outputs_rejects_unexpected_dimensions():
    with pytest.raises(ValueError, match="Unexpected normalized SHAP value shape"):
        normalize_shap_outputs(
            np.ones((2, 3)),
            0.0,
            classes=np.asarray([0, 1]),
            row_count=2,
            feature_count=2,
        )