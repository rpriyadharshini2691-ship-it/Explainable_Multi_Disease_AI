import numpy as np
import pytest

from src.evaluate import compute_classification_metrics, select_youden_threshold


def test_youden_threshold_is_selected_from_supplied_validation_predictions():
    target = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.1, 0.4, 0.35, 0.8])

    threshold, youden_j = select_youden_threshold(target, probabilities)

    assert threshold == pytest.approx(0.35)
    assert youden_j == pytest.approx(0.5)


def test_metrics_include_sensitivity_specificity_false_negative_rate_and_brier():
    target = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.1, 0.7, 0.6, 0.9])

    metrics = compute_classification_metrics(target, probabilities, threshold=0.5)

    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["sensitivity"] == pytest.approx(1.0)
    assert metrics["specificity"] == pytest.approx(0.5)
    assert metrics["false_negative_rate"] == pytest.approx(0.0)
    assert metrics["brier_score"] == pytest.approx(0.1675)


def test_metrics_reject_probabilities_outside_valid_range():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        compute_classification_metrics([0, 1], [0.2, 1.1], threshold=0.5)