import numpy as np
import pytest

from src.compare_models import (
    WeightedSoftVotingClassifier,
    derive_ensemble_weights,
)


class _FixedProbabilityEstimator:
    def __init__(self, probabilities, classes):
        self.probabilities = np.asarray(probabilities, dtype=float)
        self.classes_ = np.asarray(classes)

    def predict_proba(self, features):
        return np.tile(self.probabilities, (len(features), 1))


def test_ensemble_weights_are_nonnegative_and_sum_to_one():
    target = np.asarray([0, 0, 1, 1, 0, 1])
    oof_probabilities = np.asarray(
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
            [0.8, 0.7, 0.6],
            [0.9, 0.8, 0.7],
            [0.3, 0.4, 0.5],
            [0.7, 0.6, 0.5],
        ]
    )

    weights = derive_ensemble_weights(oof_probabilities, target)

    assert set(weights) == {"logistic_regression", "random_forest", "xgboost"}
    assert all(weight >= 0 for weight in weights.values())
    assert sum(weights.values()) == pytest.approx(1.0)


def test_ensemble_aligns_probabilities_using_estimator_class_labels():
    estimators = {
        "normal": _FixedProbabilityEstimator([0.8, 0.2], classes=[0, 1]),
        "reversed": _FixedProbabilityEstimator([0.1, 0.9], classes=[1, 0]),
    }
    ensemble = WeightedSoftVotingClassifier(
        estimators,
        weights={"normal": 0.25, "reversed": 0.75},
    )

    probabilities = ensemble.predict_proba(["row one", "row two"])

    assert probabilities.shape == (2, 2)
    assert probabilities[:, 1].tolist() == pytest.approx([0.125, 0.125])
    assert probabilities.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])


def test_ensemble_rejects_invalid_weights():
    estimator = _FixedProbabilityEstimator([0.5, 0.5], classes=[0, 1])

    with pytest.raises(ValueError, match="non-negative"):
        WeightedSoftVotingClassifier({"model": estimator}, {"model": -0.1})