import numpy as np
import pandas as pd
import pytest

from src.reliability import (
    aggregate_source_importance,
    perturb_numeric_features,
    rank_correlation,
    top_k_jaccard,
)


def test_numeric_perturbation_keeps_categorical_values_and_clips_to_training_range():
    training = pd.DataFrame(
        {"measurement": [0.0, 10.0, 20.0], "category": ["a", "b", "a"]}
    )
    original = pd.DataFrame(
        {"measurement": [0.0, 20.0], "category": ["b", "a"]}
    )

    perturbed = perturb_numeric_features(
        original,
        training,
        numeric_features=["measurement"],
        magnitude=10.0,
        random_generator=np.random.default_rng(42),
    )

    assert perturbed["category"].tolist() == original["category"].tolist()
    assert perturbed["measurement"].between(0.0, 20.0).all()
    assert not np.array_equal(
        perturbed["measurement"].to_numpy(), original["measurement"].to_numpy()
    )


def test_aggregate_source_importance_combines_one_hot_columns():
    shap_values = np.asarray([[1.0, -2.0, 3.0], [-4.0, 1.0, -2.0]])
    names = ["numeric__age", "categorical__sex_0", "categorical__sex_1"]
    mapping = {
        "numeric__age": "age",
        "categorical__sex_0": "sex",
        "categorical__sex_1": "sex",
    }

    source_names, importance = aggregate_source_importance(
        shap_values, names, mapping
    )

    assert source_names == ["age", "sex"]
    assert importance.tolist() == [[1.0, 5.0], [4.0, 3.0]]


def test_rank_correlation_and_top_k_jaccard_match_expected_values():
    first = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    second = [5.0, 4.0, 0.0, 3.0, 2.0, 1.0]

    assert rank_correlation(first, second) == pytest.approx(0.6571428571428571)
    assert top_k_jaccard(first, second, k=5) == pytest.approx(4 / 6)


def test_rank_correlation_handles_constant_equal_vectors():
    assert rank_correlation([0.0, 0.0], [0.0, 0.0]) == 1.0