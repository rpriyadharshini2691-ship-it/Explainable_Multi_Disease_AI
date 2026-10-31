import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.preprocessing import build_training_pipeline


def test_prediction_probabilities_and_model_round_trip(tmp_path):
    features = pd.DataFrame(
        {
            "measurement": [0.2, 0.8, 1.2, 1.8, 2.2, 2.8],
            "category": ["a", "b", "a", "b", "a", "b"],
        }
    )
    target = pd.Series([0, 0, 0, 1, 1, 1])
    pipeline = build_training_pipeline(LogisticRegression(), features)
    pipeline.fit(features, target)
    model_path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, model_path)

    loaded_pipeline = joblib.load(model_path)
    new_features = pd.DataFrame(
        {"measurement": [1.0, 2.5], "category": ["unseen", "b"]}
    )
    probabilities = loaded_pipeline.predict_proba(new_features)

    assert loaded_pipeline.classes_.tolist() == [0, 1]
    assert probabilities.shape == (2, 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert probabilities.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])