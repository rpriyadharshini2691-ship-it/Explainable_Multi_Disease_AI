from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_streamlit_app_starts_and_renders_project_overview():
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    assert not app.exception
    assert app.title[0].value == "Explainable Multi-Disease AI"
    assert app.radio[0].value == "Overview"
    assert "Early-stage diabetes" in app.radio[0].options
    assert "Heart disease" in app.radio[0].options
    assert "Chronic kidney disease" in app.radio[0].options