from pathlib import Path

import pytest

from bikeshare import serving

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.mark.skipif(not (serving.MODELS / "model_card.json").exists(), reason="run notebooks/03_forecasting first")
def test_app_runs_without_errors():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    assert app.title[0].value.startswith("Seoul bike demand")
    # the what-if slider should change the forecast
    before = app.metric[-1].value
    app.slider[-1].set_value(10.0).run()
    assert not app.exception
    assert app.metric[-1].value != before


@pytest.mark.skipif(not (serving.MODELS / "model_card.json").exists(), reason="run notebooks/03_forecasting first")
def test_app_handles_a_closed_day():
    import datetime

    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.date_input[0].set_value(datetime.date(2018, 10, 2)).run()
    assert not app.exception
    assert "closed" in app.info[0].value
