import pytest
from streamlit.testing.v1 import AppTest

@pytest.fixture
def app():
    return AppTest.from_file("streamlit_app.py").run()
