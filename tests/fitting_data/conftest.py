import pytest


@pytest.fixture
def mock_load_json(mocker):
    return mocker.patch("json.load")
