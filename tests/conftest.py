import pytest


@pytest.fixture
def json_dump_mock(mocker):
    return mocker.patch("json.dump")
