import pytest


@pytest.fixture
def mock_load_workbook(mocker):
    return mocker.patch("openpyxl.load_workbook")


@pytest.fixture
def mock_load_json(mocker):
    return mocker.patch("json.load")


@pytest.fixture
def mock_save_as_excel(mocker):
    return mocker.patch("eddington.io_util.save_as_excel")


@pytest.fixture
def mock_save_as_csv(mocker):
    return mocker.patch("eddington.io_util.save_as_csv")
