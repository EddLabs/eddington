import pytest

from tests.fit_data import DEFAULT_SHEET


@pytest.fixture
def mock_load_workbook(mocker):
    return mocker.patch("openpyxl.load_workbook")


@pytest.fixture
def mock_create_workbook(mocker):
    create_workbook = mocker.patch("openpyxl.Workbook")
    create_workbook.return_value.active.title = DEFAULT_SHEET
    return create_workbook


@pytest.fixture
def mock_csv_write(mocker):
    return mocker.patch("csv.writer")
