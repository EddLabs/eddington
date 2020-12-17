from pathlib import Path

import mock
import numpy as np
import pytest

from eddington.io_util import save_as_csv, save_as_excel
from tests.util import assert_calls

CONTENT = [np.random.uniform(0, 1, size=12) for _ in range(20)]
EPSILON = 1e-3
DEFAULT_SHEET = "default_sheet"


@pytest.fixture
def mock_openpyxl_workbook(mocker):
    workbook_mock = mocker.patch("openpyxl.Workbook")
    workbook_mock.return_value.active.title = DEFAULT_SHEET
    return workbook_mock


@pytest.fixture
def mock_csv_writer(mocker):
    return mocker.patch("csv.writer")


def assert_workbook(mock_openpyxl_workbook, sheet_name, saved_file_path):
    mock_openpyxl_workbook.assert_called_once_with()
    workbook = mock_openpyxl_workbook.return_value
    worksheet = workbook.active
    assert worksheet.title == sheet_name
    assert_calls(worksheet.append, [([record], {}) for record in CONTENT], rel=EPSILON)
    workbook.save.assert_called_once_with(saved_file_path)


def test_save_as_excel_without_sheet_name(mock_openpyxl_workbook):
    output_directory = "/path/to/directory"
    file_name = "data"
    save_as_excel(
        content=CONTENT, output_directory=output_directory, file_name=file_name
    )
    assert_workbook(
        mock_openpyxl_workbook,
        sheet_name=DEFAULT_SHEET,
        saved_file_path=Path("/path/to/directory/data.xlsx"),
    )


def test_save_as_excel_with_sheet_name(mock_openpyxl_workbook):
    output_directory = "/path/to/directory"
    file_name = "data"
    sheet_name = "sheet1"
    save_as_excel(
        content=CONTENT,
        output_directory=output_directory,
        file_name=file_name,
        sheet=sheet_name,
    )
    assert_workbook(
        mock_openpyxl_workbook,
        sheet_name=sheet_name,
        saved_file_path=Path("/path/to/directory/data.xlsx"),
    )


def test_save_as_csv(mock_csv_writer):
    output_directory = "/path/to/directory"
    file_name = "data"
    mock_open = mock.mock_open()
    with mock.patch("eddington.io_util.open", mock_open):
        save_as_csv(
            content=CONTENT, file_name=file_name, output_directory=output_directory
        )
    mock_open.assert_called_once_with(
        Path("/path/to/directory/data.csv"), mode="w+", newline=""
    )
    mock_csv_writer.return_value.writerows.assert_called_once_with(CONTENT)
