from pathlib import Path

import numpy as np
import pytest
from mock import call, mock_open, patch

from eddington import FittingData
from tests.fitting_data import (
    COLUMNS,
    COLUMNS_NAMES,
    CONTENT,
    DEFAULT_SHEET,
    NUMBER_OF_RECORDS,
)

DECIMAL = 5

DIRECTORY_PATH = Path("/path/to/directory")
EPSILON = 1e-5

FIT_DATA = FittingData(COLUMNS)


def assert_workbook_calls(workbook, sheet_name=DEFAULT_SHEET, name="fitting_data"):
    worksheet = workbook.active
    assert worksheet.title == sheet_name
    assert (
        worksheet.append.call_count == NUMBER_OF_RECORDS + 1
    ), "worksheet append called unexpected number of times"
    assert worksheet.append.call_args_list[0] == call(
        COLUMNS_NAMES
    ), "worksheet append called unexpected number of times"
    for i in range(NUMBER_OF_RECORDS):
        assert worksheet.append.call_args_list[i + 1][0][0] == pytest.approx(
            CONTENT[i], rel=EPSILON
        ), "worksheet append called unexpected number of times"
    workbook.save.assert_called_once_with(DIRECTORY_PATH / f"{name}.xlsx")


def test_save_to_excel_without_name_and_sheet(mock_create_workbook):
    FIT_DATA.save_excel(DIRECTORY_PATH)
    workbook = mock_create_workbook.return_value
    assert_workbook_calls(workbook)


def test_save_to_excel_with_sheet(mock_create_workbook):
    sheet_name = "some_sheet_name"
    FIT_DATA.save_excel(DIRECTORY_PATH, sheet=sheet_name)
    workbook = mock_create_workbook.return_value
    assert_workbook_calls(workbook, sheet_name=sheet_name)


def test_save_to_excel_with_name(mock_create_workbook):
    file_name = "some_sheet_name"
    FIT_DATA.save_excel(DIRECTORY_PATH, name=file_name)
    workbook = mock_create_workbook.return_value
    assert_workbook_calls(workbook, name=file_name)


def test_save_to_csv_without_name(mock_csv_write):
    m_open = mock_open()
    with patch("eddington.fitting_data.open", m_open):
        FIT_DATA.save_csv(DIRECTORY_PATH)
        m_open.assert_called_once_with(
            DIRECTORY_PATH / "fitting_data.csv", mode="w+", newline=""
        )
        mock_csv_write.assert_called_once_with(m_open.return_value)
    csv_writer = mock_csv_write.return_value
    csv_writer.writerow.assert_called_once_with(COLUMNS_NAMES)
    assert (
        csv_writer.writerows.call_count == 1
    ), "csv_writer.writerows called different than expected."
    np.testing.assert_almost_equal(
        list(csv_writer.writerows.call_args_list[0][0][0]), CONTENT, decimal=DECIMAL
    )


def test_save_to_csv_with_name(mock_csv_write):
    name = "some_csv_name"
    m_open = mock_open()
    with patch("eddington.fitting_data.open", m_open):
        FIT_DATA.save_csv(DIRECTORY_PATH, name=name)
        m_open.assert_called_once_with(
            DIRECTORY_PATH / f"{name}.csv", mode="w+", newline=""
        )
        mock_csv_write.assert_called_once_with(m_open.return_value)
    csv_writer = mock_csv_write.return_value
    csv_writer.writerow.assert_called_once_with(COLUMNS_NAMES)
    assert (
        csv_writer.writerows.call_count == 1
    ), "csv_writer.writerows called different than expected."
    np.testing.assert_almost_equal(
        list(csv_writer.writerows.call_args_list[0][0][0]), CONTENT, decimal=DECIMAL
    )
