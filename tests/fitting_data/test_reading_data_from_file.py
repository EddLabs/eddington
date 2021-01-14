from collections import OrderedDict
from pathlib import Path
from unittest import mock

import pytest
from pytest_cases import fixture

from eddington import FittingData, FittingDataError
from eddington.raw_data_builder import RawDataBuilder
from tests.fitting_data import COLUMNS
from tests.util import assert_dict_equal

EPSILON = 1e-5
SHEET1, SHEET2, SHEET3 = "sheet1", "sheet2", "sheet3"
SHEETS = [SHEET1, SHEET2, SHEET3]
NO_EXISTING_SHEET = "some_sheet"
DIR_PATH = Path("path/to/dir")
EXCEL_PATH = DIR_PATH / "data.xlsx"
CSV_PATH = DIR_PATH / "data.csv"
JSON_PATH = DIR_PATH / "data.json"


@fixture
def mock_building_raw_data(mocker):
    builder = mocker.patch.object(RawDataBuilder, "build_raw_data")
    builder.return_value = COLUMNS
    return builder


@fixture
def mock_load_workbook(mocker):
    load_workbook_mocker = mocker.patch("openpyxl.load_workbook")
    load_workbook_mocker.return_value.sheetnames = SHEETS
    return load_workbook_mocker


@pytest.fixture
def mock_csv_reader(mocker):
    return mocker.patch("csv.reader")


def test_reading_data_from_excel_with_file_successful(
    mock_load_workbook, mock_building_raw_data
):
    data = FittingData.read_from_excel(EXCEL_PATH, sheet=SHEET1)
    mock_load_workbook.assert_called_with(EXCEL_PATH, data_only=True)
    assert_dict_equal(data.data, COLUMNS, rel=EPSILON)


def test_reading_data_from_excel_with_str_successful(
    mock_load_workbook, mock_building_raw_data
):
    data = FittingData.read_from_excel(str(EXCEL_PATH), sheet=SHEET1)
    mock_load_workbook.assert_called_with(EXCEL_PATH, data_only=True)
    assert_dict_equal(data.data, COLUMNS, rel=EPSILON)


def test_reading_data_from_excel_fail_for_no_existing_sheet(
    mock_load_workbook, mock_building_raw_data
):
    with pytest.raises(
        FittingDataError,
        match=(
            f'^Sheet named "{NO_EXISTING_SHEET}" '
            f'does not exist in "{EXCEL_PATH.name}"$'
        ),
    ):
        FittingData.read_from_excel(EXCEL_PATH, sheet=NO_EXISTING_SHEET)


def test_reading_data_from_csv_with_file_successful(
    mock_csv_reader, mock_building_raw_data
):
    mock_open = mock.mock_open()
    with mock.patch("eddington.fitting_data.open", mock_open):
        data = FittingData.read_from_csv(CSV_PATH)
    mock_open.assert_called_once_with(CSV_PATH, mode="r")
    mock_csv_reader.assert_called_with(mock_open.return_value)
    assert_dict_equal(data.data, COLUMNS, rel=EPSILON)


def test_reading_data_from_csv_with_str_successful(
    mock_csv_reader, mock_building_raw_data
):
    mock_open = mock.mock_open()
    with mock.patch("eddington.fitting_data.open", mock_open):
        data = FittingData.read_from_csv(str(CSV_PATH))
    mock_open.assert_called_once_with(CSV_PATH, mode="r")
    mock_csv_reader.assert_called_with(mock_open.return_value)
    assert_dict_equal(data.data, COLUMNS, rel=EPSILON)


def test_reading_data_from_json_with_file_successful(
    mock_load_json, mock_building_raw_data
):
    mock_load_json.return_value = COLUMNS
    mock_open = mock.mock_open()
    with mock.patch("eddington.fitting_data.open", mock_open):
        data = FittingData.read_from_json(JSON_PATH)
    mock_open.assert_called_once_with(JSON_PATH, mode="r")
    mock_load_json.assert_called_with(
        mock_open.return_value, object_pairs_hook=OrderedDict
    )
    assert_dict_equal(data.data, COLUMNS, rel=EPSILON)


def test_reading_data_from_json_with_str_successful(
    mock_load_json, mock_building_raw_data
):
    mock_load_json.return_value = COLUMNS
    mock_open = mock.mock_open()
    with mock.patch("eddington.fitting_data.open", mock_open):
        data = FittingData.read_from_json(str(JSON_PATH))
    mock_open.assert_called_once_with(JSON_PATH, mode="r")
    mock_load_json.assert_called_with(
        mock_open.return_value, object_pairs_hook=OrderedDict
    )
    assert_dict_equal(data.data, COLUMNS, rel=EPSILON)
