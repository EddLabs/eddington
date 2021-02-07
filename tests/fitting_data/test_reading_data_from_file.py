from collections import OrderedDict
from pathlib import Path
from unittest import mock

import pytest
from pytest_cases import fixture, parametrize_with_cases, THIS_MODULE

from eddington import FittingData, FittingDataError
from eddington.fitting_data import Columns
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


# Mocks


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


# Cases


def case_no_additional_args():
    kwargs = dict()
    columns = Columns(x="a", xerr="b", y="c", yerr="d")
    return kwargs, columns


def case_x_column_with_search():
    kwargs = dict(x_column="c")
    columns = Columns(x="c", xerr="d", y="e", yerr="f")
    return kwargs, columns


def case_all_columns_provided():
    kwargs = dict(x_column="c", xerr_column="g", y_column="f", yerr_column="b")
    columns = Columns(x="c", xerr="g", y="f", yerr="b")
    return kwargs, columns


def case_x_and_y_column_without_search():
    kwargs = dict(x_column="c", y_column="g", search=False)
    columns = Columns(x="c", xerr=None, y="g", yerr=None)
    return kwargs, columns


# Assertions

def assert_fitting_data(fitting_data: FittingData, columns: COLUMNS):
    assert_dict_equal(fitting_data.data, COLUMNS, rel=EPSILON)
    assert fitting_data.x_column == columns.x
    assert fitting_data.xerr_column == columns.xerr
    assert fitting_data.y_column == columns.y
    assert fitting_data.yerr_column == columns.yerr


# Tests


@parametrize_with_cases(argnames=["kwargs", "columns"], cases=THIS_MODULE)
def test_reading_data_from_excel_with_file_successful(
    kwargs, columns, mock_load_workbook, mock_building_raw_data
):
    data = FittingData.read_from_excel(EXCEL_PATH, sheet=SHEET1, **kwargs)
    mock_load_workbook.assert_called_with(EXCEL_PATH, data_only=True)
    assert_fitting_data(fitting_data=data, columns=columns)


@parametrize_with_cases(argnames=["kwargs", "columns"], cases=THIS_MODULE)
def test_reading_data_from_excel_with_str_successful(
    kwargs, columns, mock_load_workbook, mock_building_raw_data
):
    data = FittingData.read_from_excel(str(EXCEL_PATH), sheet=SHEET1, **kwargs)
    mock_load_workbook.assert_called_with(EXCEL_PATH, data_only=True)
    assert_fitting_data(fitting_data=data, columns=columns)


@parametrize_with_cases(argnames=["kwargs", "columns"], cases=THIS_MODULE)
def test_reading_data_from_excel_fail_for_no_existing_sheet(
    kwargs, columns, mock_load_workbook, mock_building_raw_data
):
    with pytest.raises(
        FittingDataError,
        match=(
            f'^Sheet named "{NO_EXISTING_SHEET}" '
            f'does not exist in "{EXCEL_PATH.name}"$'
        ),
    ):
        FittingData.read_from_excel(EXCEL_PATH, sheet=NO_EXISTING_SHEET, **kwargs)


@parametrize_with_cases(argnames=["kwargs", "columns"], cases=THIS_MODULE)
def test_reading_data_from_csv_with_file_successful(
    kwargs, columns, mock_csv_reader, mock_building_raw_data
):
    mock_open = mock.mock_open()
    with mock.patch("eddington.fitting_data.open", mock_open):
        data = FittingData.read_from_csv(CSV_PATH, **kwargs)
    mock_open.assert_called_once_with(CSV_PATH, mode="r")
    mock_csv_reader.assert_called_with(mock_open.return_value)
    assert_fitting_data(fitting_data=data, columns=columns)


@parametrize_with_cases(argnames=["kwargs", "columns"], cases=THIS_MODULE)
def test_reading_data_from_csv_with_str_successful(
    kwargs, columns, mock_csv_reader, mock_building_raw_data
):
    mock_open = mock.mock_open()
    with mock.patch("eddington.fitting_data.open", mock_open):
        data = FittingData.read_from_csv(str(CSV_PATH), **kwargs)
    mock_open.assert_called_once_with(CSV_PATH, mode="r")
    mock_csv_reader.assert_called_with(mock_open.return_value)
    assert_fitting_data(fitting_data=data, columns=columns)


@parametrize_with_cases(argnames=["kwargs", "columns"], cases=THIS_MODULE)
def test_reading_data_from_json_with_file_successful(
    kwargs, columns, mock_load_json, mock_building_raw_data
):
    mock_load_json.return_value = COLUMNS
    mock_open = mock.mock_open()
    with mock.patch("eddington.fitting_data.open", mock_open):
        data = FittingData.read_from_json(JSON_PATH, **kwargs)
    mock_open.assert_called_once_with(JSON_PATH, mode="r")
    mock_load_json.assert_called_with(
        mock_open.return_value, object_pairs_hook=OrderedDict
    )
    assert_fitting_data(fitting_data=data, columns=columns)


@parametrize_with_cases(argnames=["kwargs", "columns"], cases=THIS_MODULE)
def test_reading_data_from_json_with_str_successful(
    kwargs, columns, mock_load_json, mock_building_raw_data
):
    mock_load_json.return_value = COLUMNS
    mock_open = mock.mock_open()
    with mock.patch("eddington.fitting_data.open", mock_open):
        data = FittingData.read_from_json(str(JSON_PATH), **kwargs)
    mock_open.assert_called_once_with(JSON_PATH, mode="r")
    mock_load_json.assert_called_with(
        mock_open.return_value, object_pairs_hook=OrderedDict
    )
    assert_fitting_data(fitting_data=data, columns=columns)
