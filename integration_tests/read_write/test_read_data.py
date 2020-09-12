from pathlib import Path

import pytest
from pytest_cases import parametrize_with_cases

from eddington import FittingData, FittingDataInvalidFileSyntax

SOURCES_DIRECTORY = Path(__file__).parent.parent / "resources"

EXCEL_FILE = SOURCES_DIRECTORY / "data.xlsx"
VALID_SHEET_NAME = "data1"
INVALID_SHEET_NAME = "invalid_data1"

VALID_CSV_FILE = SOURCES_DIRECTORY / "valid_data.csv"
INVALID_CSV_FILE = SOURCES_DIRECTORY / "invalid_data.csv"

VALID_JSON_FILE = SOURCES_DIRECTORY / "valid_data.json"
INVALID_JSON_FILE = SOURCES_DIRECTORY / "invalid_data.json"


def read_excel_method(sheet_name):
    def read_method(**kwargs):
        return FittingData.read_from_excel(EXCEL_FILE, sheet_name, **kwargs)

    return read_method


def case_read_valid_excel():
    return read_excel_method(VALID_SHEET_NAME)


def case_read_invalid_excel():
    return read_excel_method(INVALID_SHEET_NAME)


def read_csv_method(file_path):
    def read_method(**kwargs):
        return FittingData.read_from_csv(file_path, **kwargs)

    return read_method


def case_read_valid_csv():
    return read_csv_method(VALID_CSV_FILE)


def case_read_invalid_csv():
    return read_csv_method(INVALID_CSV_FILE)


def read_json_method(file_path):
    def read_method(**kwargs):
        return FittingData.read_from_json(file_path, **kwargs)

    return read_method


def case_read_valid_json():
    return read_json_method(VALID_JSON_FILE)


def case_read_invalid_json():
    return read_json_method(INVALID_JSON_FILE)


VALID_CASES = [case_read_valid_excel, case_read_valid_csv, case_read_valid_json]
INVALID_CASES = [case_read_invalid_excel, case_read_invalid_csv, case_read_invalid_json]


@parametrize_with_cases(argnames="read_method", cases=VALID_CASES)
def test_simple_read(read_method):
    fit_data: FittingData = read_method()
    assert fit_data.x == pytest.approx(
        [10, 20, 30, 40, 50, 60, 70]
    ), "x is different than expected"
    assert fit_data.xerr == pytest.approx(
        [0.5, 1.0, 1.2, 0.3, 0.4, 1.1, 1.3]
    ), "x error is different than expected"
    assert fit_data.y == pytest.approx(
        [16.0, 29.0, 47.0, 56.0, 70.0, 92.0, 100.0]
    ), "y is different than expected"
    assert fit_data.yerr == pytest.approx(
        [1.0, 1.3, 0.8, 2.0, 1.1, 0.2, 2.0]
    ), "y error is different than expected"


@parametrize_with_cases(argnames="read_method", cases=VALID_CASES)
def test_read_with_y_column(read_method):
    fit_data: FittingData = read_method(y_column=5)
    assert fit_data.x == pytest.approx(
        [10, 20, 30, 40, 50, 60, 70]
    ), "x is different than expected"
    assert fit_data.xerr == pytest.approx(
        [0.5, 1.0, 1.2, 0.3, 0.4, 1.1, 1.3]
    ), "x error is different than expected"
    assert fit_data.y == pytest.approx(
        [100.0, 401.0, 910.0, 1559.0, 2480.0, 3623.0, 4910.0]
    ), "y is different than expected"
    assert fit_data.yerr == pytest.approx(
        [14.0, 10.0, 11.0, 8.0, 10.0, 5.0, 16.0]
    ), "y error is different than expected"


@parametrize_with_cases(argnames="read_method", cases=INVALID_CASES)
def test_read_invalid_data_file(read_method):
    with pytest.raises(FittingDataInvalidFileSyntax):
        read_method()
