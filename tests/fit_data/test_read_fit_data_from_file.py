from collections import namedtuple
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from mock import PropertyMock, mock_open, patch
from pytest_cases import parametrize_plus, fixture_ref

from eddington import FitData, FitDataInvalidFileSyntax
from tests.fit_data import COLUMNS, CONTENT, ROWS, VALUES

DummyCell = namedtuple("DummyCell", "value")
file_name = "file"
filepath = Path("path/to") / file_name
sheet_name = "sheet"


def check_data_by_keys(actual_fit_data):
    for key in actual_fit_data.data.keys():
        np.testing.assert_equal(
            actual_fit_data.data[key],
            COLUMNS[key],
            err_msg="Data is different than expected",
        )


def check_data_by_indexes(actual_fit_data):
    for key in actual_fit_data.data.keys():
        np.testing.assert_equal(
            actual_fit_data.data[key],
            VALUES[key],
            err_msg="Data is different than expected",
        )


def check_columns(
    actual_fit_data, x_column=0, xerr_column=1, y_column=2, yerr_column=3
):
    np.testing.assert_equal(
        actual_fit_data.x, VALUES[x_column], err_msg="X is different than expected",
    )
    np.testing.assert_equal(
        actual_fit_data.xerr,
        VALUES[xerr_column],
        err_msg="X Error is different than expected",
    )
    np.testing.assert_equal(
        actual_fit_data.y, VALUES[y_column], err_msg="Y is different than expected",
    )
    np.testing.assert_equal(
        actual_fit_data.yerr,
        VALUES[yerr_column],
        err_msg="Y Error is different than expected",
    )


def set_csv_rows(reader, rows):
    reader.return_value = rows


@pytest.fixture
def read_csv(mocker):
    reader = mocker.patch("csv.reader")
    m_open = mock_open()

    def actual_read(**kwargs):
        with patch("eddington.fit_data.open", m_open):
            actual_fit_data = FitData.read_from_csv(filepath, **kwargs)
        return actual_fit_data

    return actual_read, dict(reader=reader, row_setter=set_csv_rows)


def set_excel_rows(reader, rows):
    def nrows():
        return len(rows)

    def get_row(i):
        return [DummyCell(value=element) for element in rows[i]]

    sheet = reader.return_value.sheet_by_name.return_value

    type(sheet).nrows = PropertyMock(side_effect=nrows)
    sheet.row.side_effect = get_row


@pytest.fixture
def mock_open_workbook(mocker):
    open_workbook = mocker.patch("xlrd.open_workbook")
    return open_workbook


@pytest.fixture
def read_excel(mock_open_workbook):
    def actual_read(**kwargs):
        return FitData.read_from_excel(filepath, sheet_name, **kwargs)

    return actual_read, dict(reader=mock_open_workbook, row_setter=set_excel_rows)


@parametrize_plus("read, mocks", [fixture_ref(read_csv), fixture_ref(read_excel)])
def test_read_with_headers_successful(read, mocks):
    mocks["row_setter"](mocks["reader"], ROWS)

    actual_fit_data = read()

    check_data_by_keys(actual_fit_data)
    check_columns(actual_fit_data)


@parametrize_plus("read, mocks", [fixture_ref(read_csv), fixture_ref(read_excel)])
def test_read_without_headers_successful(read, mocks):
    mocks["row_setter"](mocks["reader"], CONTENT)

    actual_fit_data = read()

    check_data_by_indexes(actual_fit_data)
    check_columns(actual_fit_data)


@parametrize_plus("read, mocks", [fixture_ref(read_csv), fixture_ref(read_excel)])
def test_read_without_headers_unsuccessful(read, mocks):
    rows = deepcopy(CONTENT)
    rows[1][0] = "f"
    mocks["row_setter"](mocks["reader"], rows)

    with pytest.raises(FitDataInvalidFileSyntax):
        read()


@parametrize_plus("read, mocks", [fixture_ref(read_csv), fixture_ref(read_excel)])
def test_read_with_x_column(read, mocks):
    mocks["row_setter"](mocks["reader"], ROWS)

    actual_fit_data = read(x_column=3)

    check_columns(actual_fit_data, x_column=2, xerr_column=3, y_column=4, yerr_column=5)


@parametrize_plus("read, mocks", [fixture_ref(read_csv), fixture_ref(read_excel)])
def test_read_with_xerr_column(read, mocks):
    mocks["row_setter"](mocks["reader"], ROWS)

    actual_fit_data = read(xerr_column=3)

    check_columns(actual_fit_data, x_column=0, xerr_column=2, y_column=3, yerr_column=4)


@parametrize_plus("read, mocks", [fixture_ref(read_csv), fixture_ref(read_excel)])
def test_read_with_y_column(read, mocks):
    mocks["row_setter"](mocks["reader"], ROWS)

    actual_fit_data = read(y_column=5)

    check_columns(actual_fit_data, x_column=0, xerr_column=1, y_column=4, yerr_column=5)


@parametrize_plus("read, mocks", [fixture_ref(read_csv), fixture_ref(read_excel)])
def test_read_with_yerr_column(read, mocks):
    mocks["row_setter"](mocks["reader"], ROWS)

    actual_fit_data = read(yerr_column=5)

    check_columns(actual_fit_data, x_column=0, xerr_column=1, y_column=2, yerr_column=4)
