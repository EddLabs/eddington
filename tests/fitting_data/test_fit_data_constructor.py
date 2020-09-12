from copy import deepcopy

import pytest
from pytest_cases import THIS_MODULE, parametrize_with_cases

from eddington import (
    FittingData,
    FittingDataColumnExistenceError,
    FittingDataColumnIndexError,
    FittingDataColumnsLengthError,
)
from eddington.fitting_data import Columns
from tests.fitting_data import COLUMNS, COLUMNS_NAMES

COLUMNS_OPTIONS = ["x_column", "xerr_column", "y_column", "yerr_column"]


def case_default():
    fitting_data = FittingData(COLUMNS)
    expected_columns = Columns(x="a", xerr="b", y="c", yerr="d")
    return fitting_data, expected_columns


def case_int_x_column():
    fitting_data = FittingData(COLUMNS, x_column=3)
    expected_columns = Columns(x="c", xerr="d", y="e", yerr="f")
    return fitting_data, expected_columns


def case_string_x_column():
    fitting_data = FittingData(COLUMNS, x_column="c")
    expected_columns = Columns(x="c", xerr="d", y="e", yerr="f")
    return fitting_data, expected_columns


def case_int_y_column():
    fitting_data = FittingData(COLUMNS, y_column=5)
    expected_columns = Columns(x="a", xerr="b", y="e", yerr="f")
    return fitting_data, expected_columns


def case_string_y_column():
    fitting_data = FittingData(COLUMNS, y_column="e")
    expected_columns = Columns(x="a", xerr="b", y="e", yerr="f")
    return fitting_data, expected_columns


def case_int_xerr_column():
    fitting_data = FittingData(COLUMNS, xerr_column=4)
    expected_columns = Columns(x="a", xerr="d", y="e", yerr="f")
    return fitting_data, expected_columns


def case_string_xerr_column():
    fitting_data = FittingData(COLUMNS, xerr_column="d")
    expected_columns = Columns(x="a", xerr="d", y="e", yerr="f")
    return fitting_data, expected_columns


def case_int_yerr_column():
    fitting_data = FittingData(COLUMNS, yerr_column=6)
    expected_columns = Columns(x="a", xerr="b", y="c", yerr="f")
    return fitting_data, expected_columns


def case_string_yerr_column():
    fitting_data = FittingData(COLUMNS, yerr_column="f")
    expected_columns = Columns(x="a", xerr="b", y="c", yerr="f")
    return fitting_data, expected_columns


def case_x_and_y_column():
    fitting_data = FittingData(COLUMNS, x_column=3, y_column="h")
    expected_columns = Columns(x="c", xerr="d", y="h", yerr="i")
    return fitting_data, expected_columns


def case_jumbled_columns():
    fitting_data = FittingData(COLUMNS, x_column=3, xerr_column=1, y_column="b", yerr_column=9)
    expected_columns = Columns(x="c", xerr="a", y="b", yerr="i")
    return fitting_data, expected_columns


@parametrize_with_cases(argnames="fitting_data, expected_columns", cases=THIS_MODULE)
def test_x_column(fitting_data, expected_columns):
    assert (
        expected_columns.x == fitting_data.x_column
    ), "X column name is different than expected"


@parametrize_with_cases(argnames="fitting_data, expected_columns", cases=THIS_MODULE)
def test_x_data(fitting_data, expected_columns):
    assert COLUMNS[expected_columns.x] == pytest.approx(
        fitting_data.x
    ), "X is different than expected"


@parametrize_with_cases(argnames="fitting_data, expected_columns", cases=THIS_MODULE)
def test_xerr_column(fitting_data, expected_columns):
    assert (
        expected_columns.xerr == fitting_data.xerr_column
    ), "X error column name is different than expected"


@parametrize_with_cases(argnames="fitting_data, expected_columns", cases=THIS_MODULE)
def test_xerr_data(fitting_data, expected_columns):
    assert COLUMNS[expected_columns.xerr] == pytest.approx(
        fitting_data.xerr
    ), "X error is different than expected"


@parametrize_with_cases(argnames="fitting_data, expected_columns", cases=THIS_MODULE)
def test_y_column(fitting_data, expected_columns):
    assert (
        expected_columns.y == fitting_data.y_column
    ), "Y column name is different than expected"


@parametrize_with_cases(argnames="fitting_data, expected_columns", cases=THIS_MODULE)
def test_y_data(fitting_data, expected_columns):
    assert COLUMNS[expected_columns.y] == pytest.approx(
        fitting_data.y
    ), "Y is different than expected"


@parametrize_with_cases(argnames="fitting_data, expected_columns", cases=THIS_MODULE)
def test_yerr_column(fitting_data, expected_columns):
    assert (
        expected_columns.yerr == fitting_data.yerr_column
    ), "Y error column name is different than expected"


@parametrize_with_cases(argnames="fitting_data, expected_columns", cases=THIS_MODULE)
def test_yerr_data(fitting_data, expected_columns):
    assert COLUMNS[expected_columns.yerr] == pytest.approx(
        fitting_data.yerr
    ), "Y error is different than expected"


@parametrize_with_cases(argnames="fitting_data, expected_columns", cases=THIS_MODULE)
def test_all_columns(fitting_data, expected_columns):
    assert COLUMNS_NAMES == fitting_data.all_columns, "Columns are different than expected"


@parametrize_with_cases(argnames="fitting_data, expected_columns", cases=THIS_MODULE)
def test_used_columns(fitting_data, expected_columns):
    assert (
        fitting_data.used_columns == expected_columns
    ), "Used columns are different than expected"


@parametrize_with_cases(argnames="fitting_data, expected_columns", cases=THIS_MODULE)
def test_data(fitting_data, expected_columns):
    assert COLUMNS_NAMES == list(
        fitting_data.data.keys()
    ), "Data keys are different than expected"
    for key, item in fitting_data.data.items():
        assert item == pytest.approx(
            COLUMNS[key]
        ), f"Value of {key} is different than expected."


@pytest.mark.parametrize("column", COLUMNS_OPTIONS)
def test_x_not_existing(column):
    with pytest.raises(
        FittingDataColumnExistenceError, match='^Could not find column "r" in data$'
    ):
        FittingData(COLUMNS, **{column: "r"})


@pytest.mark.parametrize("column", COLUMNS_OPTIONS)
def test_x_zero_index(column):
    with pytest.raises(
        FittingDataColumnIndexError,
        match="^No column number 0 in data. index should be between 1 and 10$",
    ):
        FittingData(COLUMNS, **{column: 0})


@pytest.mark.parametrize("column", COLUMNS_OPTIONS)
def test_x_larger_than_size(column):
    with pytest.raises(
        FittingDataColumnIndexError,
        match="^No column number 11 in data. index should be between 1 and 10$",
    ):
        FittingData(COLUMNS, **{column: 11})


def test_exception_risen_because_of_columns_length():
    data = deepcopy(COLUMNS)
    data["a"] = data["a"][:-2]
    with pytest.raises(
        FittingDataColumnsLengthError,
        match="^All columns in FittingData should have the same length$",
    ):
        FittingData(data=data)
