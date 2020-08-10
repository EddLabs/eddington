import pytest
from pytest_cases import THIS_MODULE, parametrize_with_cases
from copy import deepcopy

from eddington import (
    FitData,
    FitDataColumnExistenceError,
    FitDataColumnIndexError,
    FitDataColumnsLengthError,
)
from eddington.fit_data import Columns

from tests.fit_data import COLUMNS, COLUMNS_NAMES

COLUMNS_OPTIONS = ["x_column", "xerr_column", "y_column", "yerr_column"]


def case_default():
    fit_data = FitData(COLUMNS)
    expected_columns = Columns(x="a", xerr="b", y="c", yerr="d")
    return fit_data, expected_columns


def case_int_x_column():
    fit_data = FitData(COLUMNS, x_column=3)
    expected_columns = Columns(x="c", xerr="d", y="e", yerr="f",)
    return fit_data, expected_columns


def case_string_x_column():
    fit_data = FitData(COLUMNS, x_column="c")
    expected_columns = Columns(x="c", xerr="d", y="e", yerr="f",)
    return fit_data, expected_columns


def case_int_y_column():
    fit_data = FitData(COLUMNS, y_column=5)
    expected_columns = Columns(x="a", xerr="b", y="e", yerr="f")
    return fit_data, expected_columns


def case_string_y_column():
    fit_data = FitData(COLUMNS, y_column="e")
    expected_columns = Columns(x="a", xerr="b", y="e", yerr="f")
    return fit_data, expected_columns


def case_int_xerr_column():
    fit_data = FitData(COLUMNS, xerr_column=4)
    expected_columns = Columns(x="a", xerr="d", y="e", yerr="f",)
    return fit_data, expected_columns


def case_string_xerr_column():
    fit_data = FitData(COLUMNS, xerr_column="d")
    expected_columns = Columns(x="a", xerr="d", y="e", yerr="f",)
    return fit_data, expected_columns


def case_int_yerr_column():
    fit_data = FitData(COLUMNS, yerr_column=6)
    expected_columns = Columns(x="a", xerr="b", y="c", yerr="f")
    return fit_data, expected_columns


def case_string_yerr_column():
    fit_data = FitData(COLUMNS, yerr_column="f")
    expected_columns = Columns(x="a", xerr="b", y="c", yerr="f")
    return fit_data, expected_columns


def case_x_and_y_column():
    fit_data = FitData(COLUMNS, x_column=3, y_column="h")
    expected_columns = Columns(x="c", xerr="d", y="h", yerr="i")
    return fit_data, expected_columns


def case_jumbled_columns():
    fit_data = FitData(COLUMNS, x_column=3, xerr_column=1, y_column="b", yerr_column=9)
    expected_columns = Columns(x="c", xerr="a", y="b", yerr="i")
    return fit_data, expected_columns


@parametrize_with_cases(argnames="fit_data, expected_columns", cases=THIS_MODULE)
def test_x_column(fit_data, expected_columns):
    assert (
        expected_columns.x == fit_data.x_column
    ), "X column name is different than expected"


@parametrize_with_cases(argnames="fit_data, expected_columns", cases=THIS_MODULE)
def test_x_data(fit_data, expected_columns):
    assert COLUMNS[expected_columns.x] == pytest.approx(
        fit_data.x
    ), "X is different than expected"


@parametrize_with_cases(argnames="fit_data, expected_columns", cases=THIS_MODULE)
def test_xerr_column(fit_data, expected_columns):
    assert (
        expected_columns.xerr == fit_data.xerr_column
    ), "X error column name is different than expected"


@parametrize_with_cases(argnames="fit_data, expected_columns", cases=THIS_MODULE)
def test_xerr_data(fit_data, expected_columns):
    assert COLUMNS[expected_columns.xerr] == pytest.approx(
        fit_data.xerr
    ), "X error is different than expected"


@parametrize_with_cases(argnames="fit_data, expected_columns", cases=THIS_MODULE)
def test_y_column(fit_data, expected_columns):
    assert (
        expected_columns.y == fit_data.y_column
    ), "Y column name is different than expected"


@parametrize_with_cases(argnames="fit_data, expected_columns", cases=THIS_MODULE)
def test_y_data(fit_data, expected_columns):
    assert COLUMNS[expected_columns.y] == pytest.approx(
        fit_data.y
    ), "Y is different than expected"


@parametrize_with_cases(argnames="fit_data, expected_columns", cases=THIS_MODULE)
def test_yerr_column(fit_data, expected_columns):
    assert (
        expected_columns.yerr == fit_data.yerr_column
    ), "Y error column name is different than expected"


@parametrize_with_cases(argnames="fit_data, expected_columns", cases=THIS_MODULE)
def test_yerr_data(fit_data, expected_columns):
    assert COLUMNS[expected_columns.yerr] == pytest.approx(
        fit_data.yerr
    ), "Y error is different than expected"


@parametrize_with_cases(argnames="fit_data, expected_columns", cases=THIS_MODULE)
def test_all_columns(fit_data, expected_columns):
    assert COLUMNS_NAMES == fit_data.all_columns, "Columns are different than expected"


@parametrize_with_cases(argnames="fit_data, expected_columns", cases=THIS_MODULE)
def test_used_columns(fit_data, expected_columns):
    assert (
        fit_data.used_columns == expected_columns
    ), "Used columns are different than expected"


@parametrize_with_cases(argnames="fit_data, expected_columns", cases=THIS_MODULE)
def test_data(fit_data, expected_columns):
    assert COLUMNS_NAMES == list(
        fit_data.data.keys()
    ), "Data keys are different than expected"
    for key, item in fit_data.data.items():
        assert item == pytest.approx(
            COLUMNS[key]
        ), f"Value of {key} is different than expected."


@pytest.mark.parametrize("column", COLUMNS_OPTIONS)
def test_x_not_existing(column):
    with pytest.raises(
        FitDataColumnExistenceError, match='^Could not find column "r" in data$'
    ):
        FitData(COLUMNS, **{column: "r"})


@pytest.mark.parametrize("column", COLUMNS_OPTIONS)
def test_x_zero_index(column):
    with pytest.raises(
        FitDataColumnIndexError,
        match="^No column number 0 in data. index should be between 1 and 10$",
    ):
        FitData(COLUMNS, **{column: 0})


@pytest.mark.parametrize("column", COLUMNS_OPTIONS)
def test_x_larger_than_size(column):
    with pytest.raises(
        FitDataColumnIndexError,
        match="^No column number 11 in data. index should be between 1 and 10$",
    ):
        FitData(COLUMNS, **{column: 11})


def test_exception_risen_because_of_columns_length():
    data = deepcopy(COLUMNS)
    data["a"] = data["a"][:-2]
    with pytest.raises(
        FitDataColumnsLengthError,
        match="^All columns in FitData should have the same length$",
    ):
        FitData(data=data)
