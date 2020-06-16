import pytest
from pytest_cases import cases_data, THIS_MODULE, case_tags
from collections import namedtuple
from copy import deepcopy

from eddington_core import (
    FitData,
    FitDataColumnExistenceError,
    FitDataColumnIndexError,
    FitDataColumnsLengthError,
)

from tests.fit_data import COLUMNS, COLUMNS_NAMES

ColumnsResult = namedtuple("ColumnsResult", ["x", "y", "xerr", "yerr"])

COLUMNS_OPTIONS = ["x_column", "xerr_column", "y_column", "yerr_column"]


def case_default():
    fit_data = FitData(COLUMNS)
    result = ColumnsResult(x="a", xerr="b", y="c", yerr="d")
    return fit_data, result


def case_int_x_column():
    fit_data = FitData(COLUMNS, x_column=3)
    result = ColumnsResult(x="c", xerr="d", y="e", yerr="f",)
    return fit_data, result


def case_string_x_column():
    fit_data = FitData(COLUMNS, x_column="c")
    result = ColumnsResult(x="c", xerr="d", y="e", yerr="f",)
    return fit_data, result


def case_int_y_column():
    fit_data = FitData(COLUMNS, y_column=5)
    result = ColumnsResult(x="a", xerr="b", y="e", yerr="f")
    return fit_data, result


def case_string_y_column():
    fit_data = FitData(COLUMNS, y_column="e")
    result = ColumnsResult(x="a", xerr="b", y="e", yerr="f")
    return fit_data, result


def case_int_xerr_column():
    fit_data = FitData(COLUMNS, xerr_column=4)
    result = ColumnsResult(x="a", xerr="d", y="e", yerr="f",)
    return fit_data, result


def case_string_xerr_column():
    fit_data = FitData(COLUMNS, xerr_column="d")
    result = ColumnsResult(x="a", xerr="d", y="e", yerr="f",)
    return fit_data, result


def case_int_yerr_column():
    fit_data = FitData(COLUMNS, yerr_column=6)
    result = ColumnsResult(x="a", xerr="b", y="c", yerr="f")
    return fit_data, result


def case_string_yerr_column():
    fit_data = FitData(COLUMNS, yerr_column="f")
    result = ColumnsResult(x="a", xerr="b", y="c", yerr="f")
    return fit_data, result


def case_x_and_y_column():
    fit_data = FitData(COLUMNS, x_column=3, y_column="h")
    result = ColumnsResult(x="c", xerr="d", y="h", yerr="i")
    return fit_data, result


def case_jumbled_columns():
    fit_data = FitData(COLUMNS, x_column=3, xerr_column=1, y_column="b", yerr_column=9)
    result = ColumnsResult(x="c", xerr="a", y="b", yerr="i")
    return fit_data, result


@cases_data(module=THIS_MODULE)
def test_x(case_data):
    fit_data, result = case_data.get()
    assert COLUMNS[result.x] == pytest.approx(
        fit_data.x
    ), "X is different than expected"


@cases_data(module=THIS_MODULE)
def test_x_err(case_data):
    fit_data, result = case_data.get()
    assert COLUMNS[result.xerr] == pytest.approx(
        fit_data.xerr
    ), "X error is different than expected"


@cases_data(module=THIS_MODULE)
def test_y(case_data):
    fit_data, result = case_data.get()
    assert COLUMNS[result.y] == pytest.approx(
        fit_data.y
    ), "Y is different than expected"


@cases_data(module=THIS_MODULE)
def test_y_err(case_data):
    fit_data, result = case_data.get()
    assert COLUMNS[result.yerr] == pytest.approx(
        fit_data.yerr
    ), "Y error is different than expected"


@cases_data(module=THIS_MODULE)
def test_all_columns(case_data):
    fit_data, result = case_data.get()
    assert COLUMNS_NAMES == fit_data.all_columns, "Columns are different than expected"


@cases_data(module=THIS_MODULE)
def test_data(case_data):
    fit_data, result = case_data.get()
    assert (
        COLUMNS_NAMES == list(fit_data.data.keys()),
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
