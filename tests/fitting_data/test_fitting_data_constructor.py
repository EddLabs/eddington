from copy import deepcopy

import pytest
from pytest_cases import THIS_MODULE, parametrize_with_cases

from eddington import (
    FittingData,
    FittingDataColumnExistenceError,
    FittingDataColumnsLengthError,
)
from eddington.fitting_data import Columns
from tests.fitting_data import COLUMNS, COLUMNS_NAMES, CONTENT
from tests.util import assert_list_equal

EPSILON = 1e-3
COLUMNS_OPTIONS = ["x_column", "xerr_column", "y_column", "yerr_column"]


def case_default():
    fitting_data = FittingData(COLUMNS)
    expected_columns = Columns(x="a", xerr="b", y="c", yerr="d")
    expected_indices = dict(x=1, xerr=2, y=3, yerr=4)
    return fitting_data, expected_columns, expected_indices


def case_x_column():
    fitting_data = FittingData(COLUMNS, x_column="c")
    expected_columns = Columns(x="c", xerr="d", y="e", yerr="f")
    expected_indices = dict(x=3, xerr=4, y=5, yerr=6)
    return fitting_data, expected_columns, expected_indices


def case_y_column():
    fitting_data = FittingData(COLUMNS, y_column="e")
    expected_columns = Columns(x="a", xerr="b", y="e", yerr="f")
    expected_indices = dict(x=1, xerr=2, y=5, yerr=6)
    return fitting_data, expected_columns, expected_indices


def case_xerr_column():
    fitting_data = FittingData(COLUMNS, xerr_column="d")
    expected_columns = Columns(x="a", xerr="d", y="e", yerr="f")
    expected_indices = dict(x=1, xerr=4, y=5, yerr=6)
    return fitting_data, expected_columns, expected_indices


def case_yerr_column():
    fitting_data = FittingData(COLUMNS, yerr_column="f")
    expected_columns = Columns(x="a", xerr="b", y="c", yerr="f")
    expected_indices = dict(x=1, xerr=2, y=3, yerr=6)
    return fitting_data, expected_columns, expected_indices


def case_x_and_y_column():
    fitting_data = FittingData(COLUMNS, x_column="c", y_column="h")
    expected_columns = Columns(x="c", xerr="d", y="h", yerr="i")
    expected_indices = dict(x=3, xerr=4, y=8, yerr=9)
    return fitting_data, expected_columns, expected_indices


def case_jumbled_columns():
    fitting_data = FittingData(
        COLUMNS, x_column="c", xerr_column="a", y_column="b", yerr_column="i"
    )
    expected_columns = Columns(x="c", xerr="a", y="b", yerr="i")
    expected_indices = dict(x=3, xerr=1, y=2, yerr=9)
    return fitting_data, expected_columns, expected_indices


def case_no_columns_no_search():
    fitting_data = FittingData(COLUMNS, search=False)
    expected_columns = Columns(x=None, xerr=None, y=None, yerr=None)
    expected_indices = dict(x=None, xerr=None, y=None, yerr=None)
    return fitting_data, expected_columns, expected_indices


def case_x_and_y_no_search():
    fitting_data = FittingData(COLUMNS, x_column="c", y_column="f", search=False)
    expected_columns = Columns(x="c", xerr=None, y="f", yerr=None)
    expected_indices = dict(x=3, xerr=None, y=6, yerr=None)
    return fitting_data, expected_columns, expected_indices


def case_xerr_and_yerr_no_search():
    fitting_data = FittingData(COLUMNS, xerr_column="c", yerr_column="f", search=False)
    expected_columns = Columns(x=None, xerr="c", y=None, yerr="f")
    expected_indices = dict(x=None, xerr=3, y=None, yerr=6)
    return fitting_data, expected_columns, expected_indices


@parametrize_with_cases(
    argnames=["fitting_data", "expected_columns", "expected_indices"], cases=THIS_MODULE
)
def test_x_column(fitting_data, expected_columns, expected_indices):
    assert (
        expected_columns.x == fitting_data.x_column
    ), "X column name is different than expected"


@parametrize_with_cases(
    argnames=["fitting_data", "expected_columns", "expected_indices"], cases=THIS_MODULE
)
def test_x_index(fitting_data, expected_columns, expected_indices):
    assert (
        expected_indices["x"] == fitting_data.x_index
    ), "X index is different than expected"


@parametrize_with_cases(
    argnames=["fitting_data", "expected_columns", "expected_indices"], cases=THIS_MODULE
)
def test_x_data(fitting_data, expected_columns, expected_indices):
    if expected_columns.x is None:
        assert fitting_data.x is None
    else:
        assert COLUMNS[expected_columns.x] == pytest.approx(
            fitting_data.x
        ), "X is different than expected"


@parametrize_with_cases(
    argnames=["fitting_data", "expected_columns", "expected_indices"], cases=THIS_MODULE
)
def test_xerr_column(fitting_data, expected_columns, expected_indices):
    assert (
        expected_columns.xerr == fitting_data.xerr_column
    ), "X error column name is different than expected"


@parametrize_with_cases(
    argnames=["fitting_data", "expected_columns", "expected_indices"], cases=THIS_MODULE
)
def test_xerr_index(fitting_data, expected_columns, expected_indices):
    assert (
        expected_indices["xerr"] == fitting_data.xerr_index
    ), "X error index is different than expected"


@parametrize_with_cases(
    argnames=["fitting_data", "expected_columns", "expected_indices"], cases=THIS_MODULE
)
def test_xerr_data(fitting_data, expected_columns, expected_indices):
    if expected_columns.xerr is None:
        assert fitting_data.xerr is None
    else:
        assert COLUMNS[expected_columns.xerr] == pytest.approx(
            fitting_data.xerr
        ), "X error is different than expected"


@parametrize_with_cases(
    argnames=["fitting_data", "expected_columns", "expected_indices"], cases=THIS_MODULE
)
def test_y_column(fitting_data, expected_columns, expected_indices):
    assert (
        expected_columns.y == fitting_data.y_column
    ), "Y column name is different than expected"


@parametrize_with_cases(
    argnames=["fitting_data", "expected_columns", "expected_indices"], cases=THIS_MODULE
)
def test_y_index(fitting_data, expected_columns, expected_indices):
    assert (
        expected_indices["y"] == fitting_data.y_index
    ), "Y index is different than expected"


@parametrize_with_cases(
    argnames=["fitting_data", "expected_columns", "expected_indices"], cases=THIS_MODULE
)
def test_y_data(fitting_data, expected_columns, expected_indices):
    if expected_columns.y is None:
        assert fitting_data.y is None
    else:
        assert COLUMNS[expected_columns.y] == pytest.approx(
            fitting_data.y
        ), "Y is different than expected"


@parametrize_with_cases(
    argnames=["fitting_data", "expected_columns", "expected_indices"], cases=THIS_MODULE
)
def test_yerr_column(fitting_data, expected_columns, expected_indices):
    assert (
        expected_columns.yerr == fitting_data.yerr_column
    ), "Y error column name is different than expected"


@parametrize_with_cases(
    argnames=["fitting_data", "expected_columns", "expected_indices"], cases=THIS_MODULE
)
def test_yerr_index(fitting_data, expected_columns, expected_indices):
    assert (
        expected_indices["yerr"] == fitting_data.yerr_index
    ), "Y error index is different than expected"


@parametrize_with_cases(
    argnames=["fitting_data", "expected_columns", "expected_indices"], cases=THIS_MODULE
)
def test_yerr_data(fitting_data, expected_columns, expected_indices):
    if expected_columns.yerr is None:
        assert fitting_data.yerr is None
    else:
        assert COLUMNS[expected_columns.yerr] == pytest.approx(
            fitting_data.yerr
        ), "Y error is different than expected"


@parametrize_with_cases(
    argnames=["fitting_data", "expected_columns", "expected_indices"], cases=THIS_MODULE
)
def test_all_columns(fitting_data, expected_columns, expected_indices):
    assert (
        COLUMNS_NAMES == fitting_data.all_columns
    ), "Columns are different than expected"


@parametrize_with_cases(
    argnames=["fitting_data", "expected_columns", "expected_indices"], cases=THIS_MODULE
)
def test_used_columns(fitting_data, expected_columns, expected_indices):
    assert (
        fitting_data.used_columns == expected_columns
    ), "Used columns are different than expected"


@parametrize_with_cases(
    argnames=["fitting_data", "expected_columns", "expected_indices"], cases=THIS_MODULE
)
def test_iterate_used_columns(fitting_data, expected_columns, expected_indices):
    used_columns_list = list(fitting_data.used_columns)
    assert len(used_columns_list) == 4
    assert used_columns_list[0] == expected_columns.x
    assert used_columns_list[1] == expected_columns.xerr
    assert used_columns_list[2] == expected_columns.y
    assert used_columns_list[3] == expected_columns.yerr


@parametrize_with_cases(
    argnames=["fitting_data", "expected_columns", "expected_indices"], cases=THIS_MODULE
)
def test_data(fitting_data, expected_columns, expected_indices):
    assert COLUMNS_NAMES == list(
        fitting_data.data.keys()
    ), "Data keys are different than expected"
    for key, item in fitting_data.data.items():
        assert item == pytest.approx(
            COLUMNS[key]
        ), f"Value of {key} is different than expected."


@parametrize_with_cases(
    argnames=["fitting_data", "expected_columns", "expected_indices"], cases=THIS_MODULE
)
def test_records(fitting_data, expected_columns, expected_indices):
    for actual_record, expected_record in zip(CONTENT, fitting_data.all_records):
        assert_list_equal(actual_record, expected_record, rel=EPSILON)


@pytest.mark.parametrize("column", COLUMNS_OPTIONS)
def test_x_not_existing(column):
    with pytest.raises(
        FittingDataColumnExistenceError, match='^Could not find column "r" in data$'
    ):
        FittingData(COLUMNS, **{column: "r"})


def test_exception_risen_because_of_columns_length():
    data = deepcopy(COLUMNS)
    data["a"] = data["a"][:-2]
    with pytest.raises(
        FittingDataColumnsLengthError,
        match="^All columns in FittingData should have the same length$",
    ):
        FittingData(data=data)
