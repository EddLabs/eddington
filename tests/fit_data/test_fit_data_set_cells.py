import pytest

from eddington import FitData
from eddington.exceptions import FitDataInvalidSyntax, FitDataColumnAlreadyExists

from tests.fit_data import COLUMNS, NUMBER_OF_RECORDS

import numpy as np
import string


def random_name(n):
    return "".join(np.random.choice(string.ascii_letters) for i in range(n))


def test_set_cell_allowed():
    fit_data = FitData(COLUMNS)
    rows = np.random.randint(1, NUMBER_OF_RECORDS, size=(4, 7))
    values = np.random.rand(4, 7)

    for i in range(4):
        for j in range(7):
            fit_data.set_cell(rows[i][j], COLUMNS.keys()[i], values[i][j])

    assert (
        fit_data.x[rows] == values[0]
    ), f'Some of the cells did not change as expected:\nactual values:"{fit_data.x[rows]}"\nexpected values:"{values[0]}"'
    assert (
        fit_data.xerr[rows] == values[1]
    ), f'Some of the cells did not change as expected:\nactual values:"{fit_data.x_err[rows]}"\nexpected values:"{values[1]}"'
    assert (
        fit_data.y[rows] == values[2]
    ), f'Some of the cells did not change as expected:\nactual values:"{fit_data.y[rows]}"\nexpected values:"{values[2]}"'
    assert (
        fit_data.yerr[rows] == values[3]
    ), f'Some of the cells did not change as expected:\nactual values:"{fit_data.y_err[rows]}"\nexpected values:"{values[3]}"'


def test_set_cell_not_allowed():
    fit_data = FitData(COLUMNS)
    rows = np.random.randint(1, NUMBER_OF_RECORDS, size=(4, 2))
    values = [
        [
            random_name(3),
            "".join(np.random.choice(string.punctuation) for i in range(3)),
        ]
        for i in range(4)
    ]

    for i in range(4):
        for j in range(2):
            with pytest.raises(FitDataInvalidSyntax):
                fit_data.set_cell(rows[i][j], COLUMNS.keys()[i], values[i][j])


def test_set_header_allowed():
    fit_data = FitData(COLUMNS)

    new_x = random_name(3)
    new_y = random_name(4)
    new_xerr = random_name(5)
    new_yerr = random_name(6)
    fit_data.set_header(fit_data.x_column, new_x)
    fit_data.set_header(fit_data.y_column, new_y)
    fit_data.set_header(fit_data.xerr_column, new_xerr)
    fit_data.set_header(fit_data.yerr_column, new_yerr)

    assert (
        fit_data.x_column == new_x
    ), f'Column name did not change as expected:\nactual name:"{fit_data.x_column}"\nexpected values:"{new_x}"'
    assert (
        fit_data.x_err_column == new_x_err
    ), f'Column name did not change as expected:\nactual name:"{fit_data.x_err_column}"\nexpected values:"{new_x_err}"'
    assert (
        fit_data.y_column == new_y
    ), f'Column name did not change as expected:\nactual name:"{fit_data.y_column}"\nexpected values:"{new_y}"'
    assert (
        fit_data.y_err_column == new_y_err
    ), f'Column name did not change as expected:\nactual name:"{fit_data.y_err_column}"\nexpected values:"{new_y_err}"'


def test_set_header_not_allowed():
    fit_data = FitData(COLUMNS)
    old_x = fit_data.x_column

    with pytest.raises(FitDataColumnAlreadyExists):
        fit_data.set_header(fit_data.x_column, fit_data.y_column)

    assert (
        fit_data.x_column == old_x
    ), "Column name did not change back to original value"


def test_set_header_bad_after_good():
    fit_data = FitData(COLUMNS)
    old_x = fit_data.x_column
    new_x = "".join(np.random.choice(string.ascii_letters) for i in range(3))

    with pytest.raises(FitDataColumnAlreadyExists):
        fit_data.set_header(fit_data.x_column, fit_data.y_column)

    assert (
        fit_data.x_column == new_x
    ), f'Column name did not change as expected:\nactual name:"{fit_data.x_column}"\nexpected values:"{new_x}"'
    assert (
        fit_data.x_column != old_x
    ), f"Column name changed to original value instead of the new one after failing setting"
