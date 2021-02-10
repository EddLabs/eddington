import random

import pytest

from eddington import (
    FittingData,
    FittingDataColumnExistenceError,
    FittingDataColumnIndexError,
)
from tests.fitting_data import COLUMNS, COLUMNS_NAMES, NUMBER_OF_COLUMNS


def random_column_and_index():
    return random.choice(list(enumerate(COLUMNS_NAMES, start=1)))


def test_x_column_select_column_by_name():
    fitting_data = FittingData(COLUMNS, search=False)
    column_index, column_name = random_column_and_index()
    fitting_data.x_column = column_name
    assert fitting_data.x_column == column_name
    assert fitting_data.x_index == column_index


def test_x_column_select_column_by_index():
    fitting_data = FittingData(COLUMNS, search=False)
    column_index, column_name = random_column_and_index()
    fitting_data.x_index = column_index
    assert fitting_data.x_column == column_name
    assert fitting_data.x_index == column_index


def test_xerr_column_select_column_by_name():
    fitting_data = FittingData(COLUMNS, search=False)
    column_index, column_name = random_column_and_index()
    fitting_data.xerr_column = column_name
    assert fitting_data.xerr_column == column_name
    assert fitting_data.xerr_index == column_index


def test_xerr_column_select_column_by_index():
    fitting_data = FittingData(COLUMNS, search=False)
    column_index, column_name = random_column_and_index()
    fitting_data.xerr_index = column_index
    assert fitting_data.xerr_column == column_name
    assert fitting_data.xerr_index == column_index


def test_y_column_select_column_by_name():
    fitting_data = FittingData(COLUMNS, search=False)
    column_index, column_name = random_column_and_index()
    fitting_data.y_column = column_name
    assert fitting_data.y_column == column_name
    assert fitting_data.y_index == column_index


def test_y_column_select_column_by_index():
    fitting_data = FittingData(COLUMNS, search=False)
    column_index, column_name = random_column_and_index()
    fitting_data.y_index = column_index
    assert fitting_data.y_column == column_name
    assert fitting_data.y_index == column_index


def test_yerr_column_select_column_by_name():
    fitting_data = FittingData(COLUMNS, search=False)
    column_index, column_name = random_column_and_index()
    fitting_data.yerr_column = column_name
    assert fitting_data.yerr_column == column_name
    assert fitting_data.yerr_index == column_index


def test_yerr_column_select_column_by_index():
    fitting_data = FittingData(COLUMNS, search=False)
    column_index, column_name = random_column_and_index()
    fitting_data.yerr_index = column_index
    assert fitting_data.yerr_column == column_name
    assert fitting_data.yerr_index == column_index


def test_x_column_select_none():
    fitting_data = FittingData(COLUMNS, search=True)
    fitting_data.x_column = None
    assert fitting_data.x_column is None
    assert fitting_data.x_index is None


def test_x_index_select_none():
    fitting_data = FittingData(COLUMNS, search=True)
    fitting_data.x_index = None
    assert fitting_data.x_column is None
    assert fitting_data.x_index is None


def test_xerr_column_select_none():
    fitting_data = FittingData(COLUMNS, search=True)
    fitting_data.xerr_column = None
    assert fitting_data.xerr_column is None
    assert fitting_data.xerr_index is None


def test_xerr_index_select_none():
    fitting_data = FittingData(COLUMNS, search=True)
    fitting_data.xerr_index = None
    assert fitting_data.xerr_column is None
    assert fitting_data.xerr_index is None


def test_y_column_select_none():
    fitting_data = FittingData(COLUMNS, search=True)
    fitting_data.y_column = None
    assert fitting_data.y_column is None
    assert fitting_data.y_index is None


def test_y_index_select_none():
    fitting_data = FittingData(COLUMNS, search=True)
    fitting_data.y_index = None
    assert fitting_data.y_column is None
    assert fitting_data.y_index is None


def test_yerr_column_select_none():
    fitting_data = FittingData(COLUMNS, search=True)
    fitting_data.yerr_column = None
    assert fitting_data.yerr_column is None
    assert fitting_data.yerr_index is None


def test_yerr_index_select_none():
    fitting_data = FittingData(COLUMNS, search=True)
    fitting_data.yerr_index = None
    assert fitting_data.yerr_column is None
    assert fitting_data.yerr_index is None


def test_x_column_select_no_existing_column_raises_exception():
    fitting_data = FittingData(COLUMNS, search=False)

    with pytest.raises(
        FittingDataColumnExistenceError,
        match='^Could not find column "I do not exist!" in data$',
    ):

        fitting_data.x_column = "I do not exist!"


def test_x_index_select_zero_raises_exception():
    fitting_data = FittingData(COLUMNS, search=False)

    with pytest.raises(
        FittingDataColumnIndexError,
        match=(
            "^No column number 0 in data. "
            f"index should be between 1 and {NUMBER_OF_COLUMNS}$"
        ),
    ):

        fitting_data.x_index = 0


def test_x_index_select_too_big_index_raises_exception():
    fitting_data = FittingData(COLUMNS, search=False)

    with pytest.raises(
        FittingDataColumnIndexError,
        match=(
            f"^No column number {NUMBER_OF_COLUMNS + 1} in data. "
            f"index should be between 1 and {NUMBER_OF_COLUMNS}$"
        ),
    ):

        fitting_data.x_index = NUMBER_OF_COLUMNS + 1


def test_xerr_column_select_no_existing_column_raises_exception():
    fitting_data = FittingData(COLUMNS, search=False)

    with pytest.raises(
        FittingDataColumnExistenceError,
        match='^Could not find column "I do not exist!" in data$',
    ):

        fitting_data.xerr_column = "I do not exist!"


def test_xerr_index_select_zero_raises_exception():
    fitting_data = FittingData(COLUMNS, search=False)

    with pytest.raises(
        FittingDataColumnIndexError,
        match=(
            "^No column number 0 in data. "
            f"index should be between 1 and {NUMBER_OF_COLUMNS}$"
        ),
    ):

        fitting_data.xerr_index = 0


def test_xerr_index_select_too_big_index_raises_exception():
    fitting_data = FittingData(COLUMNS, search=False)

    with pytest.raises(
        FittingDataColumnIndexError,
        match=(
            f"^No column number {NUMBER_OF_COLUMNS + 1} in data. "
            f"index should be between 1 and {NUMBER_OF_COLUMNS}$"
        ),
    ):

        fitting_data.xerr_index = NUMBER_OF_COLUMNS + 1


def test_y_column_select_no_existing_column_raises_exception():
    fitting_data = FittingData(COLUMNS, search=False)

    with pytest.raises(
        FittingDataColumnExistenceError,
        match='^Could not find column "I do not exist!" in data$',
    ):

        fitting_data.y_column = "I do not exist!"


def test_y_index_select_zero_raises_exception():
    fitting_data = FittingData(COLUMNS, search=False)

    with pytest.raises(
        FittingDataColumnIndexError,
        match=(
            "^No column number 0 in data. "
            f"index should be between 1 and {NUMBER_OF_COLUMNS}$"
        ),
    ):

        fitting_data.y_index = 0


def test_y_index_select_too_big_index_raises_exception():
    fitting_data = FittingData(COLUMNS, search=False)

    with pytest.raises(
        FittingDataColumnIndexError,
        match=(
            f"^No column number {NUMBER_OF_COLUMNS + 1} in data. "
            f"index should be between 1 and {NUMBER_OF_COLUMNS}$"
        ),
    ):

        fitting_data.y_index = NUMBER_OF_COLUMNS + 1


def test_yerr_column_select_no_existing_column_raises_exception():
    fitting_data = FittingData(COLUMNS, search=False)

    with pytest.raises(
        FittingDataColumnExistenceError,
        match='^Could not find column "I do not exist!" in data$',
    ):

        fitting_data.yerr_column = "I do not exist!"


def test_yerr_index_select_zero_raises_exception():
    fitting_data = FittingData(COLUMNS, search=False)

    with pytest.raises(
        FittingDataColumnIndexError,
        match=(
            "^No column number 0 in data. "
            f"index should be between 1 and {NUMBER_OF_COLUMNS}$"
        ),
    ):

        fitting_data.yerr_index = 0


def test_yerr_index_select_too_big_index_raises_exception():
    fitting_data = FittingData(COLUMNS, search=False)

    with pytest.raises(
        FittingDataColumnIndexError,
        match=(
            f"^No column number {NUMBER_OF_COLUMNS + 1} in data. "
            f"index should be between 1 and {NUMBER_OF_COLUMNS}$"
        ),
    ):

        fitting_data.yerr_index = NUMBER_OF_COLUMNS + 1
