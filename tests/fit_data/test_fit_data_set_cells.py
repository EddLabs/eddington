import itertools
from copy import deepcopy

import numpy as np
import pytest
from pytest_cases import parametrize

from eddington import FitData
from eddington.exceptions import FitDataSetError
from tests.fit_data import COLUMNS, COLUMNS_NAMES, NUMBER_OF_RECORDS


@parametrize(
    argnames="record_number, column_name",
    argvalues=itertools.product(range(1, NUMBER_OF_RECORDS + 1), COLUMNS_NAMES),
)
def test_set_cell_allowed(record_number, column_name):
    fit_data = FitData(deepcopy(COLUMNS))
    value = np.random.uniform(0, 10)
    fit_data.set_cell(
        record_number=record_number,
        column_name=column_name,
        value=value,
    )

    for column in COLUMNS_NAMES:
        for i in range(NUMBER_OF_RECORDS):
            if column == column_name and i + 1 == record_number:
                assert fit_data.data[column][i] == value, "New value was not set"
            else:
                assert (
                    fit_data.data[column][i] == COLUMNS[column][i]
                ), "Data was change unexpectedly"


def test_set_cell_not_allowed_because_of_non_existing_column():
    fit_data = FitData(COLUMNS)
    value = np.random.uniform(0, 10)

    with pytest.raises(
        FitDataSetError, match='^Column name "I do not exist" does not exists$'
    ):
        fit_data.set_cell(0, "I do not exist", value)


def test_set_cell_not_allowed_because_of_non_existing_row():
    fit_data = FitData(COLUMNS)
    value = np.random.uniform(0, 10)
    column = np.random.choice(COLUMNS_NAMES)

    with pytest.raises(FitDataSetError, match="^Record number 13 does not exists$"):
        fit_data.set_cell(NUMBER_OF_RECORDS + 1, column, value)


def test_set_cell_not_allowed_because_of_non_float_value():
    fit_data = FitData(COLUMNS)
    column = np.random.choice(COLUMNS_NAMES)
    record = np.random.randint(1, NUMBER_OF_RECORDS + 1)
    value = "I'm not a float"

    with pytest.raises(
        FitDataSetError,
        match=(
            f'^The cell at record number:"{record}", column:"{column}"'
            f" has invalid syntax: I'm not a float.$"
        ),
    ):
        fit_data.set_cell(record, column, value)


@parametrize("i", range(len(COLUMNS_NAMES)))
def test_set_header_allowed(i):
    old_header = COLUMNS_NAMES[i]
    new_header = "new_header"
    fit_data = FitData(deepcopy(COLUMNS))
    fit_data.set_header(old_header, new_header)

    constant_headers = [header for header in COLUMNS_NAMES if header != old_header]
    for header in constant_headers:
        assert fit_data.data[header] == pytest.approx(
            COLUMNS[header]
        ), f'Header "{header}" has changed unexpectedly'
    assert fit_data.data[new_header] == pytest.approx(
        COLUMNS[old_header]
    ), f'Header "{new_header}" has changed unexpectedly'
    assert set(fit_data.all_columns) == set(
        constant_headers + [new_header]
    ), "Did not update all columns"


@parametrize("header_name", COLUMNS_NAMES)
def test_set_header_with_same_header_does_not_change_anything(header_name):
    fit_data = FitData(COLUMNS)
    fit_data.set_header(header_name, header_name)

    for header in COLUMNS_NAMES:
        assert fit_data.data[header] == pytest.approx(
            COLUMNS[header]
        ), f'Header "{header}" has changed unexpectedly'


@parametrize("header_name", COLUMNS_NAMES)
def test_header_cannot_be_set_to_be_empty(header_name):
    fit_data = FitData(COLUMNS)
    with pytest.raises(FitDataSetError, match="^Cannot set new header to be empty$"):
        fit_data.set_header(header_name, "")


@parametrize("header_name", COLUMNS_NAMES)
def test_header_cannot_already_exist(header_name):
    fit_data = FitData(COLUMNS)
    new_header = np.random.choice(
        [header for header in COLUMNS_NAMES if header != header_name]
    )
    with pytest.raises(
        FitDataSetError, match=f'^The column name:"{new_header}" is already used.$'
    ):
        fit_data.set_header(header_name, new_header)
