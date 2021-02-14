import itertools
import random
from copy import deepcopy

import numpy as np
import pytest
from pytest_cases import parametrize

from eddington import FittingData
from eddington.exceptions import (
    FittingDataColumnExistenceError,
    FittingDataRecordIndexError,
    FittingDataSetError,
)
from eddington.interval import Interval
from tests.fitting_data import COLUMNS, COLUMNS_NAMES, NUMBER_OF_RECORDS
from tests.util import assert_float_equal, assert_numpy_array_equal, \
    random_selected_records

EPSILON = 1e-5


@parametrize(
    argnames="record_number, column_name",
    argvalues=itertools.product(range(1, NUMBER_OF_RECORDS + 1), COLUMNS_NAMES),
)
def test_set_cell_allowed(record_number, column_name):
    fitting_data = FittingData(deepcopy(COLUMNS))
    value = np.random.uniform(0, 10)
    fitting_data.set_cell(
        column_name=column_name,
        index=record_number,
        value=value,
    )

    for column in COLUMNS_NAMES:
        for i in range(NUMBER_OF_RECORDS):
            if column == column_name and i + 1 == record_number:
                assert fitting_data.data[column][i] == value, "New value was not set"
            else:
                assert (
                    fitting_data.data[column][i] == COLUMNS[column][i]
                ), "Data was change unexpectedly"


def test_set_cell_not_allowed_because_of_non_existing_column():
    fitting_data = FittingData(COLUMNS)
    value = np.random.uniform(0, 10)

    with pytest.raises(
        FittingDataColumnExistenceError,
        match='^Could not find column "I do not exist" in data$',
    ):
        fitting_data.set_cell(column_name="I do not exist", index=1, value=value)


def test_set_cell_not_allowed_because_of_non_existing_row():
    fitting_data = FittingData(COLUMNS)
    value = np.random.uniform(0, 10)
    column = np.random.choice(COLUMNS_NAMES)

    with pytest.raises(
        FittingDataRecordIndexError,
        match="^Could not find record with index 13 in data. "
        "Index should be between 1 and 12.$",
    ):
        fitting_data.set_cell(
            column_name=column, index=NUMBER_OF_RECORDS + 1, value=value
        )


def test_set_cell_not_allowed_because_of_non_float_value():
    fitting_data = FittingData(COLUMNS)
    column = np.random.choice(COLUMNS_NAMES)
    record = np.random.randint(1, NUMBER_OF_RECORDS + 1)
    value = "I'm not a float"

    with pytest.raises(
        FittingDataSetError,
        match=(
            f'^The cell at record number "{record}", column "{column}",'
            f" has invalid syntax: I'm not a float.$"
        ),
    ):
        fitting_data.set_cell(column_name=column, index=record, value=value)


@parametrize("i", range(len(COLUMNS_NAMES)))
def test_set_header_allowed(i):
    old_header = COLUMNS_NAMES[i]
    new_header = "new_header"
    fitting_data = FittingData(deepcopy(COLUMNS))
    fitting_data.set_header(old_header, new_header)

    constant_headers = [header for header in COLUMNS_NAMES if header != old_header]
    for header in constant_headers:
        assert fitting_data.data[header] == pytest.approx(
            COLUMNS[header]
        ), f'Header "{header}" has changed unexpectedly'
    assert fitting_data.data[new_header] == pytest.approx(
        COLUMNS[old_header]
    ), f'Header "{new_header}" has changed unexpectedly'
    assert set(fitting_data.all_columns) == set(
        constant_headers + [new_header]
    ), "Did not update all columns"


@parametrize("header_name", COLUMNS_NAMES)
def test_set_header_with_same_header_does_not_change_anything(header_name):
    fitting_data = FittingData(COLUMNS)
    fitting_data.set_header(header_name, header_name)

    for header in COLUMNS_NAMES:
        assert fitting_data.data[header] == pytest.approx(
            COLUMNS[header]
        ), f'Header "{header}" has changed unexpectedly'


@parametrize("header_name", COLUMNS_NAMES)
def test_header_cannot_be_set_to_be_empty(header_name):
    fitting_data = FittingData(COLUMNS)
    with pytest.raises(
        FittingDataSetError, match="^Cannot set new header to be empty$"
    ):
        fitting_data.set_header(header_name, "")


@parametrize("header_name", COLUMNS_NAMES)
def test_header_cannot_already_exist(header_name):
    fitting_data = FittingData(COLUMNS)
    new_header = np.random.choice(
        [header for header in COLUMNS_NAMES if header != header_name]
    )
    with pytest.raises(
        FittingDataSetError, match=f'^The column name "{new_header}" is already used.$'
    ):
        fitting_data.set_header(header_name, new_header)


@parametrize("column_type", ["x_column", "xerr_column", "y_column", "yerr_column"])
@parametrize("header_name", COLUMNS_NAMES)
def test_rename_selected_column(column_type, header_name):
    fitting_data = FittingData(COLUMNS, **{column_type: header_name, "search": False})
    new_header = "new header"
    assert getattr(fitting_data, column_type) == header_name
    fitting_data.set_header(header_name, new_header)
    assert getattr(fitting_data, column_type) == new_header


def test_set_header_with_non_existing_name():
    fitting_data = FittingData(COLUMNS)

    with pytest.raises(
        FittingDataColumnExistenceError,
        match='^Could not find column "I do not exist" in data$',
    ):
        fitting_data.set_header(old_column="I do not exist", new_column="new value")


@parametrize("header_name", COLUMNS_NAMES)
def test_get_column_data_only_selected(header_name):
    fitting_data = FittingData(COLUMNS)
    records_indices = [random.randint(0, 1) == 1 for _ in range(NUMBER_OF_RECORDS)]
    fitting_data.records_indices = records_indices
    assert_numpy_array_equal(
        fitting_data.column_data(header_name),
        COLUMNS[header_name][records_indices],
        rel=EPSILON,
    )


@parametrize("header_name", COLUMNS_NAMES)
def test_get_column_data_all_records(header_name):
    fitting_data = FittingData(COLUMNS)
    fitting_data.records_indices = [
        random.randint(0, 1) == 1 for _ in range(NUMBER_OF_RECORDS)
    ]
    assert_numpy_array_equal(
        fitting_data.column_data(header_name, only_selected=False),
        COLUMNS[header_name],
        rel=EPSILON,
    )


def test_column_data_get_non_existing_column():
    fitting_data = FittingData(COLUMNS)

    with pytest.raises(
        FittingDataColumnExistenceError,
        match='^Could not find column "I do not exist" in data$',
    ):
        fitting_data.column_data("I do not exist")


@parametrize("header_name", COLUMNS_NAMES)
def test_get_cell_data(header_name):
    fitting_data = FittingData(COLUMNS)
    index = random.randint(1, NUMBER_OF_RECORDS)
    assert_float_equal(
        fitting_data.cell_data(column_name=header_name, index=index),
        COLUMNS[header_name][index - 1],
        rel=EPSILON,
    )


def test_cell_data_get_non_existing_column():
    fitting_data = FittingData(COLUMNS)

    with pytest.raises(
        FittingDataColumnExistenceError,
        match='^Could not find column "I do not exist" in data$',
    ):
        fitting_data.cell_data("I do not exist", index=1)


def test_cell_data_get_non_existing_record_index():
    fitting_data = FittingData(COLUMNS)

    with pytest.raises(
        FittingDataRecordIndexError,
        match=(
            "^Could not find record with index -4 in data. "
            "Index should be between 1 and 12.$"
        ),
    ):
        fitting_data.cell_data(column_name="b", index=-4)


@parametrize("header_name", COLUMNS_NAMES)
def test_get_column_domain_only_selected(header_name):
    fitting_data = FittingData(COLUMNS)
    values = fitting_data.column_data(header_name)
    domain = fitting_data.column_domain(header_name)
    assert isinstance(domain, Interval)
    assert_float_equal(domain.min_val, np.min(values), rel=EPSILON)
    assert_float_equal(domain.max_val, np.max(values), rel=EPSILON)


@parametrize("header_name", COLUMNS_NAMES)
def test_get_column_domain_all_records(header_name):
    fitting_data = FittingData(COLUMNS)
    fitting_data.records_indices = random_selected_records(
        records_num=NUMBER_OF_RECORDS, min_selected=1
    )
    values = fitting_data.column_data(header_name, only_selected=False)
    domain = fitting_data.column_domain(header_name, only_selected=False)
    assert isinstance(domain, Interval)
    assert_float_equal(domain.min_val, np.min(values), rel=EPSILON)
    assert_float_equal(domain.max_val, np.max(values), rel=EPSILON)


@parametrize("header_name", COLUMNS_NAMES)
def test_get_x_domain(header_name):
    fitting_data = FittingData(COLUMNS, x_column=header_name, search=False)
    fitting_data.records_indices = random_selected_records(
        records_num=NUMBER_OF_RECORDS, min_selected=1
    )
    assert fitting_data.x_domain == fitting_data.column_domain(column_name=header_name)


@parametrize("header_name", COLUMNS_NAMES)
def test_get_y_domain(header_name):
    fitting_data = FittingData(COLUMNS, y_column=header_name, search=False)
    fitting_data.records_indices = random_selected_records(
        records_num=NUMBER_OF_RECORDS, min_selected=1
    )
    assert fitting_data.y_domain == fitting_data.column_domain(column_name=header_name)
