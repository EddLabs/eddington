import random

import numpy as np
import pytest
from pytest_cases import THIS_MODULE, parametrize, parametrize_with_cases

from eddington import FittingData
from eddington.exceptions import (
    FittingDataRecordIndexError,
    FittingDataRecordsSelectionError,
)
from tests.fitting_data import COLUMNS, CONTENT, NUMBER_OF_RECORDS, VALUES
from tests.util import assert_list_equal

EPSILON = 1e-3


def extract_values(column, indices):
    return [column[i - 1] for i in indices]


def case_unselect_one_record():
    fitting_data = FittingData(COLUMNS)
    fitting_data.unselect_record(2)
    return (
        fitting_data,
        [1] + list(range(3, NUMBER_OF_RECORDS + 1)),
    )


def case_unselect_two_records():
    fitting_data = FittingData(COLUMNS)
    fitting_data.unselect_record(2)
    fitting_data.unselect_record(5)
    return fitting_data, [1, 3, 4] + list(range(6, NUMBER_OF_RECORDS + 1))


def case_unselect_multiple():
    fitting_data = FittingData(COLUMNS)
    fitting_data.unselect_record(2)
    fitting_data.unselect_record(5)
    fitting_data.unselect_record(3)
    fitting_data.unselect_record(10)
    return fitting_data, [1, 4, 6, 7, 8, 9] + list(range(11, NUMBER_OF_RECORDS + 1))


def case_unselect_all():
    fitting_data = FittingData(COLUMNS)
    fitting_data.unselect_all_records()
    return fitting_data, []


def case_select_one_records():
    fitting_data = FittingData(COLUMNS)
    fitting_data.unselect_all_records()
    fitting_data.select_record(2)
    return fitting_data, [2]


def case_select_two_records():
    fitting_data = FittingData(COLUMNS)
    fitting_data.unselect_all_records()
    fitting_data.select_record(2)
    fitting_data.select_record(5)
    return fitting_data, [2, 5]


def case_select_multiple_records():
    fitting_data = FittingData(COLUMNS)
    fitting_data.unselect_all_records()
    fitting_data.select_record(2)
    fitting_data.select_record(5)
    fitting_data.select_record(3)
    fitting_data.select_record(10)
    return fitting_data, [2, 3, 5, 10]


def case_reselect_record():
    fitting_data = FittingData(COLUMNS)
    fitting_data.unselect_record(2)
    fitting_data.select_record(2)
    return fitting_data, list(range(1, NUMBER_OF_RECORDS + 1))


def case_select_all_records():
    fitting_data = FittingData(COLUMNS)
    fitting_data.unselect_all_records()
    fitting_data.select_record(2)
    fitting_data.select_record(5)
    fitting_data.select_all_records()
    return fitting_data, list(range(1, NUMBER_OF_RECORDS + 1))


def case_x_domain_lower_bound():
    xmin = 0.5
    fitting_data = FittingData(COLUMNS)
    selected_indices = [i for i, x in enumerate(fitting_data.x, start=1) if x >= xmin]
    fitting_data.select_by_x_domain(xmin=xmin)
    return fitting_data, selected_indices


def case_x_domain_upper_bound():
    xmax = 0.5
    fitting_data = FittingData(COLUMNS)
    selected_indices = [i for i, x in enumerate(fitting_data.x, start=1) if x <= xmax]
    fitting_data.select_by_x_domain(xmax=xmax)
    return fitting_data, selected_indices


def case_x_domain_both_bounds():
    xmin, xmax = 0.33, 0.66
    fitting_data = FittingData(COLUMNS)
    selected_indices = [
        i for i, x in enumerate(fitting_data.x, start=1) if xmin <= x <= xmax
    ]
    fitting_data.select_by_x_domain(xmin=xmin, xmax=xmax)
    return fitting_data, selected_indices


def case_x_domain_select_all():
    fitting_data = FittingData(COLUMNS)
    fitting_data.select_by_x_domain()
    return fitting_data, list(range(1, NUMBER_OF_RECORDS + 1))


def case_x_domain_update_selected_false():
    xmax = 0.5
    fitting_data = FittingData(COLUMNS)
    selected_indices = [i for i, x in enumerate(fitting_data.x, start=1) if x <= xmax]
    index = random.choice(selected_indices)
    fitting_data.unselect_record(index)
    fitting_data.select_by_x_domain(xmax=xmax)
    return fitting_data, selected_indices


def case_x_domain_update_selected_true():
    xmax = 0.5
    fitting_data = FittingData(COLUMNS)
    selected_indices = [i for i, x in enumerate(fitting_data.x, start=1) if x <= xmax]
    index = random.choice(selected_indices)
    selected_indices.remove(index)
    fitting_data.unselect_record(index)
    fitting_data.select_by_x_domain(xmax=xmax, update_selected=True)
    return fitting_data, selected_indices


def case_y_domain_lower_bound():
    ymin = 0.5
    fitting_data = FittingData(COLUMNS)
    selected_indices = [i for i, y in enumerate(fitting_data.y, start=1) if y >= ymin]
    fitting_data.select_by_y_domain(ymin=ymin)
    return fitting_data, selected_indices


def case_y_domain_upper_bound():
    ymax = 0.5
    fitting_data = FittingData(COLUMNS)
    selected_indices = [i for i, y in enumerate(fitting_data.y, start=1) if y <= ymax]
    fitting_data.select_by_y_domain(ymax=ymax)
    return fitting_data, selected_indices


def case_y_domain_both_bounds():
    ymin, ymax = 0.33, 0.66
    fitting_data = FittingData(COLUMNS)
    selected_indices = [
        i for i, y in enumerate(fitting_data.y, start=1) if ymin <= y <= ymax
    ]
    fitting_data.select_by_y_domain(ymin=ymin, ymax=ymax)
    return fitting_data, selected_indices


def case_y_domain_select_all():
    fitting_data = FittingData(COLUMNS)
    fitting_data.select_by_y_domain()
    return fitting_data, list(range(1, NUMBER_OF_RECORDS + 1))


def case_y_domain_update_selected_false():
    ymax = 0.5
    fitting_data = FittingData(COLUMNS)
    selected_indices = [i for i, y in enumerate(fitting_data.y, start=1) if y <= ymax]
    index = random.choice(selected_indices)
    fitting_data.unselect_record(index)
    fitting_data.select_by_y_domain(ymax=ymax)
    return fitting_data, selected_indices


def case_y_domain_update_selected_true():
    ymax = 0.5
    fitting_data = FittingData(COLUMNS)
    selected_indices = [i for i, y in enumerate(fitting_data.y, start=1) if y <= ymax]
    index = random.choice(selected_indices)
    selected_indices.remove(index)
    fitting_data.unselect_record(index)
    fitting_data.select_by_y_domain(ymax=ymax, update_selected=True)
    return fitting_data, selected_indices


def case_xy_domain_half_bounded():
    xmin, ymax = 0.33, 0.66
    fitting_data = FittingData(COLUMNS)
    selected_indices = [
        i
        for i, (x, y) in enumerate(zip(fitting_data.x, fitting_data.y), start=1)
        if xmin <= x and y <= ymax
    ]
    fitting_data.select_by_domains(xmin=xmin, ymax=ymax)
    return fitting_data, selected_indices


def case_xy_domain_update_selected_false():
    xmin, ymax = 0.33, 0.66
    fitting_data = FittingData(COLUMNS)
    selected_indices = [
        i
        for i, (x, y) in enumerate(zip(fitting_data.x, fitting_data.y), start=1)
        if xmin <= x and y <= ymax
    ]
    index = random.choice(selected_indices)
    fitting_data.unselect_record(index)
    fitting_data.select_by_domains(xmin=xmin, ymax=ymax)
    return fitting_data, selected_indices


def case_xy_domain_update_selected_true():
    xmin, ymax = 0.33, 0.66
    fitting_data = FittingData(COLUMNS)
    selected_indices = [
        i
        for i, (x, y) in enumerate(zip(fitting_data.x, fitting_data.y), start=1)
        if xmin <= x and y <= ymax
    ]
    index = random.choice(selected_indices)
    selected_indices.remove(index)
    fitting_data.unselect_record(index)
    fitting_data.select_by_domains(xmin=xmin, ymax=ymax, update_selected=True)
    return fitting_data, selected_indices


def case_set_selected_records():
    fitting_data = FittingData(COLUMNS)
    fitting_data.records_indices = [False, True, False, False, True] + [False] * (
        NUMBER_OF_RECORDS - 5
    )
    return fitting_data, [2, 5]


@parametrize_with_cases(argnames="fitting_data, selected_indices", cases=THIS_MODULE)
def test_x(fitting_data, selected_indices):
    expected_x = extract_values(VALUES[0], selected_indices)
    assert fitting_data.x.shape == np.shape(
        selected_indices
    ), "X shape is different than expected"
    assert fitting_data.x == pytest.approx(expected_x), "X is different than expected"


@parametrize_with_cases(argnames="fitting_data, selected_indices", cases=THIS_MODULE)
def test_xerr(fitting_data, selected_indices):
    assert fitting_data.xerr == pytest.approx(
        extract_values(VALUES[1], selected_indices)
    ), "X error is different than expected"


@parametrize_with_cases(argnames="fitting_data, selected_indices", cases=THIS_MODULE)
def test_y(fitting_data, selected_indices):
    assert fitting_data.y == pytest.approx(
        extract_values(VALUES[2], selected_indices)
    ), "Y is different than expected"


@parametrize_with_cases(argnames="fitting_data, selected_indices", cases=THIS_MODULE)
def test_yerr(fitting_data, selected_indices):
    assert fitting_data.yerr == pytest.approx(
        extract_values(VALUES[3], selected_indices)
    ), "Y error is different than expected"


@parametrize_with_cases(argnames="fitting_data, selected_indices", cases=THIS_MODULE)
def test_is_selected(fitting_data, selected_indices):
    for i in range(1, fitting_data.number_of_records + 1):
        if i in selected_indices:
            assert fitting_data.is_selected(i), f"Record {i} was not selected"
        else:
            assert not fitting_data.is_selected(i), f"Record {i} was selected"


@parametrize_with_cases(argnames="fitting_data, selected_indices", cases=THIS_MODULE)
def test_records(fitting_data, selected_indices):
    selected_records = [
        record for i, record in enumerate(CONTENT, start=1) if i in selected_indices
    ]
    for actual_record, expected_record in zip(selected_records, fitting_data.records):
        assert_list_equal(actual_record, expected_record, rel=EPSILON)


def test_set_selection_with_different_size():
    fitting_data = FittingData(COLUMNS)

    def set_records_indices():
        fitting_data.records_indices = [False, False, True]

    with pytest.raises(
        FittingDataRecordsSelectionError,
        match=f"^Should select {NUMBER_OF_RECORDS} records, only 3 selected.$",
    ):
        set_records_indices()


def test_set_selection_with_non_boolean_values():
    fitting_data = FittingData(COLUMNS)

    def set_records_indices():
        fitting_data.records_indices = [False, False, "dummy"] + [True] * (
            NUMBER_OF_RECORDS - 3
        )

    with pytest.raises(
        FittingDataRecordsSelectionError,
        match="^When setting record indices, all values should be booleans.$",
    ):
        set_records_indices()


@parametrize("index", [0, -1, NUMBER_OF_RECORDS + 1])
def test_select_record_with_invalid_index(index):
    fitting_data = FittingData(COLUMNS)

    with pytest.raises(
        FittingDataRecordIndexError,
        match=(
            f"^Could not find record with index {index} in data. "
            f"Index should be between 1 and {NUMBER_OF_RECORDS}.$"
        ),
    ):
        fitting_data.select_record(index)


@parametrize("index", [0, -1, NUMBER_OF_RECORDS + 1])
def test_unselect_record_with_invalid_index(index):
    fitting_data = FittingData(COLUMNS)

    with pytest.raises(
        FittingDataRecordIndexError,
        match=(
            f"^Could not find record with index {index} in data. "
            f"Index should be between 1 and {NUMBER_OF_RECORDS}.$"
        ),
    ):
        fitting_data.unselect_record(index)
