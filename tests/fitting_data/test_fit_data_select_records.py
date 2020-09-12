import numpy as np
import pytest
from pytest_cases import THIS_MODULE, parametrize_with_cases

from eddington import FittingData
from eddington.exceptions import FittingDataColumnsSelectionError
from tests.fitting_data import COLUMNS, NUMBER_OF_RECORDS, VALUES


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
    for i in range(1, fitting_data.length + 1):
        if i in selected_indices:
            assert fitting_data.is_selected(i), f"Record {i} was not selected"
        else:
            assert not fitting_data.is_selected(i), f"Record {i} was selected"


def test_set_selection_with_different_size():
    fitting_data = FittingData(COLUMNS)

    def set_records_indices():
        fitting_data.records_indices = [False, False, True]

    with pytest.raises(
        FittingDataColumnsSelectionError,
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
        FittingDataColumnsSelectionError,
        match="^When setting record indices, all values should be booleans.$",
    ):
        set_records_indices()
