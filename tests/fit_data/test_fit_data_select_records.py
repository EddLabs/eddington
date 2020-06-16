import pytest
from pytest_cases import cases_data, THIS_MODULE

import numpy as np
from eddington_core import FitData
from eddington_core.exceptions import FitDataColumnsSelectionError

from tests.fit_data import COLUMNS, NUMBER_OF_RECORDS, VALUES


def extract_values(column, indices):
    return [column[i - 1] for i in indices]


def case_unselect_one_record():
    fit_data = FitData(COLUMNS)
    fit_data.unselect_record(2)
    return (
        fit_data,
        [1] + list(range(3, NUMBER_OF_RECORDS + 1)),
    )


def case_unselect_two_records():
    fit_data = FitData(COLUMNS)
    fit_data.unselect_record(2)
    fit_data.unselect_record(5)
    return fit_data, [1, 3, 4] + list(range(6, NUMBER_OF_RECORDS + 1))


def case_unselect_multiple():
    fit_data = FitData(COLUMNS)
    fit_data.unselect_record(2)
    fit_data.unselect_record(5)
    fit_data.unselect_record(3)
    fit_data.unselect_record(10)
    return fit_data, [1, 4, 6, 7, 8, 9] + list(range(11, NUMBER_OF_RECORDS + 1))


def case_unselect_all():
    fit_data = FitData(COLUMNS)
    fit_data.unselect_all_records()
    return fit_data, []


def case_select_one_records():
    fit_data = FitData(COLUMNS)
    fit_data.unselect_all_records()
    fit_data.select_record(2)
    return fit_data, [2]


def case_select_two_records():
    fit_data = FitData(COLUMNS)
    fit_data.unselect_all_records()
    fit_data.select_record(2)
    fit_data.select_record(5)
    return fit_data, [2, 5]


def case_select_multiple_records():
    fit_data = FitData(COLUMNS)
    fit_data.unselect_all_records()
    fit_data.select_record(2)
    fit_data.select_record(5)
    fit_data.select_record(3)
    fit_data.select_record(10)
    return fit_data, [2, 3, 5, 10]


def case_reselect_record():
    fit_data = FitData(COLUMNS)
    fit_data.unselect_record(2)
    fit_data.select_record(2)
    return fit_data, list(range(1, NUMBER_OF_RECORDS + 1))


def case_select_all_records():
    fit_data = FitData(COLUMNS)
    fit_data.unselect_all_records()
    fit_data.select_record(2)
    fit_data.select_record(5)
    fit_data.select_all_records()
    return fit_data, list(range(1, NUMBER_OF_RECORDS + 1))


def case_set_selected_records():
    fit_data = FitData(COLUMNS)
    fit_data.records_indices = [False, True, False, False, True] + [False] * (
        NUMBER_OF_RECORDS - 5
    )
    return fit_data, [2, 5]


@cases_data(module=THIS_MODULE)
def test_x(case_data):
    fit_data, selected_indices = case_data.get()
    expected_x = extract_values(VALUES[0], selected_indices)
    assert fit_data.x.shape == np.shape(
        selected_indices
    ), "X shape is different than expected"
    assert fit_data.x == pytest.approx(expected_x), "X is different than expected"


@cases_data(module=THIS_MODULE)
def test_xerr(case_data):
    fit_data, selected_indices = case_data.get()
    assert fit_data.xerr == pytest.approx(
        extract_values(VALUES[1], selected_indices)
    ), "X error is different than expected"


@cases_data(module=THIS_MODULE)
def test_y(case_data):
    fit_data, selected_indices = case_data.get()
    assert fit_data.y == pytest.approx(
        extract_values(VALUES[2], selected_indices)
    ), "Y is different than expected"


@cases_data(module=THIS_MODULE)
def test_yerr(case_data):
    fit_data, selected_indices = case_data.get()
    assert fit_data.yerr == pytest.approx(
        extract_values(VALUES[3], selected_indices)
    ), "Y error is different than expected"


@cases_data(module=THIS_MODULE)
def test_is_selected(case_data):
    fit_data, selected_indices = case_data.get()
    for i in range(1, fit_data.length + 1):
        if i in selected_indices:
            assert fit_data.is_selected(i), f"Record {i} was not selected"
        else:
            assert not fit_data.is_selected(i), f"Record {i} was selected"


def test_set_selection_with_different_size():
    fit_data = FitData(COLUMNS)

    def set_records_indices():
        fit_data.records_indices = [False, False, True]

    with pytest.raises(
        FitDataColumnsSelectionError,
        match=f"^Should select {NUMBER_OF_RECORDS} records, only 3 selected.$",
    ):
        set_records_indices()


def test_set_selection_with_non_boolean_values():
    fit_data = FitData(COLUMNS)

    def set_records_indices():
        fit_data.records_indices = [False, False, "dummy"] + [True] * (
            NUMBER_OF_RECORDS - 3
        )

    with pytest.raises(
        FitDataColumnsSelectionError,
        match="^When setting record indices, all values should be booleans.$",
    ):
        set_records_indices()
