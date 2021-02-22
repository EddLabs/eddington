import random
from collections import OrderedDict

import numpy as np
import pytest
from pytest_cases import THIS_MODULE, case, parametrize, parametrize_with_cases

from eddington import FittingData
from eddington.exceptions import (
    FittingDataRecordIndexError,
    FittingDataRecordsSelectionError,
)
from tests.util import assert_list_equal


NUMBER_OF_RECORDS = 10

EPSILON = 1e-3
ALL_SELECTED = "all_selected"
PARTIALLY_SELECTED = "partially_selected"
NON_SELECTED = "non_selected"


def make_data():
    x = np.arange(1, NUMBER_OF_RECORDS + 1)
    y = np.arange(1, NUMBER_OF_RECORDS + 1)
    np.random.shuffle(y)
    xerr = np.random.uniform(size=NUMBER_OF_RECORDS)
    yerr = np.random.uniform(size=NUMBER_OF_RECORDS)
    return OrderedDict(
        [
            ("x", x),
            ("xerr", xerr),
            ("y", y),
            ("yerr", yerr),
        ]
    )


def extract_values(column, indices):
    return [column[i - 1] for i in indices]


@case(tags=[ALL_SELECTED])
def case_default_selected():
    raw_data = make_data()
    fitting_data = FittingData(raw_data)
    return fitting_data, raw_data, list(range(1, NUMBER_OF_RECORDS + 1))


@case(tags=[PARTIALLY_SELECTED])
def case_unselect_one_record():
    raw_data = make_data()
    fitting_data = FittingData(raw_data)
    fitting_data.unselect_record(2)
    return (
        fitting_data, raw_data,
        [1] + list(range(3, NUMBER_OF_RECORDS + 1)),
    )


@case(tags=[PARTIALLY_SELECTED])
def case_unselect_two_records():
    raw_data = make_data()
    fitting_data = FittingData(raw_data)
    fitting_data.unselect_record(2)
    fitting_data.unselect_record(5)
    return fitting_data, raw_data, [1, 3, 4] + list(range(6, NUMBER_OF_RECORDS + 1))


@case(tags=[PARTIALLY_SELECTED])
def case_unselect_multiple():
    raw_data = make_data()
    fitting_data = FittingData(raw_data)
    fitting_data.unselect_record(2)
    fitting_data.unselect_record(5)
    fitting_data.unselect_record(3)
    fitting_data.unselect_record(10)
    return fitting_data, raw_data, [1, 4, 6, 7, 8, 9] + list(range(11, NUMBER_OF_RECORDS + 1))


@case(tags=[NON_SELECTED])
def case_unselect_all():
    raw_data = make_data()
    fitting_data = FittingData(raw_data)
    fitting_data.unselect_all_records()
    return fitting_data, raw_data, []


@case(tags=[PARTIALLY_SELECTED])
def case_select_one_records():
    raw_data = make_data()
    fitting_data = FittingData(raw_data)
    fitting_data.unselect_all_records()
    fitting_data.select_record(2)
    return fitting_data, raw_data, [2]


@case(tags=[PARTIALLY_SELECTED])
def case_select_two_records():
    raw_data = make_data()
    fitting_data = FittingData(raw_data)
    fitting_data.unselect_all_records()
    fitting_data.select_record(2)
    fitting_data.select_record(5)
    return fitting_data, raw_data, [2, 5]


@case(tags=[PARTIALLY_SELECTED])
def case_select_multiple_records():
    raw_data = make_data()
    fitting_data = FittingData(raw_data)
    fitting_data.unselect_all_records()
    fitting_data.select_record(2)
    fitting_data.select_record(5)
    fitting_data.select_record(3)
    fitting_data.select_record(10)
    return fitting_data, raw_data, [2, 3, 5, 10]


@case(tags=[ALL_SELECTED])
def case_reselect_record():
    raw_data = make_data()
    fitting_data = FittingData(raw_data)
    fitting_data.unselect_record(2)
    fitting_data.select_record(2)
    return fitting_data, raw_data, list(range(1, NUMBER_OF_RECORDS + 1))


@case(tags=[ALL_SELECTED])
def case_select_all_records():
    raw_data = make_data()
    fitting_data = FittingData(raw_data)
    fitting_data.unselect_all_records()
    fitting_data.select_record(2)
    fitting_data.select_record(5)
    fitting_data.select_all_records()
    return fitting_data, raw_data, list(range(1, NUMBER_OF_RECORDS + 1))


@case(tags=[PARTIALLY_SELECTED])
def case_x_domain_lower_bound():
    raw_data = make_data()
    fitting_data = FittingData(raw_data)
    xmin = 5
    selected_indices = [i for i, x in enumerate(fitting_data.x, start=1) if x >= xmin]
    fitting_data.select_by_x_domain(xmin=xmin)
    return fitting_data, raw_data, selected_indices


@case(tags=[PARTIALLY_SELECTED])
def case_x_domain_upper_bound():
    raw_data = make_data()
    fitting_data = FittingData(raw_data)
    xmax = 5
    selected_indices = [i for i, x in enumerate(fitting_data.x, start=1) if x <= xmax]
    fitting_data.select_by_x_domain(xmax=xmax)
    return fitting_data, raw_data, selected_indices


@case(tags=[PARTIALLY_SELECTED])
def case_x_domain_both_bounds():
    raw_data = make_data()
    fitting_data = FittingData(raw_data)
    xmin, xmax = 3, 6
    selected_indices = [
        i for i, x in enumerate(fitting_data.x, start=1) if xmin <= x <= xmax
    ]
    fitting_data.select_by_x_domain(xmin=xmin, xmax=xmax)
    return fitting_data, raw_data, selected_indices


@case(tags=[ALL_SELECTED])
def case_x_domain_select_all():
    raw_data = make_data()
    fitting_data = FittingData(raw_data)
    fitting_data.select_by_x_domain()
    return fitting_data, raw_data, list(range(1, NUMBER_OF_RECORDS + 1))


@case(tags=[PARTIALLY_SELECTED])
def case_x_domain_update_selected_false():
    raw_data = make_data()
    fitting_data = FittingData(raw_data)
    xmax = 5
    selected_indices = [i for i, x in enumerate(fitting_data.x, start=1) if x <= xmax]
    index = random.choice(selected_indices)
    fitting_data.unselect_record(index)
    fitting_data.select_by_x_domain(xmax=xmax)
    return fitting_data, raw_data, selected_indices


@case(tags=[PARTIALLY_SELECTED])
def case_x_domain_update_selected_true():
    raw_data = make_data()
    fitting_data = FittingData(raw_data)
    xmax = 5
    selected_indices = [i for i, x in enumerate(fitting_data.x, start=1) if x <= xmax]
    index = random.choice(selected_indices)
    selected_indices.remove(index)
    fitting_data.unselect_record(index)
    fitting_data.select_by_x_domain(xmax=xmax, update_selected=True)
    return fitting_data, raw_data, selected_indices


@case(tags=[PARTIALLY_SELECTED])
def case_y_domain_lower_bound():
    raw_data = make_data()
    fitting_data = FittingData(raw_data)
    ymin = 5
    selected_indices = [i for i, y in enumerate(fitting_data.y, start=1) if y >= ymin]
    fitting_data.select_by_y_domain(ymin=ymin)
    return fitting_data, raw_data, selected_indices


@case(tags=[PARTIALLY_SELECTED])
def case_y_domain_upper_bound():
    raw_data = make_data()
    fitting_data = FittingData(raw_data)
    ymax = 5
    selected_indices = [i for i, y in enumerate(fitting_data.y, start=1) if y <= ymax]
    fitting_data.select_by_y_domain(ymax=ymax)
    return fitting_data, raw_data, selected_indices


@case(tags=[PARTIALLY_SELECTED])
def case_y_domain_both_bounds():
    raw_data = make_data()
    fitting_data = FittingData(raw_data)
    ymin, ymax = 3, 6
    selected_indices = [
        i for i, y in enumerate(fitting_data.y, start=1) if ymin <= y <= ymax
    ]
    fitting_data.select_by_y_domain(ymin=ymin, ymax=ymax)
    return fitting_data, raw_data, selected_indices


@case(tags=[ALL_SELECTED])
def case_y_domain_select_all():
    raw_data = make_data()
    fitting_data = FittingData(raw_data)
    fitting_data.select_by_y_domain()
    return fitting_data, raw_data, list(range(1, NUMBER_OF_RECORDS + 1))


@case(tags=[PARTIALLY_SELECTED])
def case_y_domain_update_selected_false():
    raw_data = make_data()
    fitting_data = FittingData(raw_data)
    ymax = 5
    selected_indices = [i for i, y in enumerate(fitting_data.y, start=1) if y <= ymax]
    index = random.choice(selected_indices)
    fitting_data.unselect_record(index)
    fitting_data.select_by_y_domain(ymax=ymax)
    return fitting_data, raw_data, selected_indices


@case(tags=[PARTIALLY_SELECTED])
def case_y_domain_update_selected_true():
    raw_data = make_data()
    fitting_data = FittingData(raw_data)
    ymax = 5
    selected_indices = [i for i, y in enumerate(fitting_data.y, start=1) if y <= ymax]
    index = random.choice(selected_indices)
    selected_indices.remove(index)
    fitting_data.unselect_record(index)
    fitting_data.select_by_y_domain(ymax=ymax, update_selected=True)
    return fitting_data, raw_data, selected_indices


@case(tags=[PARTIALLY_SELECTED])
def case_xy_domain_half_bounded():
    raw_data = make_data()
    fitting_data = FittingData(raw_data)
    xmin, ymax = 3, 6
    selected_indices = [
        i
        for i, (x, y) in enumerate(zip(fitting_data.x, fitting_data.y), start=1)
        if xmin <= x and y <= ymax
    ]
    fitting_data.select_by_domains(xmin=xmin, ymax=ymax)
    return fitting_data, raw_data, selected_indices


@case(tags=[PARTIALLY_SELECTED])
def case_xy_domain_update_selected_false():
    raw_data = make_data()

    # Make sure enough cells pass filter
    raw_data["y"][3] = 5
    raw_data["y"][6] = 2

    fitting_data = FittingData(raw_data)
    xmin, ymax = 3, 6
    selected_indices = [
        i
        for i, (x, y) in enumerate(zip(fitting_data.x, fitting_data.y), start=1)
        if xmin <= x and y <= ymax
    ]
    index = random.choice(selected_indices)
    fitting_data.unselect_record(index)
    fitting_data.select_by_domains(xmin=xmin, ymax=ymax)
    return fitting_data, raw_data, selected_indices


@case(tags=[PARTIALLY_SELECTED])
def case_xy_domain_update_selected_true():
    raw_data = make_data()

    # Make sure enough cells pass filter
    raw_data["y"][3] = 5
    raw_data["y"][6] = 2

    fitting_data = FittingData(raw_data)
    xmin, ymax = 3, 6
    selected_indices = [
        i
        for i, (x, y) in enumerate(zip(fitting_data.x, fitting_data.y), start=1)
        if xmin <= x and y <= ymax
    ]
    index = random.choice(selected_indices)
    selected_indices.remove(index)
    fitting_data.unselect_record(index)
    fitting_data.select_by_domains(xmin=xmin, ymax=ymax, update_selected=True)
    return fitting_data, raw_data, selected_indices


@case(tags=[PARTIALLY_SELECTED])
def case_set_selected_records():
    raw_data = make_data()
    fitting_data = FittingData(raw_data)
    fitting_data.records_indices = [False, True, False, False, True] + [False] * (
        NUMBER_OF_RECORDS - 5
    )
    return fitting_data, raw_data, [2, 5]


@parametrize_with_cases(argnames="fitting_data, raw_data, selected_indices", cases=THIS_MODULE)
def test_x(fitting_data, raw_data, selected_indices):
    expected_x = extract_values(raw_data["x"], selected_indices)
    assert fitting_data.x.shape == np.shape(
        selected_indices
    ), "X shape is different than expected"
    assert fitting_data.x == pytest.approx(expected_x), "X is different than expected"


@parametrize_with_cases(argnames="fitting_data, raw_data, selected_indices", cases=THIS_MODULE)
def test_xerr(fitting_data, raw_data, selected_indices):
    assert fitting_data.xerr == pytest.approx(
        extract_values(raw_data["xerr"], selected_indices)
    ), "X error is different than expected"


@parametrize_with_cases(argnames="fitting_data, raw_data, selected_indices", cases=THIS_MODULE)
def test_y(fitting_data, raw_data, selected_indices):
    assert fitting_data.y == pytest.approx(
        extract_values(raw_data["y"], selected_indices)
    ), "Y is different than expected"


@parametrize_with_cases(argnames="fitting_data, raw_data, selected_indices", cases=THIS_MODULE)
def test_yerr(fitting_data, raw_data, selected_indices):
    assert fitting_data.yerr == pytest.approx(
        extract_values(raw_data["yerr"], selected_indices)
    ), "Y error is different than expected"


@parametrize_with_cases(argnames="fitting_data, raw_data, selected_indices", cases=THIS_MODULE)
def test_is_selected(fitting_data, raw_data, selected_indices):
    for i in range(1, fitting_data.number_of_records + 1):
        if i in selected_indices:
            assert fitting_data.is_selected(i), f"Record {i} was not selected"
        else:
            assert not fitting_data.is_selected(i), f"Record {i} was selected"


@parametrize_with_cases(argnames="fitting_data, raw_data, selected_indices", cases=THIS_MODULE)
def test_records(fitting_data, raw_data, selected_indices):
    selected_records = [
        fitting_data.record_data(i) for i in selected_indices
    ]
    for actual_record, expected_record in zip(selected_records, fitting_data.records):
        assert_list_equal(actual_record, expected_record, rel=EPSILON)


@parametrize_with_cases(
    argnames="fitting_data, raw_data, selected_indices", cases=THIS_MODULE, has_tag=ALL_SELECTED
)
def test_all_selected(fitting_data, raw_data, selected_indices):
    assert fitting_data.all_selected()
    assert not fitting_data.non_selected()


@parametrize_with_cases(
    argnames="fitting_data, raw_data, selected_indices",
    cases=THIS_MODULE,
    has_tag=PARTIALLY_SELECTED,
)
def test_partially_selected(fitting_data, raw_data, selected_indices):
    assert not fitting_data.all_selected()
    assert not fitting_data.non_selected()


@parametrize_with_cases(
    argnames="fitting_data, raw_data, selected_indices", cases=THIS_MODULE, has_tag=NON_SELECTED
)
def test_non_selected(fitting_data, raw_data, selected_indices):
    assert not fitting_data.all_selected()
    assert fitting_data.non_selected()


def test_set_selection_with_different_size():
    raw_data = make_data()
    fitting_data = FittingData(raw_data)

    def set_records_indices():
        fitting_data.records_indices = [False, False, True]

    with pytest.raises(
        FittingDataRecordsSelectionError,
        match=f"^Should select {NUMBER_OF_RECORDS} records, only 3 selected.$",
    ):
        set_records_indices()


def test_set_selection_with_non_boolean_values():
    raw_data = make_data()
    fitting_data = FittingData(raw_data)

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
    raw_data = make_data()
    fitting_data = FittingData(raw_data)

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
    raw_data = make_data()
    fitting_data = FittingData(raw_data)

    with pytest.raises(
        FittingDataRecordIndexError,
        match=(
            f"^Could not find record with index {index} in data. "
            f"Index should be between 1 and {NUMBER_OF_RECORDS}.$"
        ),
    ):
        fitting_data.unselect_record(index)
