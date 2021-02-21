import random

import pytest

from eddington.exceptions import FittingDataColumnExistenceError
from eddington.fitting_data import FittingData
from eddington.statistics import Statistics
from tests.fitting_data import COLUMNS, COLUMNS_NAMES, NUMBER_OF_RECORDS, STATISTICS
from tests.util import assert_statistics

EPSILON = 1e-7
STATISTICS_CONTENT = [
    ["Parameters", *COLUMNS_NAMES],
    ["Mean", *[statistics.mean for statistics in STATISTICS.values()]],
    ["Median", *[statistics.median for statistics in STATISTICS.values()]],
    ["Variance", *[statistics.variance for statistics in STATISTICS.values()]],
    [
        "Standard Deviation",
        *[statistics.standard_deviation for statistics in STATISTICS.values()],
    ],
    [
        "Maximum Value",
        *[statistics.maximum_value for statistics in STATISTICS.values()],
    ],
    [
        "Minimum Value",
        *[statistics.minimum_value for statistics in STATISTICS.values()],
    ],
]


def test_initial_statistics():
    fitting_data = FittingData(COLUMNS)
    assert set(fitting_data.statistics_map.keys()) == set(COLUMNS_NAMES)
    for header in COLUMNS_NAMES:
        assert_statistics(
            fitting_data.statistics(header), STATISTICS[header], rel=EPSILON
        )
        assert_statistics(
            fitting_data.statistics_map[header], STATISTICS[header], rel=EPSILON
        )


def test_unselect_record_statistics():
    fitting_data = FittingData(COLUMNS)
    record_index = random.randint(1, NUMBER_OF_RECORDS)
    fitting_data.unselect_record(record_index)
    assert set(fitting_data.statistics_map.keys()) == set(COLUMNS_NAMES)
    for header in COLUMNS_NAMES:
        header_statistics = fitting_data.statistics(header)
        assert header_statistics.mean != pytest.approx(STATISTICS[header].mean)
        assert_statistics(
            fitting_data.statistics(header),
            Statistics.from_array(fitting_data.column_data(header)),
            rel=EPSILON,
        )
        assert_statistics(
            fitting_data.statistics_map[header],
            Statistics.from_array(fitting_data.column_data(header)),
            rel=EPSILON,
        )


def test_set_record_indices_statistics():
    fitting_data = FittingData(COLUMNS)
    record_indices = [False, True, False, False, True] + [False] * (
        NUMBER_OF_RECORDS - 5
    )
    fitting_data.records_indices = record_indices
    assert set(fitting_data.statistics_map.keys()) == set(COLUMNS_NAMES)
    for header in COLUMNS_NAMES:
        header_statistics = fitting_data.statistics(header)
        assert header_statistics.mean != pytest.approx(STATISTICS[header].mean)
        assert_statistics(
            fitting_data.statistics(header),
            Statistics.from_array(fitting_data.column_data(header)),
            rel=EPSILON,
        )
        assert_statistics(
            fitting_data.statistics_map[header],
            Statistics.from_array(fitting_data.column_data(header)),
            rel=EPSILON,
        )


def test_unselect_all_statistics():
    fitting_data = FittingData(COLUMNS)
    fitting_data.unselect_all_records()
    assert len(fitting_data.statistics_map.keys()) == 0
    for header in COLUMNS_NAMES:
        assert fitting_data.statistics(header) is None


def test_reselect_all_statistics():
    fitting_data = FittingData(COLUMNS)
    fitting_data.unselect_all_records()
    fitting_data.select_all_records()
    assert set(fitting_data.statistics_map.keys()) == set(COLUMNS_NAMES)
    for header in COLUMNS_NAMES:
        assert_statistics(
            fitting_data.statistics(header), STATISTICS[header], rel=EPSILON
        )
        assert_statistics(
            fitting_data.statistics_map[header], STATISTICS[header], rel=EPSILON
        )


def test_get_statistics_with_unknown_header():
    fitting_data = FittingData(COLUMNS)

    with pytest.raises(
        FittingDataColumnExistenceError,
        match='^Could not find column "I do not exist" in data$',
    ):
        fitting_data.statistics(column_name="I do not exist")
