import random

import pytest

from eddington.fitting_data import FittingData
from eddington.statistics import Statistics
from tests.fitting_data import COLUMNS, COLUMNS_NAMES, NUMBER_OF_RECORDS, STATISTICS
from tests.util import assert_calls, assert_statistics

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
    for header in COLUMNS_NAMES:
        assert_statistics(
            fitting_data.statistics(header), STATISTICS[header], rel=EPSILON
        )


def test_unselect_record_statistics():
    fitting_data = FittingData(COLUMNS)
    record_index = random.randint(0, NUMBER_OF_RECORDS)
    fitting_data.unselect_record(record_index)
    for header in COLUMNS_NAMES:
        header_statistics = fitting_data.statistics(header)
        assert header_statistics.mean != pytest.approx(STATISTICS[header].mean)
        assert_statistics(
            fitting_data.statistics(header),
            Statistics.from_array(fitting_data.column_data(header)),
            rel=EPSILON,
        )


def test_set_record_indices_statistics():
    fitting_data = FittingData(COLUMNS)
    record_indices = [False, True, False, False, True] + [False] * (
        NUMBER_OF_RECORDS - 5
    )
    fitting_data.records_indices = record_indices
    for header in COLUMNS_NAMES:
        header_statistics = fitting_data.statistics(header)
        assert header_statistics.mean != pytest.approx(STATISTICS[header].mean)
        assert_statistics(
            fitting_data.statistics(header),
            Statistics.from_array(fitting_data.column_data(header)),
            rel=EPSILON,
        )


def test_unselect_all_statistics():
    fitting_data = FittingData(COLUMNS)
    fitting_data.unselect_all_records()
    for header in COLUMNS_NAMES:
        assert fitting_data.statistics(header) is None


def test_reselect_all_statistics():
    fitting_data = FittingData(COLUMNS)
    fitting_data.unselect_all_records()
    fitting_data.select_all_records()
    for header in COLUMNS_NAMES:
        assert_statistics(
            fitting_data.statistics(header), STATISTICS[header], rel=EPSILON
        )


def test_statistics_to_excel_with_default_parameters(mock_save_as_excel):
    fitting_data = FittingData(COLUMNS)
    output_directory = "/path/to/directory"
    fitting_data.save_statistics_excel(output_directory=output_directory)

    assert_calls(
        mock_save_as_excel,
        [
            (
                [],
                dict(
                    content=STATISTICS_CONTENT,
                    output_directory=output_directory,
                    file_name="fitting_data_statistics",
                    sheet=None,
                ),
            )
        ],
        rel=EPSILON,
    )


def test_statistics_to_excel_with_parameters(mock_save_as_excel):
    fitting_data = FittingData(COLUMNS)
    output_directory = "/path/to/directory"
    file_name = "data"
    sheet = "sheet"
    fitting_data.save_statistics_excel(
        output_directory=output_directory, name=file_name, sheet=sheet
    )

    assert_calls(
        mock_save_as_excel,
        [
            (
                [],
                dict(
                    content=STATISTICS_CONTENT,
                    output_directory=output_directory,
                    file_name=file_name,
                    sheet=sheet,
                ),
            )
        ],
        rel=EPSILON,
    )


def test_statistics_to_csv_with_default_parameters(mock_save_as_csv):
    fitting_data = FittingData(COLUMNS)
    output_directory = "/path/to/directory"
    fitting_data.save_statistics_csv(output_directory=output_directory)

    assert_calls(
        mock_save_as_csv,
        [
            (
                [],
                dict(
                    content=STATISTICS_CONTENT,
                    output_directory=output_directory,
                    file_name="fitting_data_statistics",
                ),
            )
        ],
        rel=EPSILON,
    )


def test_statistics_to_csv_with_parameters(mock_save_as_csv):
    fitting_data = FittingData(COLUMNS)
    output_directory = "/path/to/directory"
    file_name = "data"
    fitting_data.save_statistics_csv(
        output_directory=output_directory,
        name=file_name,
    )

    assert_calls(
        mock_save_as_csv,
        [
            (
                [],
                dict(
                    content=STATISTICS_CONTENT,
                    output_directory=output_directory,
                    file_name=file_name,
                ),
            )
        ],
        rel=EPSILON,
    )
