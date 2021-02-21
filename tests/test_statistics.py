import numpy as np
import pytest

from eddington.statistics import Statistics
from tests.util import assert_statistics

EPSILON = 1e-3


def random_statistics_map():
    keys = ["a", "b", "c"]
    return {
        key: Statistics.from_array(np.random.uniform(0, 100, size=3))
        for key in keys
    }


def assert_save_call(save_csv_call, statistics_map, **kwargs):
    assert save_csv_call.args == tuple()
    assert set(save_csv_call.kwargs.keys()) == {
        "content", *list(kwargs.keys())
    }
    for key, value in kwargs.items():
        assert save_csv_call.kwargs[key] == value
    assert_content(save_csv_call.kwargs["content"], statistics_map)


def assert_content(content, statistics_map):
    assert len(content) == 7
    assert content[0] == ["Parameters", *statistics_map.keys()]
    assert content[1] == ["Mean", *[stat.mean for stat in statistics_map.values()]]
    assert content[2] == ["Median", *[stat.median for stat in statistics_map.values()]]
    assert content[3] == [
        "Variance", *[stat.variance for stat in statistics_map.values()]
    ]
    assert content[4] == [
        "Standard Deviation",
        *[stat.standard_deviation for stat in statistics_map.values()],
    ]
    assert content[5] == [
        "Maximum Value", *[stat.maximum_value for stat in statistics_map.values()]
    ]
    assert content[6] == [
        "Minimum Value", *[stat.minimum_value for stat in statistics_map.values()]
    ]


def test_statistics_parameters():
    assert Statistics.parameters() == [
        "mean",
        "median",
        "variance",
        "standard_deviation",
        "maximum_value",
        "minimum_value",
    ]


def test_calculate_statistics_for_one_value():
    k = 6
    values = [k]
    stats = Statistics.from_array(values)
    assert_statistics(
        stats,
        Statistics(
            mean=k,
            median=k,
            variance=0,
            standard_deviation=0,
            maximum_value=k,
            minimum_value=k,
        ),
        rel=EPSILON,
    )


def test_calculate_statistics_for_two_values():
    a, b = 5, 8
    values = [a, b]
    stats = Statistics.from_array(values)
    assert_statistics(
        stats,
        Statistics(
            mean=(a + b) / 2,
            median=(a + b) / 2,
            variance=(b - a) ** 2 / 4,
            standard_deviation=(b - a) / 2,
            maximum_value=b,
            minimum_value=a,
        ),
        rel=EPSILON,
    )


def test_calculate_statistics_for_three_values():
    values = [5, 9, 8]
    stats = Statistics.from_array(values)
    assert_statistics(
        stats,
        Statistics(
            mean=7.3333,
            median=8,
            variance=2.888,
            standard_deviation=1.699,
            maximum_value=9,
            minimum_value=5,
        ),
        rel=EPSILON,
    )


def test_save_statistics_map_as_csv_with_default_name(mock_save_as_csv):
    statistics_map = random_statistics_map()
    output_dir = "/path/to/output/dir"
    Statistics.save_as_csv(statistics_map=statistics_map, output_directory=output_dir)
    assert mock_save_as_csv.call_count == 1
    save_csv_call = mock_save_as_csv.call_args_list[0]
    assert_save_call(
        save_csv_call,
        output_directory=output_dir,
        statistics_map=statistics_map,
        file_name="fitting_data_statistics",
    )


def test_save_statistics_map_as_csv_with_custom_name(mock_save_as_csv):
    statistics_map = random_statistics_map()
    output_dir = "/path/to/output/dir"
    name = "my_statistics"
    Statistics.save_as_csv(
        statistics_map=statistics_map, output_directory=output_dir, name=name
    )
    assert mock_save_as_csv.call_count == 1
    save_csv_call = mock_save_as_csv.call_args_list[0]
    assert_save_call(
        save_csv_call,
        output_directory=output_dir,
        statistics_map=statistics_map,
        file_name=name,
    )


def test_save_statistics_map_as_excel_with_default_name(mock_save_as_excel):
    statistics_map = random_statistics_map()
    output_dir = "/path/to/output/dir"
    Statistics.save_as_excel(statistics_map=statistics_map, output_directory=output_dir)
    assert mock_save_as_excel.call_count == 1
    save_excel_call = mock_save_as_excel.call_args_list[0]
    assert_save_call(
        save_excel_call,
        output_directory=output_dir,
        statistics_map=statistics_map,
        sheet=None,
        file_name="fitting_data_statistics",
    )


def test_save_statistics_map_as_excel_with_custom_name(mock_save_as_excel):
    statistics_map = random_statistics_map()
    output_dir = "/path/to/output/dir"
    name = "my_statistics"
    Statistics.save_as_excel(
        statistics_map=statistics_map, output_directory=output_dir, name=name
    )
    assert mock_save_as_excel.call_count == 1
    save_csv_call = mock_save_as_excel.call_args_list[0]
    assert_save_call(
        save_csv_call,
        output_directory=output_dir,
        statistics_map=statistics_map,
        sheet=None,
        file_name=name,
    )


def test_save_statistics_map_as_excel_with_custom_sheet(mock_save_as_excel):
    statistics_map = random_statistics_map()
    output_dir = "/path/to/output/dir"
    sheet = "my_sheet"
    Statistics.save_as_excel(
        statistics_map=statistics_map, output_directory=output_dir, sheet=sheet
    )
    assert mock_save_as_excel.call_count == 1
    save_csv_call = mock_save_as_excel.call_args_list[0]
    assert_save_call(
        save_csv_call,
        output_directory=output_dir,
        statistics_map=statistics_map,
        sheet=sheet,
        file_name="fitting_data_statistics",
    )


def test_calculate_statistics_raises_error_for_no_values():
    with pytest.raises(ValueError, match="^Cannot calculate statistics of no values.$"):
        Statistics.from_array([])
