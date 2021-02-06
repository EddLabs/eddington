import random
from numbers import Number
from typing import Any, Dict, List, Tuple
from unittest.mock import Mock

import numpy as np
import pytest

from eddington import fitting_function

# Assertions
from eddington.statistics import Statistics


def assert_calls(mock_object: Mock, calls: List[Tuple[List[Any], Dict[str, Any]]], rel):
    expected_number_of_calls = len(calls)
    assert mock_object.call_count == expected_number_of_calls, (
        f"Lists should have the same length. Expected {expected_number_of_calls}, "
        f"but got {mock_object.call_count}"
    )
    for i, (args, kwargs) in enumerate(calls):
        assert_list_equal(mock_object.call_args_list[i][0], args, rel)
        assert_dict_equal(mock_object.call_args_list[i][1], kwargs, rel)


def assert_list_equal(list1: List[Any], list2: List[Any], rel):
    length1, length2 = len(list1), len(list2)
    assert (
        length1 == length2
    ), f"Lists should have the same length. {length1} != {length2}"
    for item1, item2 in zip(list1, list2):
        assert_equal_item(item1, item2, rel)


def assert_dict_equal(dict1: Dict[Any, Any], dict2: Dict[Any, Any], rel):
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    assert keys1 == keys2, f"Keys are different than expected. {keys1} != {keys2}"
    for key, value in dict1.items():
        value2 = dict2[key]
        assert_equal_item(value, value2, rel)


def assert_equal_item(value1, value2, rel):
    if isinstance(value1, (list, np.ndarray)):
        if all(isinstance(item, Number) for item in value1):
            assert_numpy_array_equal(value1, value2, rel)
        else:
            assert_list_equal(value1, value2, rel)
    elif isinstance(value1, float):
        assert value1 == pytest.approx(
            value2, rel=rel
        ), f"{value1} should be the same as {value2}"
    elif isinstance(value1, dict):
        assert_dict_equal(value1, value2, rel=rel)
    else:
        assert value1 == value2, f"{value1} should be the same as {value2}"


def assert_numpy_array_equal(array1, array2, rel):
    if len(np.shape(array1)) <= 1:
        assert array1 == pytest.approx(array2, rel=rel)
        return
    assert np.shape(array1) == np.shape(array2)
    for i in range(np.shape(array1)[0]):
        assert array1[i] == pytest.approx(array2[i], rel=rel)


def assert_statistics(
    actual_stats: Statistics,
    expected_stats: Statistics,
    rel: float,
):
    assert actual_stats.mean == pytest.approx(
        expected_stats.mean, rel=rel
    ), "Mean value is different than expected"
    assert actual_stats.median == pytest.approx(
        expected_stats.median, rel=rel
    ), "Median value is different than expected"
    assert actual_stats.variance == pytest.approx(
        expected_stats.variance, rel=rel
    ), "Variance value is different than expected"
    assert actual_stats.standard_deviation == pytest.approx(
        expected_stats.standard_deviation, rel=rel
    ), "Standard deviation value is different than expected"
    assert actual_stats.maximum_value == pytest.approx(
        expected_stats.maximum_value, rel=rel
    ), "Maximum value is different than expected"
    assert actual_stats.minimum_value == pytest.approx(
        expected_stats.minimum_value, rel=rel
    ), "Minimum value is different than expected"


# Additional methods


def dummy_function(name, syntax, save=True):
    value = random.random()

    @fitting_function(n=2, name=name, syntax=syntax, save=save)
    def dummy_func(a, x):  # pylint: disable=W0613
        return value

    return dummy_func
