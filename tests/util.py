from typing import Any, Dict, List

import numpy as np
import pytest


def assert_list_equal(list1: List[Any], list2: List[Any], rel):
    assert len(list1) == len(list2), "Lists should have the same length"
    for item1, item2 in zip(list1, list2):
        assert_equal_item(item1, item2, rel)


def assert_dict_equal(dict1: Dict[Any, Any], dict2: Dict[Any, Any], rel):
    assert set(dict1.keys()) == set(dict2.keys()), "Keys are different than expected"
    for key, value in dict1.items():
        value2 = dict2[key]
        assert_equal_item(value, value2, rel)


def assert_equal_item(value1, value2, rel):
    if isinstance(value1, (list, np.ndarray)):
        assert_numpy_array_equal(value1, value2, rel)
    elif isinstance(value1, float):
        assert value1 == pytest.approx(
            value2, rel=rel
        ), f"{value1} should be the same as {value2}"
    else:
        assert value1 == value2, f"{value1} should be the same as {value2}"


def assert_numpy_array_equal(array1, array2, rel):
    if len(np.shape(array1)) <= 1:
        assert array1 == pytest.approx(array2, rel=rel)
        return
    assert np.shape(array1) == np.shape(array2)
    for i in range(np.shape(array1)[0]):
        assert array1[i] == pytest.approx(array2[i], rel=rel)
