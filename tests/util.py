from typing import Any, Dict

import numpy as np
import pytest


def assert_dict_equal(dict1: Dict[Any, Any], dict2: Dict[Any, Any], rel):
    assert set(dict1.keys()) == set(dict2.keys()), "Keys are different than expected"
    for key, value in dict1.items():
        value2 = dict2[key]
        if isinstance(value, (list, np.ndarray)):
            assert_numpy_array_equal(value, value2, rel)
        elif isinstance(value, float):
            assert value == pytest.approx(
                value2, rel=rel
            ), f"{value} should be the same as {value2}"
        else:
            assert value == value2, f"{value} should be the same as {value2}"


def assert_numpy_array_equal(array1, array2, rel):
    if len(np.shape(array1)) <= 1:
        assert array1 == pytest.approx(array2, rel=rel)
        return
    assert np.shape(array1) == np.shape(array2)
    for i in range(np.shape(array1)[0]):
        assert array1[i] == pytest.approx(array2[i], rel=rel)
