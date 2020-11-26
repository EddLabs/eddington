import numpy as np
from pytest_cases import THIS_MODULE, parametrize_with_cases

from eddington.print_util import to_digit_string, to_relevant_precision_string


def case_positive_int_add_zeros():
    return 31, 4, "31.000"


def case_big_positive_int_reduce_digits():
    return 31415, 3, "3.142e+4"


def case_small_positive_int_reduce_digits():
    return 25.5033557, 4, "25.503"


def case_negative_int_add_zeros():
    return -31, 3, "-31.00"


def case_negative_int_reduce_digits():
    return -31415, 3, "-3.142e+4"


def case_one_adds_zeros():
    return 1, 3, "1.000"


def case_negative_one_add_zeros():
    return -1, 3, "-1.000"


def case_zero_add_zeros():
    return 0.0, 3, "0.000"


def case_minus_zero_add_zeros():
    return -0.0, 3, "0.000"


def case_float_bigger_than_one_reduce_digits():
    return np.pi, 2, "3.14"


def case_float_bigger_than_one_add_zeroes():
    return np.pi, 5, "3.14159"


def case_float_smaller_than_one_reduce_digits():
    return 0.712, 1, "0.71"


def case_float_smaller_than_one_add_zeroes():
    return 0.712, 4, "0.71200"


def case_small_float_reduce_digits():
    return 3.289e-5, 1, "3.3e-5"


def case_small_float_add_zeroes():
    return 3.289e-5, 4, "3.2890e-5"


def case_infinity():
    return np.inf, 3, "inf"


def case_negative_infinity():
    return -np.inf, 3, "-inf"


def case_nan():
    return np.nan, 3, "nan"


@parametrize_with_cases(argnames=["a", "n", "string"], cases=THIS_MODULE)
def test_precise_string(a, n, string):
    assert (
        to_relevant_precision_string(a, relevant_digits=n) == string
    ), "Relevant precision is different than expected"


def test_inf_to_digit_string():
    assert to_digit_string(np.inf, 5) == "inf"
