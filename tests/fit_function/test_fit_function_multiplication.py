from unittest import TestCase
from tests.fit_function.dummy_functions import dummy_func1, dummy_func2
from tests.fit_function.fit_function_base_test_case import FitFunctionBaseTestCase


class TestFitFunctionMultiplyByNumberFromRight(FitFunctionBaseTestCase, TestCase):
    func = dummy_func1 * 5.2
    n = 2
    values = [
        ([2, 1], 3, 57.2),
        ([-2.1, 4], 0.6, -3.432),
        ([4, -0.5], 3, -2.6),
        ([9, 2], 0, 46.8),
    ]


class TestFitFunctionMultiplyFitFunction(FitFunctionBaseTestCase, TestCase):
    func = dummy_func1 * dummy_func2
    n = 2
    values = [
        ([2, 1], 3, 66),
        ([-2.1, 4], 0.6, 0.8316),
        ([4, -0.5], 3, -6),
        ([9, 2], 0, 0),
    ]
