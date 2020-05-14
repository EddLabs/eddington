from unittest import TestCase

from tests.fit_function.dummy_functions import dummy_func1, dummy_func2
from tests.fit_function.fit_function_base_test_case import FitFunctionBaseTestCase


class TestFitFunctionPower1(FitFunctionBaseTestCase, TestCase):
    func = dummy_func1 ** 1
    n = 2
    values = [
        ([2, 1], 3, 11),
        ([-2.1, 4], 0.6, -0.66),
        ([4, -0.5], 3, -0.5),
        ([9, 2], 0, 9),
    ]


class TestFitFunctionPower2(FitFunctionBaseTestCase, TestCase):
    func = dummy_func1 ** 2
    n = 2
    values = [
        ([2, 1], 3, 121),
        ([-2.1, 4], 0.6, 0.4356),
        ([4, -0.5], 3, 0.25),
        ([9, 2], 0, 81),
    ]


class TestFitFunctionPowerHalf(FitFunctionBaseTestCase, TestCase):
    func = dummy_func1 ** 0.5
    n = 2
    values = [
        ([2, 1], 3, 3.31662),
        ([-1.1, 4], 0.6, 0.5831),
        ([5, -0.5], 3, 0.70711),
        ([9, 2], 0, 3),
    ]


class TestFitFunctionPowerAnotherFitFunction(FitFunctionBaseTestCase, TestCase):
    func = dummy_func1 ** dummy_func2
    n = 2
    values = [
        ([2, 1], 3, 1771561),
        ([-1.1, 4], 0.6, 2.03809),
        ([5, -0.5], 3, 3.05175e-5),
        ([9, 2], 0, 1),
    ]
