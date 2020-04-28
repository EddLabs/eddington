from unittest import TestCase
from tests.fit_function.dummy_functions import dummy_func1, dummy_func2
from tests.fit_function.fit_function_base_test_case import FitFunctionBaseTestCase


class TestDummyFitFunction1(FitFunctionBaseTestCase, TestCase):
    func = dummy_func1
    n = 2
    values = [
        ([2, 1], 3, 11),
        ([-2.1, 4], 0.6, -0.66),
        ([4, -0.5], 3, -0.5),
        ([9, 2], 0, 9),
    ]


class TestDummyFitFunction2(FitFunctionBaseTestCase, TestCase):
    func = dummy_func2
    n = 1
    values = [
        ([2], 3, 6),
        ([-2.1], 0.6, -1.26),
        ([4], 3, 12),
        ([9], 0, 0),
    ]


class TestFitFunctionNegative(FitFunctionBaseTestCase, TestCase):
    func = -dummy_func1
    n = 2
    values = [
        ([2, 1], 3, -11),
        ([-2.1, 4], 0.6, 0.66),
        ([4, -0.5], 3, 0.5),
        ([9, 2], 0, -9),
    ]


class TestFitFunctionAddNumberFromRight(FitFunctionBaseTestCase, TestCase):
    func = dummy_func1 + 5.2
    n = 2
    values = [
        ([2, 1], 3, 16.2),
        ([-2.1, 4], 0.6, 4.54),
        ([4, -0.5], 3, 4.7),
        ([9, 2], 0, 14.2),
    ]


class TestFitFunctionAddNumberFromLeft(FitFunctionBaseTestCase, TestCase):
    func = 5.2 + dummy_func1
    n = 2
    values = [
        ([2, 1], 3, 16.2),
        ([-2.1, 4], 0.6, 4.54),
        ([4, -0.5], 3, 4.7),
        ([9, 2], 0, 14.2),
    ]


class TestFitFunctionAddFitFunction(FitFunctionBaseTestCase, TestCase):
    func = dummy_func1 + dummy_func2
    n = 2
    values = [
        ([2, 1], 3, 17),
        ([-2.1, 4], 0.6, -1.92),
        ([4, -0.5], 3, 11.5),
        ([9, 2], 0, 9),
    ]


class TestDummyFitSubtractNumberFromRight(FitFunctionBaseTestCase, TestCase):
    func = dummy_func1 - 5.2
    n = 2
    values = [
        ([2, 1], 3, 5.8),
        ([-2.1, 4], 0.6, -5.86),
        ([4, -0.5], 3, -5.7),
        ([9, 2], 0, 3.8),
    ]


class TestDummyFitSubtractNumberFromLeft(FitFunctionBaseTestCase, TestCase):
    func = 5.2 - dummy_func1
    n = 2
    values = [
        ([2, 1], 3, -5.8),
        ([-2.1, 4], 0.6, 5.86),
        ([4, -0.5], 3, 5.7),
        ([9, 2], 0, -3.8),
    ]


class TestFitFunctionSubtractFitFunction(FitFunctionBaseTestCase, TestCase):
    func = dummy_func1 - dummy_func2
    n = 2
    values = [
        ([2, 1], 3, 5),
        ([-2.1, 4], 0.6, 0.6),
        ([4, -0.5], 3, -12.5),
        ([9, 2], 0, 9),
    ]
