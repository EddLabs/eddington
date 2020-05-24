from tests.fit_function.dummy_functions import dummy_func1, dummy_func2
from tests.fit_function.fit_function_base_test_case import FitFunctionMetaTestCase


class TestFitFunctionMultiplyByNumberFromRight(metaclass=FitFunctionMetaTestCase,):
    func = dummy_func1 * 5.2
    n = 2
    values = [
        ([2, 1], 3, 57.2),
        ([-2.1, 4], 0.6, -3.432),
        ([4, -0.5], 3, -2.6),
        ([9, 2], 0, 46.8),
    ]


class TestFitFunctionMultiplyByNumberFromLeft(metaclass=FitFunctionMetaTestCase,):
    func = 5.2 * dummy_func1
    n = 2
    values = [
        ([2, 1], 3, 57.2),
        ([-2.1, 4], 0.6, -3.432),
        ([4, -0.5], 3, -2.6),
        ([9, 2], 0, 46.8),
    ]


class TestFitFunctionMultiplyFitFunction(metaclass=FitFunctionMetaTestCase,):
    func = dummy_func1 * dummy_func2
    n = 2
    values = [
        ([2, 1], 3, 66),
        ([-2.1, 4], 0.6, 0.8316),
        ([4, -0.5], 3, -6),
        ([9, 2], 0, 0),
    ]


class TestFitFunctionDivideByNumberFromRight(metaclass=FitFunctionMetaTestCase,):
    func = dummy_func1 / 5.2
    n = 2
    values = [
        ([2, 1], 3, 2.11538),
        ([-2.1, 4], 0.6, -0.12692),
        ([4, -0.5], 3, -0.09615),
        ([9, 2], 0, 1.73077),
    ]


class TestFitFunctionDivideByNumberFromLeft(metaclass=FitFunctionMetaTestCase,):
    func = 5.2 / dummy_func1
    n = 2
    values = [
        ([2, 1], 3, 0.47273),
        ([-2.1, 4], 0.6, -7.87879),
        ([4, -0.5], 3, -10.4),
        ([9, 2], 0, 0.57778),
    ]


class TestFitFunctionDivideFitFunction(metaclass=FitFunctionMetaTestCase,):
    func = dummy_func2 / dummy_func1
    n = 2
    values = [
        ([2, 1], 3, 0.54545),
        ([-2.1, 4], 0.6, 1.90909),
        ([4, -0.5], 3, -24),
        ([9, 2], 0, 0),
    ]
