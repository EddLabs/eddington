from unittest import TestCase

from eddington.fit_functions.fit_functions_list import (
    constant,
    linear,
    parabolic,
    hyperbolic,
    exponential,
)
from eddington.fit_functions.fit_functions_registry import FitFunctionsRegistry
from test.registry.fitting_test_case import (
    add_test_case,
    linear_case,
    parabolic_case,
    hyperbolic_case,
    exponential_case,
    constant_case,
)


class TestConstantFitFunction(TestCase):
    func = constant
    case = constant_case


class TestLinearFitFunction(TestCase):
    func = linear
    case = linear_case


class TestParabolicFitFunction(TestCase):
    func = parabolic
    case = parabolic_case


class TestHyperbolicFitFunction(TestCase):
    func = hyperbolic
    case = hyperbolic_case


class TestExponentialFitFunction(TestCase):
    func = exponential
    case = exponential_case


def init_fit_cases(cls):
    add_test_case(cls=cls, func=cls.func, case=cls.case)
    add_test_case(
        cls=cls,
        func=FitFunctionsRegistry.load(cls.case.name),
        case=cls.case,
        name=f"loaded_{cls.case.name}",
    )


init_fit_cases(TestConstantFitFunction)
init_fit_cases(TestLinearFitFunction)
init_fit_cases(TestParabolicFitFunction)
init_fit_cases(TestHyperbolicFitFunction)
init_fit_cases(TestExponentialFitFunction)
