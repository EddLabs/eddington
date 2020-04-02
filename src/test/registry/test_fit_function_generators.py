from unittest import TestCase
import numpy as np

from eddington_core.exceptions import InvalidGeneratorInitialization
from test.registry.fitting_test_case import (
    linear_case,
    add_test_case,
    parabolic_case,
    polynom_3_case,
    hyperbolic_case,
    inverse_power_2_case,
    straight_power_2_case,
    straight_power_3_case,
)
from eddington_core.fit_functions.fit_function import fit_function
from eddington_core.fit_functions.fit_function_generator import fit_function_generator
from eddington_core.fit_functions.fit_function_generators_list import (
    polynom,
    straight_power,
    inverse_power,
)
from eddington_core.fit_functions.fit_functions_registry import FitFunctionsRegistry


class FitFunctionGeneratorsBaseTestCase:
    eps = 1e-5
    decimal = 5

    def assert_raises_unfit_parameters(self, n0, func):
        self.assertRaisesRegex(
            InvalidGeneratorInitialization,
            f"^input length should be {self.n}, got {n0}$",
            func,
            np.random.random(n0),
            np.random.random(),
        )

    def test_generator_name(self):
        self.assertEqual(
            self.name,
            self.generator.name,
            msg="Generator name is different than expected",
        )

    def test_generator_signature(self):
        self.assertEqual(
            self.signature,
            self.generator.signature,
            msg="Generator signature is different than expected",
        )

    def test_generator_syntax(self):
        self.assertEqual(
            self.syntax,
            self.generator.syntax,
            msg="Generator syntax is different than expected",
        )

    def test_generator_parameters(self):
        self.assertEqual(
            self.parameters,
            self.generator.parameters,
            msg="Generator parameters are different than expected",
        )

    def test_is_generator(self):
        self.assertTrue(
            self.generator.is_generator(),
            msg="Fit function generator should be a generator",
        )


class TestDummyFitFunctionGenerator(TestCase, FitFunctionGeneratorsBaseTestCase):
    def setUp(self):
        self.name = "dummy_generator"
        self.parameters = ["a", "b"]
        self.syntax = "b * a[0] + c * a[1] * x"

        @fit_function_generator(parameters=self.parameters, syntax=self.syntax)
        def dummy_generator(b, c):
            @fit_function(n=2, syntax="b * a[0] + c * a[1] * x", save=False)
            def dummy_func(a, x):
                return b * a[0] + c * a[1] * x

            return dummy_func

        self.generator = dummy_generator
        self.signature = "dummy_generator(a, b)"

    def tearDown(self):
        FitFunctionsRegistry.remove(self.name)


class TestPolynomFitFunctionGenerator(TestCase, FitFunctionGeneratorsBaseTestCase):

    generator = polynom
    name = "polynom"
    signature = "polynom(n)"
    syntax = "a[0] + a[1] * x + ... + a[n] * x ^ n"
    parameters = "n"
    cases = {1: linear_case, 2: parabolic_case, 3: polynom_3_case}

    def test_polynom_with_zero_degree_raises_error(self):
        self.assertRaisesRegex(
            InvalidGeneratorInitialization, r"^n must be positive, got 0$", polynom, 0
        )

    def test_polynom_with_negative_degree_raises_error(self):
        self.assertRaisesRegex(
            InvalidGeneratorInitialization, "^n must be positive, got -1$", polynom, -1
        )


class TestStraightPowerFitFunctionGenerator(
    TestCase, FitFunctionGeneratorsBaseTestCase
):

    generator = straight_power
    name = "straight_power"
    signature = "straight_power(n)"
    syntax = "a[0] * (x + a[1]) ^ n + a[2]"
    parameters = "n"
    cases = {2: straight_power_2_case, 3: straight_power_3_case}

    def test_straight_power_with_zero_degree_raises_error(self):
        self.assertRaisesRegex(
            InvalidGeneratorInitialization,
            "^n must be positive, got 0$",
            straight_power,
            0,
        )

    def test_straight_power_with_negative_degree_raises_error(self):
        self.assertRaisesRegex(
            InvalidGeneratorInitialization,
            "^n must be positive, got -1$",
            straight_power,
            -1,
        )

    def test_straight_power_with_one_degree_raises_error(self):
        self.assertRaisesRegex(
            InvalidGeneratorInitialization,
            '^n cannot be 1. use "linear" fit instead.$',
            straight_power,
            1,
        )


class TestInversePowerFitFunctionGenerator(TestCase, FitFunctionGeneratorsBaseTestCase):

    generator = inverse_power
    name = "inverse_power"
    signature = "inverse_power(n)"
    syntax = "a[0] / (x + a[1]) ^ n + a[2]"
    parameters = "n"
    cases = {1: hyperbolic_case, 2: inverse_power_2_case}

    def test_inverse_power_with_zero_degree_raises_error(self):
        self.assertRaisesRegex(
            InvalidGeneratorInitialization,
            "^n must be positive, got 0$",
            inverse_power,
            0,
        )

    def test_inverse_power_with_negative_degree_raises_error(self):
        self.assertRaisesRegex(
            InvalidGeneratorInitialization,
            "^n must be positive, got -1$",
            inverse_power,
            -1,
        )


def init_fit_cases(cls):
    for parameter, case in cls.cases.items():
        add_test_case(
            cls=cls,
            func=cls.generator(parameter),
            case=case,
            name=f"{cls.name}_{parameter}",
        )
        add_test_case(
            cls=cls,
            func=FitFunctionsRegistry.load(cls.name, parameter),
            case=case,
            name=f"loaded_{cls.name}_{parameter}",
        )


init_fit_cases(TestPolynomFitFunctionGenerator)
init_fit_cases(TestStraightPowerFitFunctionGenerator)
init_fit_cases(TestInversePowerFitFunctionGenerator)
