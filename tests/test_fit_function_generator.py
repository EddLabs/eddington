from unittest import TestCase

import numpy as np

from tests.dummy_functions import (
    dummy_generator_with_1_parameter,
    dummy_generator_with_2_parameters,
)


class TestFitFunctionGenerator(TestCase):
    decimal = 5

    def test_call_generator(self):
        func = dummy_generator_with_2_parameters(1, 2)
        a = np.array([3, 4])
        x = 5
        result = func(a, x)
        self.assertAlmostEqual(
            43,
            result,
            places=self.decimal,
            msg="Fit function result is different than expected",
        )

    def test_name(self):
        self.assertEqual(
            "generator_with_2_parameters",
            dummy_generator_with_2_parameters.name,
            msg="Generator name is different than expected",
        )

    def test_signature_with_2_parameters(self):
        self.assertEqual(
            "generator_with_2_parameters(p0, p1)",
            dummy_generator_with_2_parameters.signature,
            msg="Generator signature is different than expected",
        )

    def test_signature_with_1_parameter(self):
        self.assertEqual(
            "generator_with_1_parameter(p1)",
            dummy_generator_with_1_parameter.signature,
            msg="Generator signature is different than expected",
        )

    def test_syntax(self):
        self.assertEqual(
            "Some syntax",
            dummy_generator_with_2_parameters.syntax,
            msg="Generator syntax is different than expected",
        )

    def test_parameters_list(self):
        self.assertEqual(
            ["p0", "p1"],
            dummy_generator_with_2_parameters.parameters,
            msg="Generator parameters are different than expected",
        )

    def test_parameters_string(self):
        self.assertEqual(
            "p1",
            dummy_generator_with_1_parameter.parameters,
            msg="Generator parameters are different than expected",
        )

    def test_is_generator(self):
        self.assertTrue(
            dummy_generator_with_1_parameter.is_generator(),
            msg="Fit function generator is a generator",
        )
