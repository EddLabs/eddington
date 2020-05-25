from unittest import TestCase

import numpy as np
from eddington_core import FitFunctionRuntimeError

from tests.fit_function.dummy_functions import dummy_func1


class TestFitFunction(TestCase):
    decimal = 5

    def test_is_generator(self):
        self.assertFalse(
            dummy_func1.is_generator(), msg="Fit function is not a generator"
        )

    def test_name(self):
        self.assertEqual(
            "dummy_func1", dummy_func1.name, msg="Name is different than expected",
        )

    def test_signature(self):
        self.assertEqual(
            "dummy_func1",
            dummy_func1.signature,
            msg="Signature is different than expected",
        )

    def test_title_name(self):
        self.assertEqual(
            "Dummy Func1",
            dummy_func1.title_name,
            msg="Title name is different than expected",
        )

    def test_call_success(self):
        a = np.array([1, 2])
        x = 3
        result = dummy_func1(a, x)
        self.assertAlmostEqual(
            19,
            result,
            places=self.decimal,
            msg="Execution result is different than expected",
        )

    def test_call_failure_because_of_not_enough_parameters(self):
        a = np.array([1])
        x = 3
        self.assertRaisesRegex(
            FitFunctionRuntimeError,
            "^input length should be 2, got 1$",
            dummy_func1,
            a,
            x,
        )

    def test_call_failure_because_of_too_many_parameters(self):
        a = np.array([1, 2, 3])
        x = 4
        self.assertRaisesRegex(
            FitFunctionRuntimeError,
            "^input length should be 2, got 3$",
            dummy_func1,
            a,
            x,
        )

    def test_assign(self):
        a = np.array([1, 2])
        x = 3
        assign_func = dummy_func1.assign(a)
        result = assign_func(x)
        self.assertAlmostEqual(
            19,
            result,
            places=self.decimal,
            msg="Execution result is different than expected",
        )

    def test_assign_failure_because_of_not_enough_parameters(self):
        a = np.array([1])
        self.assertRaisesRegex(
            FitFunctionRuntimeError,
            "^input length should be 2, got 1$",
            dummy_func1.assign,
            a,
        )

    def test_assign_failure_because_of_too_many_parameters(self):
        a = np.array([1, 2, 3])
        self.assertRaisesRegex(
            FitFunctionRuntimeError,
            "^input length should be 2, got 3$",
            dummy_func1.assign,
            a,
        )

    def test_fit_function_representation(self):
        self.assertEqual(
            "FitFunction(name='dummy_func1', syntax='a[0] + a[1] * x ** 2')",
            str(dummy_func1),
            msg="Representation is different than expected",
        )
