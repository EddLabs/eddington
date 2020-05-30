from unittest import TestCase

import numpy as np
from eddington_core import FitFunctionRuntimeError

from tests.fit_function.dummy_functions import dummy_func1, dummy_func2


class TestFitFunction(TestCase):
    decimal = 5

    def setUp(self):
        dummy_func1.clear_fixed()


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
            "^Input length should be 2, got 1$",
            dummy_func1,
            a,
            x,
        )

    def test_call_failure_because_of_too_many_parameters(self):
        a = np.array([1, 2, 3])
        x = 4
        self.assertRaisesRegex(
            FitFunctionRuntimeError,
            "^Input length should be 2, got 3$",
            dummy_func1,
            a,
            x,
        )

    def test_assign(self):
        a = np.array([1, 2])
        x = 3
        dummy_func1.assign(a)
        result = dummy_func1(x)
        self.assertAlmostEqual(
            19,
            result,
            places=self.decimal,
            msg="Execution result is different than expected",
        )

    def test_call_func_with_fix_value(self):
        a = np.array([7, 2, 1])
        x = 2
        dummy_func2.fix(1, 3)
        result = dummy_func2(a, x)
        self.assertAlmostEqual(
            29,
            result,
            places=self.decimal,
            msg="Execution result is different than expected",
        )

    def test_call_x_derivative_with_fix_value(self):
        a = np.array([7, 2, 1])
        x = 2
        dummy_func2.fix(1, 3)
        result = dummy_func2.x_derivative(a, x)
        self.assertAlmostEqual(
            23,
            result,
            places=self.decimal,
            msg="x derivative execution result is different than expected",
        )

    def test_call_a_derivative_with_fix_value(self):
        a = np.array([7, 2, 1])
        x = 2
        dummy_func2.fix(1, 3)
        result = dummy_func2.a_derivative(a, x)
        np.testing.assert_almost_equal(
            result,
            np.array([1, 4, 8]),
            decimal=self.decimal,
            err_msg="a derivative execution result is different than expected",
        )

    def test_override_fix_value(self):
        a = np.array([7, 2, 1])
        x = 2
        dummy_func2.fix(1, 9)
        dummy_func2.fix(1, 3)
        result = dummy_func2(a, x)
        self.assertAlmostEqual(
            29,
            result,
            places=self.decimal,
            msg="Execution result is different than expected",
        )

    def test_unfix_value(self):
        dummy_func2.fix(1, 5)
        dummy_func2.unfix(1)
        a = np.array([7, 3, 2, 1])
        x = 2
        result = dummy_func2(a, x)
        self.assertAlmostEqual(
            29,
            result,
            places=self.decimal,
            msg="Execution result is different than expected",
        )

    def test_clear_fixed(self):
        dummy_func2.fix(1, 5)
        dummy_func2.fix(3, 2)
        dummy_func2.clear_fixed()
        a = np.array([7, 3, 2, 1])
        x = 2
        result = dummy_func2(a, x)
        self.assertAlmostEqual(
            29,
            result,
            places=self.decimal,
            msg="Execution result is different than expected",
        )

    def test_assign_failure_because_of_not_enough_parameters(self):
        a = np.array([1])
        self.assertRaisesRegex(
            FitFunctionRuntimeError,
            "^Input length should be 2, got 1$",
            dummy_func1.assign,
            a,
        )

    def test_assign_failure_because_of_too_many_parameters(self):
        a = np.array([1, 2, 3])
        self.assertRaisesRegex(
            FitFunctionRuntimeError,
            "^Input length should be 2, got 3$",
            dummy_func1.assign,
            a,
        )

    def test_fix_failure_when_trying_to_fix_negative_index(self):
        self.assertRaisesRegex(
            FitFunctionRuntimeError,
            "^Cannot fix index -1. Indices should be between 0 and 1$",
            dummy_func1.fix,
            -1,
            10,
        )

    def test_fix_failure_when_trying_to_fix_too_big_index(self):
        self.assertRaisesRegex(
            FitFunctionRuntimeError,
            "^Cannot fix index 2. Indices should be between 0 and 1$",
            dummy_func1.fix,
            2,
            10,
        )

    def test_call_failure_because_not_enough_parameters_after_fix(self):
        a = np.array([7, 2])
        x = 2
        dummy_func2.fix(1, 3)

        self.assertRaisesRegex(
            FitFunctionRuntimeError,
            "^Input length should be 4, got 3$",
            dummy_func2,
            a,
            x,
        )

    def test_call_failure_because_too_much_parameters_after_fix(self):
        a = np.array([7, 2, 8, 2])
        x = 2
        dummy_func2.fix(1, 3)

        self.assertRaisesRegex(
            FitFunctionRuntimeError,
            "^Input length should be 4, got 5$",
            dummy_func2,
            a,
            x,
        )

    def test_call_failure_when_trying_to_run_without_arguments(self):
        self.assertRaisesRegex(
            FitFunctionRuntimeError,
            '^No parameters has been given to "dummy_func2"$',
            dummy_func2,
        )

    def test_fit_function_representation(self):
        self.assertEqual(
            "FitFunction(name='dummy_func1', syntax='a[0] + a[1] * x ** 2')",
            str(dummy_func1),
            msg="Representation is different than expected",
        )
