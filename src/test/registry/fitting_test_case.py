from dataclasses import dataclass
import numpy as np
from typing import List, Union

from eddington_core import FitFunction


@dataclass()
class FittingTestCase:
    n: int
    name: str
    title_name: str
    syntax: Union[str, None]
    a: np.ndarray
    x: np.ndarray
    y: List[float]
    x_derivatives: List[float]
    a_derivatives: List[List[float]]
    eps: float = 1e-5
    decimal: int = 5


def add_test_case(
    cls: type, func: FitFunction, case: FittingTestCase, name: str = None
):
    if name is None:
        name = case.name

    def assert_raises_unfit_parameters(self, n0):
        self.assertRaisesRegex(
            ValueError,
            f"input length should be {case.n}, got {n0}$",
            func,
            np.random.random(n0),
            np.random.random(),
        )

    def test_number_of_parameters(self):
        self.assertEqual(
            case.n, func.n, msg="Func gets unexpected number of parameters"
        )
        if case.n > 1:
            assert_raises_unfit_parameters(self, case.n - 1)
        assert_raises_unfit_parameters(self, case.n + 1)

    def test_name(self):
        self.assertEqual(
            case.name, func.name, msg="Func name is different than expected"
        )

    def test_title_name(self):
        self.assertEqual(
            case.title_name,
            func.title_name,
            msg="Func title name is different than expected",
        )

    def test_signature(self):
        self.assertEqual(
            case.name, func.signature, msg="Func signature is different than expected"
        )

    def test_syntax(self):
        self.assertEqual(
            case.syntax, func.syntax, msg="Func syntax is different than expected"
        )

    def test_assign(self):
        assigned_func = func.assign(case.a)
        for x_val, y_val in zip(case.x, case.y):
            self.assertAlmostEqual(
                y_val,
                assigned_func(x_val),
                delta=case.eps,
                msg="Y value is different than expected in assigned function",
            )

    def test_execute_on_single_value(self):
        for x_val, y_val in zip(case.x, case.y):
            self.assertAlmostEqual(
                y_val,
                func(case.a, x_val),
                delta=case.eps,
                msg="Y value is different than expected in called function",
            )

    def test_execute_on_array(self):
        y_array_calculation = func(case.a, case.x)
        for y_val, y_array_val, in zip(case.y, y_array_calculation):
            self.assertAlmostEqual(
                y_val,
                y_array_val,
                delta=case.eps,
                msg="Y value is different than expected in array function",
            )

    def test_execute_x_derivative_on_single_value(self):
        for x_val, x_derivative in zip(case.x, case.x_derivatives):
            self.assertAlmostEqual(
                x_derivative,
                func.x_derivative(case.a, x_val),
                delta=case.eps,
                msg=f"X derivative of ({case.a}, {x_val}) is different than expected",
            )

    def test_execute_x_derivative_on_array(self):
        x_derivative_array_calculation = func.x_derivative(case.a, case.x)
        for expected_x_derivative, actual_x_derivative in zip(
            x_derivative_array_calculation, case.x_derivatives
        ):
            self.assertAlmostEqual(
                expected_x_derivative,
                actual_x_derivative,
                delta=case.eps,
                msg=f"Array calculation of x derivative is different than expected",
            )

    def test_execute_a_derivative_on_single_value(self):
        for x_val, a_derivative in zip(case.x, case.a_derivatives):
            np.testing.assert_almost_equal(
                desired=a_derivative,
                actual=func.a_derivative(case.a, x_val),
                decimal=case.decimal,
                err_msg="A derivative is different than expected",
            )

    def test_execute_a_derivative_on_array(self):
        a_derivative_array_calculation = func.a_derivative(case.a, case.x)
        for expected_a_derivative, actual_a_derivative in zip(
            a_derivative_array_calculation.T, case.a_derivatives
        ):
            np.testing.assert_almost_equal(
                expected_a_derivative,
                actual_a_derivative,
                decimal=case.decimal,
                err_msg=f"Array calculation of a derivative is different than expected",
            )

    def test_is_generator(self):
        self.assertFalse(
            func.is_generator(), msg="Fit function should not be a generator"
        )

    def test_is_costumed(self):
        self.assertFalse(
            func.is_costumed(), msg="Fit function should not be a generator"
        )

    setattr(cls, f"test_{name}_number_of_parameters", test_number_of_parameters)
    setattr(cls, f"test_{name}_name", test_name)
    setattr(cls, f"test_{name}_title_name", test_title_name)
    setattr(cls, f"test_{name}_signature", test_signature)
    setattr(cls, f"test_{name}_syntax", test_syntax)
    setattr(cls, f"test_{name}_assign", test_assign)
    setattr(cls, f"test_{name}_execute_on_single_value", test_execute_on_single_value)
    setattr(cls, f"test_{name}_execute_on_array", test_execute_on_array)
    setattr(
        cls,
        f"test_{name}_execute_x_derivative_on_single_value",
        test_execute_x_derivative_on_single_value,
    )
    setattr(
        cls,
        f"test_{name}_execute_x_derivative_on_array",
        test_execute_x_derivative_on_array,
    )
    setattr(
        cls,
        f"test_{name}_execute_a_derivative_on_single_value",
        test_execute_a_derivative_on_single_value,
    )
    setattr(
        cls,
        f"test_{name}_execute_a_derivative_on_array",
        test_execute_a_derivative_on_array,
    )
    setattr(cls, f"test_{name}_is_generator_returns_false", test_is_generator)
    setattr(cls, f"test_{name}_is_costumed_returns_false", test_is_costumed)


constant_case = FittingTestCase(
    n=1,
    name="constant",
    title_name="Constant",
    syntax="a[0]",
    a=np.array([2]),
    x=np.arange(5),
    y=[2, 2, 2, 2, 2],
    x_derivatives=[0, 0, 0, 0, 0],
    a_derivatives=[[1], [1], [1], [1], [1]],
)
linear_case = FittingTestCase(
    n=2,
    name="linear",
    title_name="Linear",
    syntax="a[0] + a[1] * x",
    a=np.array([-7, 2]),
    x=np.arange(5),
    y=[-7, -5, -3, -1, 1],
    x_derivatives=[2, 2, 2, 2, 2],
    a_derivatives=[[1, 0], [1, 1], [1, 2], [1, 3], [1, 4]],
)
parabolic_case = FittingTestCase(
    n=3,
    name="parabolic",
    title_name="Parabolic",
    syntax="a[0] + a[1] * x + a[2] * x ^ 2",
    a=np.array([3, 4, -2]),
    x=np.arange(5),
    y=[3, 5, 3, -3, -13],
    x_derivatives=[4, 0, -4, -8, -12],
    a_derivatives=[[1, 0, 0], [1, 1, 1], [1, 2, 4], [1, 3, 9], [1, 4, 16]],
)
hyperbolic_case = FittingTestCase(
    n=3,
    name="hyperbolic",
    title_name="Hyperbolic",
    syntax="a[0] / (x + a[1]) + a[2]",
    a=np.array([3, 4, -2]),
    x=np.arange(5),
    y=[-1.25, -1.4, -1.5, -1.57142, -1.625],
    x_derivatives=[-0.1875, -0.12, -0.08333, -0.06122, -0.04687],
    a_derivatives=[
        [0.25, -0.1875, 1],
        [0.2, -0.12, 1],
        [0.16666, -0.08333, 1],
        [0.14285, -0.06122, 1],
        [0.125, -0.04687, 1],
    ],
)
exponential_case = FittingTestCase(
    name="exponential",
    title_name="Exponential",
    syntax="a[0] * exp(a[1] * x) + a[2]",
    n=3,
    a=np.array([3, 0.5, -1]),
    x=np.arange(5),
    y=[2, 3.94616, 7.15484, 12.44506, 21.16716],
    x_derivatives=[1.5, 2.47308, 4.07742, 6.72253, 11.08358],
    a_derivatives=[
        [1, 0, 1],
        [1.64872, 4.94616, 1],
        [2.71828, 16.30969, 1],
        [4.48169, 40.3352, 1],
        [7.38906, 88.66867, 1],
    ],
)
polynom_3_case = FittingTestCase(
    name="polynom_3",
    title_name="Polynom 3",
    syntax=None,
    n=4,
    a=np.array([3, 4, -2, 1]),
    x=np.arange(5),
    y=[3, 6, 11, 24, 51],
    x_derivatives=[4, 3, 8, 19, 36],
    a_derivatives=[
        [1, 0, 0, 0],
        [1, 1, 1, 1],
        [1, 2, 4, 8],
        [1, 3, 9, 27],
        [1, 4, 16, 64],
    ],
)
straight_power_2_case = FittingTestCase(
    name="straight_power_2",
    title_name="Straight Power 2",
    syntax=None,
    n=3,
    a=np.array([2, 1, -3]),
    x=np.arange(5),
    y=[-1, 5, 15, 29, 47],
    x_derivatives=[4, 8, 12, 16, 20],
    a_derivatives=[[1, 4, 1], [4, 8, 1], [9, 12, 1], [16, 16, 1], [25, 20, 1]],
)
straight_power_3_case = FittingTestCase(
    name="straight_power_3",
    title_name="Straight Power 3",
    syntax=None,
    n=3,
    a=np.array([2, 1, -3]),
    x=np.arange(5),
    y=[-1, 13, 51, 125, 247],
    x_derivatives=[6, 24, 54, 96, 150],
    a_derivatives=[[1, 6, 1], [8, 24, 1], [27, 54, 1], [64, 96, 1], [125, 150, 1]],
)
inverse_power_2_case = FittingTestCase(
    name="inverse_power_2",
    title_name="Inverse Power 2",
    syntax=None,
    n=3,
    a=np.array([2, 1, -3]),
    x=np.arange(5),
    y=[-1, -2.5, -2.77777, -2.875, -2.92],
    x_derivatives=[-4, -0.5, -0.14814, -0.0625, -0.032],
    a_derivatives=[
        [1, -4, 1],
        [0.25, -0.5, 1],
        [0.11111, -0.14814, 1],
        [0.0625, -0.0625, 1],
        [0.04, -0.032, 1],
    ],
)
