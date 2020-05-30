from dataclasses import dataclass
from typing import List, Union

import numpy as np

from eddington_core.exceptions import FitFunctionRuntimeError
from eddington_test import MetaTestCase


class FitFunctionMetaTestCase(MetaTestCase):
    def tearDown(self):
        self.func.clear_fixed()

    def assert_raises_unfit_parameters(self, n0):
        self.assertRaisesRegex(
            FitFunctionRuntimeError,
            f"^Input length should be {self.case.n}, got {n0}$",
            self.func,
            np.random.random(n0),
            np.random.random(),
        )

    def test_number_of_parameters(self):
        self.assertEqual(
            self.case.n, self.func.n, msg="Func gets unexpected number of parameters"
        )
        if self.case.n > 1:
            self.assert_raises_unfit_parameters(self.case.n - 1)
        self.assert_raises_unfit_parameters(self.case.n + 1)

    def test_name(self):
        self.assertEqual(
            self.case.name, self.func.name, msg="Func name is different than expected"
        )

    def test_title_name(self):
        self.assertEqual(
            self.case.title_name,
            self.func.title_name,
            msg="Func title name is different than expected",
        )

    def test_signature(self):
        self.assertEqual(
            self.case.name,
            self.func.signature,
            msg="Func signature is different than expected",
        )

    def test_syntax(self):
        self.assertEqual(
            self.case.syntax,
            self.func.syntax,
            msg="Func syntax is different than expected",
        )

    def test_assign(self):
        assigned_func = self.func.assign(self.case.a)
        for i, (x_val, y_val) in enumerate(zip(self.case.x, self.case.y), start=1):
            self.assertAlmostEqual(
                y_val,
                assigned_func(x_val),
                delta=self.case.eps,
                msg="Y value is different than expected in assigned function "
                f"for the {i} value",
            )

    def test_execute_on_single_value(self):
        for x_val, y_val in zip(self.case.x, self.case.y):
            self.assertAlmostEqual(
                y_val,
                self.func(self.case.a, x_val),
                delta=self.case.eps,
                msg="Y value is different than expected in called function",
            )

    def test_execute_on_array(self):  # pylint: disable=W0613
        y_array_calculation = self.func(self.case.a, self.case.x)
        np.testing.assert_almost_equal(
            y_array_calculation,
            self.case.y,
            decimal=self.case.decimal,
            err_msg="Y value is different than expected in array function",
        )

    def test_execute_x_derivative_on_single_value(self):
        for x_val, x_derivative in zip(self.case.x, self.case.x_derivatives):
            self.assertAlmostEqual(
                x_derivative,
                self.func.x_derivative(self.case.a, x_val),
                delta=self.case.eps,
                msg=(
                    f"X derivative of ({self.case.a}, {x_val}) "
                    "is different than expected"
                ),
            )

    def test_execute_x_derivative_on_array(self):  # pylint: disable=W0613
        x_derivative_array_calculation = self.func.x_derivative(
            self.case.a, self.case.x
        )
        np.testing.assert_almost_equal(
            x_derivative_array_calculation,
            self.case.x_derivatives,
            decimal=self.case.decimal,
            err_msg="Array calculation of x derivative is different than expected",
        )

    def test_execute_a_derivative_on_single_value(self):  # pylint: disable=W0613
        for i, (x_val, a_derivative) in enumerate(
            zip(self.case.x, self.case.a_derivatives), start=1
        ):
            np.testing.assert_almost_equal(
                desired=a_derivative,
                actual=self.func.a_derivative(self.case.a, x_val),
                decimal=self.case.decimal,
                err_msg=f"A derivative is different than expected on value {i}",
            )

    def test_execute_a_derivative_on_array(self):  # pylint: disable=W0613
        a_derivative_array_calculation = self.func.a_derivative(
            self.case.a, self.case.x
        )
        for i, (expected_a_derivative, actual_a_derivative) in enumerate(
            zip(a_derivative_array_calculation.T, self.case.a_derivatives), start=1
        ):
            np.testing.assert_almost_equal(
                expected_a_derivative,
                actual_a_derivative,
                decimal=self.case.decimal,
                err_msg="Array calculation of a derivative is different than expected "
                f"on value {i}",
            )


@dataclass()
class FittingFunctionTestCase:
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


constant_case = FittingFunctionTestCase(
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
linear_case = FittingFunctionTestCase(
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
parabolic_case = FittingFunctionTestCase(
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
hyperbolic_case = FittingFunctionTestCase(
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
exponential_case = FittingFunctionTestCase(
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
cos_case = FittingFunctionTestCase(
    name="cos",
    title_name="Cos",
    syntax="a[0] * cos(a[1] * x + a[2]) + a[3]",
    n=4,
    a=np.array([3, 0.5 * np.pi, 0.25 * np.pi, 2]),
    x=np.arange(5),
    y=[4.12132, -0.12132, -0.12132, 4.12132, 4.12132],
    x_derivatives=[-3.33216, -3.33216, 3.33216, 3.33216, -3.33216],
    a_derivatives=[
        [0.70711, -0, -2.12132, 1],
        [-0.70711, -2.12132, -2.12132, 1],
        [-0.70711, 4.24264, 2.12132, 1],
        [0.70711, 6.36396, 2.12132, 1],
        [0.70711, -8.48528, -2.12132, 1],
    ],
)
sin_case = FittingFunctionTestCase(
    name="sin",
    title_name="Sin",
    syntax="a[0] * sin(a[1] * x + a[2]) + a[3]",
    n=4,
    a=np.array([3, 0.5 * np.pi, 0.25 * np.pi, 2]),
    x=np.arange(5),
    y=[4.12132, 4.12132, -0.12132, -0.12132, 4.12132],
    x_derivatives=[3.33216, -3.33216, -3.33216, 3.33216, 3.33216],
    a_derivatives=[
        [0.70711, -0, 2.12132, 1],
        [0.70711, -2.12132, -2.12132, 1],
        [-0.70711, -4.24264, -2.12132, 1],
        [-0.70711, 6.36396, 2.12132, 1],
        [0.70711, 8.48528, 2.12132, 1],
    ],
)
polynom_3_case = FittingFunctionTestCase(
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
straight_power_2_case = FittingFunctionTestCase(
    name="straight_power",
    title_name="Straight Power",
    syntax=None,
    n=4,
    a=np.array([2, 1, 2, -3]),
    x=np.arange(5),
    y=[-1, 5, 15, 29, 47],
    x_derivatives=[4, 8, 12, 16, 20],
    a_derivatives=[
        [1, 4, 0, 1],
        [4, 8, 5.54518, 1],
        [9, 12, 19.77502, 1],
        [16, 16, 44.36142, 1],
        [25, 20, 80.4719, 1],
    ],
)
straight_power_3_case = FittingFunctionTestCase(
    name="straight_power",
    title_name="Straight Power",
    syntax=None,
    n=4,
    a=np.array([2, 1, 3, -3]),
    x=np.arange(5),
    y=[-1, 13, 51, 125, 247],
    x_derivatives=[6, 24, 54, 96, 150],
    a_derivatives=[
        [1, 6, 0, 1],
        [8, 24, 11.09035, 1],
        [27, 54, 59.32506, 1],
        [64, 96, 177.44568, 1],
        [125, 150, 402.35948, 1],
    ],
)
inverse_power_2_case = FittingFunctionTestCase(
    name="inverse_power",
    title_name="Inverse Power",
    syntax=None,
    n=4,
    a=np.array([2, 1, 2, -3]),
    x=np.arange(5),
    y=[-1, -2.5, -2.77777, -2.875, -2.92],
    x_derivatives=[-4, -0.5, -0.14814, -0.0625, -0.032],
    a_derivatives=[
        [1, -4, 0, 1],
        [0.25, -0.5, -5.54518, 1],
        [0.11111, -0.14814, -19.77502, 1],
        [0.0625, -0.0625, -44.36142, 1],
        [0.04, -0.032, -80.4719, 1],
    ],
)
