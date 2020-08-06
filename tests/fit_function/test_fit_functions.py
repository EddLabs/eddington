from dataclasses import dataclass
from typing import List, Union
import numpy as np
import pytest
from pytest_cases import cases_data, THIS_MODULE


from eddington import (
    constant,
    exponential,
    hyperbolic,
    linear,
    parabolic,
    polynom,
    cos,
    sin,
    straight_power,
    inverse_power,
    FitFunctionRuntimeError,
    FitFunctionLoadError,
    FitFunction,
)


@dataclass()
class FittingFunctionTestCase:
    func: FitFunction
    func_name: str
    title: str
    n: int
    syntax: Union[str, None]
    a: np.ndarray
    x: np.ndarray
    y: List[float]
    x_derivatives: List[float]
    a_derivatives: List[List[float]]
    eps: float = 1e-5
    decimal: int = 5


def case_constant():
    return FittingFunctionTestCase(
        func=constant,
        func_name="constant",
        title="Constant",
        n=1,
        syntax="a[0]",
        a=np.array([2]),
        x=np.arange(5),
        y=[2, 2, 2, 2, 2],
        x_derivatives=[0, 0, 0, 0, 0],
        a_derivatives=[[1], [1], [1], [1], [1]],
    )


def case_linear():
    return FittingFunctionTestCase(
        func=linear,
        func_name="linear",
        title="Linear",
        n=2,
        syntax="a[0] + a[1] * x",
        a=np.array([-7, 2]),
        x=np.arange(5),
        y=[-7, -5, -3, -1, 1],
        x_derivatives=[2, 2, 2, 2, 2],
        a_derivatives=[[1, 0], [1, 1], [1, 2], [1, 3], [1, 4]],
    )


def case_polynom_1():
    return FittingFunctionTestCase(
        func=polynom(1),
        func_name="linear",
        title="Linear",
        n=2,
        syntax="a[0] + a[1] * x",
        a=np.array([-7, 2]),
        x=np.arange(5),
        y=[-7, -5, -3, -1, 1],
        x_derivatives=[2, 2, 2, 2, 2],
        a_derivatives=[[1, 0], [1, 1], [1, 2], [1, 3], [1, 4]],
    )


def case_parabolic():
    return FittingFunctionTestCase(
        func=parabolic,
        func_name="parabolic",
        title="Parabolic",
        n=3,
        syntax="a[0] + a[1] * x + a[2] * x ^ 2",
        a=np.array([3, 4, -2]),
        x=np.arange(5),
        y=[3, 5, 3, -3, -13],
        x_derivatives=[4, 0, -4, -8, -12],
        a_derivatives=[[1, 0, 0], [1, 1, 1], [1, 2, 4], [1, 3, 9], [1, 4, 16]],
    )


def case_hyperbolic():
    return FittingFunctionTestCase(
        func=hyperbolic,
        func_name="hyperbolic",
        title="Hyperbolic",
        n=3,
        syntax="a[0] / (x + a[1]) + a[2]",
        a=np.array([3, 4, -2]),
        x=np.arange(5),
        y=[-1.25, -1.4, -1.5, -1.57142, -1.625],
        x_derivatives=[-0.1875, -0.12, -0.0833333, -0.0612244, -0.046875],
        a_derivatives=[
            [0.25, -0.1875, 1],
            [0.2, -0.12, 1],
            [0.1666666, -0.0833333, 1],
            [0.1428571, -0.0612244, 1],
            [0.125, -0.046875, 1],
        ],
    )


def case_exponential():
    return FittingFunctionTestCase(
        func=exponential,
        func_name="exponential",
        title="Exponential",
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


def case_cos():
    return FittingFunctionTestCase(
        func=cos,
        func_name="cos",
        title="Cos",
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


def case_sin():
    return FittingFunctionTestCase(
        func=sin,
        func_name="sin",
        title="Sin",
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


def case_polynom_3():
    return FittingFunctionTestCase(
        func=polynom(3),
        func_name="polynom_3",
        title="Polynom 3",
        syntax="a[0] + a[1] * x + a[2] * x ^ 2 + a[3] * x ^ 3",
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


def case_straight_power_2():
    return FittingFunctionTestCase(
        func=straight_power,
        func_name="straight_power",
        title="Straight Power",
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


def case_straight_power_3():
    return FittingFunctionTestCase(
        func=straight_power,
        func_name="straight_power",
        title="Straight Power",
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


def case_inverse_power_2():
    return FittingFunctionTestCase(
        func=inverse_power,
        func_name="inverse_power",
        title="Inverse Power",
        syntax=None,
        n=4,
        a=np.array([2, 1, 2, -3]),
        x=np.arange(5),
        y=[-1, -2.5, -2.77777, -2.875, -2.92],
        x_derivatives=[-4, -0.5, -0.1481481, -0.0625, -0.032],
        a_derivatives=[
            [1, -4, 0, 1],
            [0.25, -0.5, -5.54518, 1],
            [0.1111111, -0.1481481, -19.77502, 1],
            [0.0625, -0.0625, -44.36142, 1],
            [0.04, -0.032, -80.4719, 1],
        ],
    )


def assert_raises_unfit_parameters(case, n0):
    with pytest.raises(
        FitFunctionRuntimeError, match=f"^Input length should be {case.n}, got {n0}$"
    ):
        case.func(np.random.random(n0), np.random.random())


@cases_data(module=THIS_MODULE)
def test_number_of_parameters(case_data):
    case = case_data.get()
    assert case.n == case.func.n, "Func gets unexpected number of parameters"
    if case.n > 1:
        assert_raises_unfit_parameters(case, case.n - 1)
    assert_raises_unfit_parameters(case, case.n + 1)


@cases_data(module=THIS_MODULE)
def test_name(case_data):
    case = case_data.get()
    assert case.func_name == case.func.name, "Func name is different than expected"


@cases_data(module=THIS_MODULE)
def test_title_name(case_data):
    case = case_data.get()
    assert (
        case.title == case.func.title_name
    ), "Func title name is different than expected"


@cases_data(module=THIS_MODULE)
def test_signature(case_data):
    case = case_data.get()
    assert (
        case.func_name == case.func.signature
    ), "Func signature is different than expected"


@cases_data(module=THIS_MODULE)
def test_syntax(case_data):
    case = case_data.get()
    assert case.syntax == case.func.syntax, "Func syntax is different than expected"


@cases_data(module=THIS_MODULE)
def test_assign(case_data):
    case = case_data.get()
    assigned_func = case.func.assign(case.a)
    for i, (x_val, y_val) in enumerate(zip(case.x, case.y), start=1):
        assert y_val == pytest.approx(assigned_func(x_val), rel=case.eps), (
            "Y value is different than expected in assigned function "
            f"for the {i} value"
        )
    case.func.clear_fixed()


@cases_data(module=THIS_MODULE)
def test_execute_on_single_value(case_data):
    case = case_data.get()
    for x_val, y_val in zip(case.x, case.y):
        assert y_val == pytest.approx(
            case.func(case.a, x_val), rel=case.eps
        ), "Y value is different than expected in called function"


@cases_data(module=THIS_MODULE)
def test_execute_on_array(case_data):  # pylint: disable=W0613
    case = case_data.get()
    y_array_calculation = case.func(case.a, case.x)
    assert y_array_calculation == pytest.approx(
        case.y, rel=case.eps
    ), "Y value is different than expected in array function"


@cases_data(module=THIS_MODULE)
def test_execute_x_derivative_on_single_value(case_data):
    case = case_data.get()
    for x_val, x_derivative in zip(case.x, case.x_derivatives):
        assert x_derivative == pytest.approx(
            case.func.x_derivative(case.a, x_val), rel=case.eps
        ), f"X derivative of ({case.a}, {x_val}) is different than expected"


@cases_data(module=THIS_MODULE)
def test_execute_x_derivative_on_array(case_data):  # pylint: disable=W0613
    case = case_data.get()
    x_derivative_array_calculation = case.func.x_derivative(case.a, case.x)
    assert x_derivative_array_calculation == pytest.approx(
        case.x_derivatives, rel=case.eps
    ), "Array calculation of x derivative is different than expected"


@cases_data(module=THIS_MODULE)
def test_execute_a_derivative_on_single_value(case_data):  # pylint: disable=W0613
    case = case_data.get()
    for i, (x_val, a_derivative) in enumerate(zip(case.x, case.a_derivatives), start=1):
        assert a_derivative == pytest.approx(
            case.func.a_derivative(case.a, x_val), rel=case.eps
        ), f"A derivative is different than expected on value {i}"


@cases_data(module=THIS_MODULE)
def test_execute_a_derivative_on_array(case_data):  # pylint: disable=W0613
    case = case_data.get()
    a_derivative_array_calculation = case.func.a_derivative(case.a, case.x)
    for i, (expected_a_derivative, actual_a_derivative) in enumerate(
        zip(a_derivative_array_calculation.T, case.a_derivatives), start=1
    ):
        assert np.array(expected_a_derivative) == pytest.approx(
            np.array(actual_a_derivative), rel=case.eps
        ), (
            "Array calculation of a derivative is different than expected "
            f"on value {i}"
        )


def test_initialize_polynom_with_0_degree_raises_error():
    with pytest.raises(FitFunctionLoadError, match="^n must be positive, got 0$"):
        polynom(0)


def test_initialize_polynom_with_negative_degree_raises_error():
    with pytest.raises(FitFunctionLoadError, match="^n must be positive, got -1$"):
        polynom(-1)
