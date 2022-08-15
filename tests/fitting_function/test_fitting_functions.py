import numpy as np
import pytest
from pytest_cases import THIS_MODULE, parametrize_with_cases

from eddington import (
    FittingFunctionLoadError,
    FittingFunctionRuntimeError,
    constant,
    cos,
    exponential,
    hyperbolic,
    inverse_power,
    linear,
    normal,
    parabolic,
    poisson,
    polynomial,
    sin,
    straight_power,
)


def case_constant():
    return dict(
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
    return dict(
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


def case_polynomial_1():
    return dict(
        func=polynomial(1),
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
    return dict(
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
    return dict(
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
    return dict(
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
    return dict(
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
    return dict(
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


def case_polynomial_3():
    return dict(
        func=polynomial(3),
        func_name="polynomial_3",
        title="Polynomial 3",
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
    return dict(
        func=straight_power,
        func_name="straight_power",
        title="Straight Power",
        syntax="a[0] * (x + a[1]) ^ a[2] + a[3]",
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
    return dict(
        func=straight_power,
        func_name="straight_power",
        title="Straight Power",
        syntax="a[0] * (x + a[1]) ^ a[2] + a[3]",
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
    return dict(
        func=inverse_power,
        func_name="inverse_power",
        title="Inverse Power",
        syntax="a[0] / (x + a[1]) ^ a[2] + a[3]",
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


def case_normal():
    return dict(
        func=normal,
        func_name="normal",
        title="Normal",
        syntax="a[0] * exp( - ((x - a[1]) / a[2]) ^ 2) + a[3]",
        n=4,
        a=np.array([3, 0, 2, 1]),
        x=np.arange(5),
        y=[4.0, 3.336402349, 2.103638, 1.316197, 1.054946916],
        x_derivatives=[
            0,
            -2.3364023492142145,
            -2.207276647028654,
            -0.948593021056779,
            -0.21978766666481014,
        ],
        a_derivatives=[
            [1, 0, 0, 1],
            [0.7788007830714049, 2.3364023492142145, 1.1682011746071073, 1],
            [0.36787944117144233, 2.207276647028654, 1.103638323514327, 1],
            [0.1053992245618643, 0.948593021056779, 0.4742965105283895, 1],
            [0.018315638888734147, 0.21978766666481014, 0.10989383333240507, 1],
        ],
    )


def case_poisson():
    return dict(
        func=poisson,
        func_name="poisson",
        title="Poisson",
        syntax="a[0] * (a[1] ^ x) * exp(-a[1]) / gamma(x+1) + a[2]",
        n=3,
        a=np.array([1, 4, 0]),
        x=np.arange(1, 10),
        y=[
            0.073262556,
            0.146525111,
            0.195366815,
            0.195366815,
            0.156293452,
            0.104195635,
            0.059540363,
            0.029770181,
            0.013231192,
        ],
        x_derivatives=[
            0.0705892068091330,
            0.0679158580633294,
            0.0254322058133843,
            -0.0234094978899069,
            -0.0499862886820318,
            -0.0506901315491914,
            -0.0374715555437846,
            -0.0224570504350002,
            -0.0114510437145612,
        ],
        a_derivatives=[
            [0.0732625555549367, -0.0549469166662025, 1],
            [0.146525111109873, -0.0732625555549367, 1],
            [0.195366814813165, -0.0488417037032911, 1],
            [0.195366814813165, 0, 1],
            [0.156293451850532, 0.0390733629626329, 1],
            [0.104195634567021, 0.0520978172835106, 1],
            [0.0595403626097263, 0.0446552719572948, 1],
            [0.0297701813048632, 0.0297701813048632, 1],
            [0.0132311916910503, 0.0165389896138129, 1],
        ],
    )


def assert_raises_unfit_parameters(case, n0):
    with pytest.raises(
        FittingFunctionRuntimeError,
        match=f"^Input length should be {case['n']}, got {n0}$",
    ):
        case["func"](np.random.random(n0), np.random.random())


@parametrize_with_cases(argnames="case", cases=THIS_MODULE)
def test_number_of_parameters(case):
    assert case["n"] == case["func"].n, "Func gets unexpected number of parameters"
    if case["n"] > 1:
        assert_raises_unfit_parameters(case, case["n"] - 1)
    assert_raises_unfit_parameters(case, case["n"] + 1)


@parametrize_with_cases(argnames="case", cases=THIS_MODULE)
def test_name(case):
    assert (
        case["func_name"] == case["func"].name
    ), "Func name is different than expected"


@parametrize_with_cases(argnames="case", cases=THIS_MODULE)
def test_title_name(case):
    assert (
        case["title"] == case["func"].title_name
    ), "Func title name is different than expected"


@parametrize_with_cases(argnames="case", cases=THIS_MODULE)
def test_syntax(case):
    assert (
        case["syntax"] == case["func"].syntax
    ), "Func syntax is different than expected"


@parametrize_with_cases(argnames="case", cases=THIS_MODULE)
def test_assign(case):
    assigned_func = case["func"].assign(case["a"])
    for i, (x_val, y_val) in enumerate(zip(case["x"], case["y"]), start=1):
        assert float(y_val) == pytest.approx(
            assigned_func(float(x_val)), rel=case.get("eps", 1e-5)
        ), (
            "Y value is different than expected in assigned function "
            f"for the {i} value"
        )
    case["func"].clear_fixed()


@parametrize_with_cases(argnames="case", cases=THIS_MODULE)
def test_execute_on_single_value(case):
    for x_val, y_val in zip(case["x"], case["y"]):
        assert float(y_val) == pytest.approx(
            case["func"](case["a"], float(x_val)), rel=case.get("eps", 1e-5)
        ), "Y value is different than expected in called function"


@parametrize_with_cases(argnames="case", cases=THIS_MODULE)
def test_execute_on_array(case):
    y_array_calculation = case["func"](case["a"], case["x"])
    assert y_array_calculation == pytest.approx(
        case["y"], rel=case.get("eps", 1e-5)
    ), "Y value is different than expected in array function"


@parametrize_with_cases(argnames="case", cases=THIS_MODULE)
def test_execute_x_derivative_on_single_value(case):
    for x_val, x_derivative in zip(case["x"], case["x_derivatives"]):
        assert float(x_derivative) == pytest.approx(
            case["func"].x_derivative(case["a"], float(x_val)),
            rel=case.get("eps", 1e-5),
        ), f"X derivative of ({case['a']}, {x_val}) is different than expected"


@parametrize_with_cases(argnames="case", cases=THIS_MODULE)
def test_execute_x_derivative_on_array(case):
    x_derivative = case["func"].x_derivative(case["a"], case["x"])
    assert x_derivative == pytest.approx(
        case["x_derivatives"], rel=case.get("eps", 1e-5)
    ), "Array calculation of x derivative is different than expected"


@parametrize_with_cases(argnames="case", cases=THIS_MODULE)
def test_execute_a_derivative_on_single_value(case):
    for i, (x_val, a_derivative) in enumerate(
        zip(case["x"], case["a_derivatives"]), start=1
    ):
        assert a_derivative == pytest.approx(
            case["func"].a_derivative(case["a"], float(x_val)),
            rel=case.get("eps", 1e-5),
        ), f"A derivative is different than expected on value {i}"


@parametrize_with_cases(argnames="case", cases=THIS_MODULE)
def test_execute_a_derivative_on_array(case):
    a_derivative_array_calculation = case["func"].a_derivative(case["a"], case["x"])
    for i, (expected_a_derivative, actual_a_derivative) in enumerate(
        zip(a_derivative_array_calculation.T, case["a_derivatives"]), start=1
    ):
        assert np.array(expected_a_derivative) == pytest.approx(
            np.array(actual_a_derivative), rel=case.get("eps", 1e-5)
        ), (
            "Array calculation of a derivative is different than expected "
            f"on value {i}"
        )


def test_initialize_polynomial_with_0_degree_raises_error():
    with pytest.raises(FittingFunctionLoadError, match="^n must be positive, got 0$"):
        polynomial(0)


def test_initialize_polynomial_with_negative_degree_raises_error():
    with pytest.raises(FittingFunctionLoadError, match="^n must be positive, got -1$"):
        polynomial(-1)
