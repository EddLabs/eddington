from argparse import Namespace

import numpy as np
import pytest

from eddington import fit, fitting_function
from eddington.exceptions import FittingError
from eddington.random_util import random_data

a0 = np.array([8, 5])
a = np.array([3, 4])
aerr = np.array([0.1, 0.2])
acov = np.array([[0.1, 0.2], [0.2, 0.3]])
chi2 = 1.5


def dummy_func_x_derivative(a, x):
    return 2 * a[0] * x


def dummy_func_a_derivative(a, x):  # pylint: disable=W0613
    return np.stack([x**2, np.ones(shape=x.shape)])


@fitting_function(n=2, save=False)
def dummy_func(a, x):
    return a[0] * x**2 + a[1]


@fitting_function(n=2, x_derivative=dummy_func_x_derivative, save=False)
def dummy_func_with_x_derivative(a, x):
    return a[0] * x**2 + a[1]


@fitting_function(n=2, a_derivative=dummy_func_a_derivative, save=False)
def dummy_func_with_a_derivative(a, x):
    return a[0] * x**2 + a[1]


@fitting_function(
    n=2,
    x_derivative=dummy_func_x_derivative,
    a_derivative=dummy_func_a_derivative,
    save=False,
)
def dummy_func_with_both_derivatives(a, x):
    return a[0] * x**2 + a[1]


@pytest.fixture
def odr_mock(mocker):
    odr = mocker.patch("eddington.fitting.ODR")
    real_data = mocker.patch("eddington.fitting.RealData")
    model = mocker.patch("eddington.fitting.Model")
    odr.return_value.run.return_value = Namespace(
        beta=a, sum_square=chi2, sd_beta=aerr, cov_beta=acov
    )
    return dict(odr=odr, real_data=real_data, model=model)


@pytest.fixture(
    params=[
        dict(func=dummy_func),
        dict(
            func=dummy_func_with_x_derivative,
            model_extra_kwargs=dict(fjacd=dummy_func_with_x_derivative.x_derivative),
        ),
        dict(
            func=dummy_func_with_a_derivative,
            model_extra_kwargs=dict(fjacb=dummy_func_with_a_derivative.a_derivative),
        ),
        dict(
            func=dummy_func_with_both_derivatives,
            model_extra_kwargs=dict(
                fjacd=dummy_func_with_both_derivatives.x_derivative,
                fjacb=dummy_func_with_both_derivatives.a_derivative,
            ),
        ),
        dict(
            func=dummy_func_with_both_derivatives,
            kwargs=dict(use_x_derivative=False),
            model_extra_kwargs=dict(
                fjacb=dummy_func_with_both_derivatives.a_derivative,
            ),
        ),
        dict(
            func=dummy_func_with_both_derivatives,
            kwargs=dict(use_a_derivative=False),
            model_extra_kwargs=dict(
                fjacd=dummy_func_with_both_derivatives.x_derivative,
            ),
        ),
        dict(func=dummy_func, a0=None),
    ]
)
def function_cases(odr_mock, request):
    func, kwargs, model_extra_kwargs, fit_a0 = (
        request.param["func"],
        request.param.get("kwargs", {}),
        request.param.get("model_extra_kwargs", {}),
        request.param.get("a0", a0),
    )
    data = random_data(fit_func=func)
    result = fit(data=data, func=func, a0=fit_a0, **kwargs)
    return dict(
        func=func,
        data=data,
        result=result,
        model_extra_kwargs=model_extra_kwargs,
        a0=fit_a0,
        mocks=odr_mock,
    )


def test_fit_result(function_cases):
    assert function_cases["result"].a == pytest.approx(
        a
    ), "Result is different than expected"


def test_model(function_cases):
    model_extra_kwargs = function_cases["model_extra_kwargs"]
    function_cases["mocks"]["model"].assert_called_once_with(
        fcn=function_cases["func"], **model_extra_kwargs
    )


def test_real_data(function_cases):
    real_data = function_cases["mocks"]["real_data"]
    data = function_cases["data"]
    assert real_data.call_args[1].keys() == {
        "sx",
        "sy",
        "y",
        "x",
    }, "Real data arguments are different than expected"
    assert real_data.call_args[1]["x"] == pytest.approx(
        data.x
    ), "X is different than expected"
    assert real_data.call_args[1]["sx"] == pytest.approx(
        data.xerr
    ), "X error is different than expected"

    assert real_data.call_args[1]["y"] == pytest.approx(
        data.y
    ), "Y is different than expected"
    assert real_data.call_args[1]["sy"] == pytest.approx(
        data.yerr
    ), "Y error is different than expected"


def test_odr(function_cases):
    model = function_cases["mocks"]["model"]
    real_data = function_cases["mocks"]["real_data"]
    odr = function_cases["mocks"]["odr"]
    fit_a0 = function_cases["a0"]
    if fit_a0 is None:
        fit_a0 = np.ones(shape=2)
    assert odr.call_args[1].keys() == {"data", "model", "beta0"}
    assert odr.call_args[1]["data"] == real_data.return_value
    assert odr.call_args[1]["model"] == model.return_value
    assert odr.call_args[1]["beta0"] == pytest.approx(fit_a0)


def test_fitting_fail_for_no_x():
    fitting_data = random_data(dummy_func)
    fitting_data.x_column = None

    with pytest.raises(FittingError, match="^Cannot fit data without x values$"):
        fit(data=fitting_data, func=dummy_func)


def test_fitting_fail_for_no_y():
    fitting_data = random_data(dummy_func)
    fitting_data.y_column = None

    with pytest.raises(FittingError, match="^Cannot fit data without y values$"):
        fit(data=fitting_data, func=dummy_func)
