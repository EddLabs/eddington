import numpy as np
import pytest
from mock import call
from pytest_cases import fixture, parametrize

from eddington.consts import (
    DEFAULT_MAX_COEFF,
    DEFAULT_MEASUREMENTS,
    DEFAULT_MIN_COEFF,
    DEFAULT_XMAX,
    DEFAULT_XMIN,
    DEFAULT_XSIGMA,
    DEFAULT_YSIGMA,
)
from eddington.random_util import random_data
from tests.fitting_function.dummy_functions import dummy_func1

a = [2, 1]
x = np.arange(0, DEFAULT_MEASUREMENTS)
xerr = np.random.normal(size=DEFAULT_MEASUREMENTS)
yerr = np.random.normal(size=DEFAULT_MEASUREMENTS)
real_xerr = np.random.normal(size=DEFAULT_MEASUREMENTS)
real_yerr = np.random.normal(size=DEFAULT_MEASUREMENTS)
delta = 10e-5


@fixture
def random_sigma_mock(mocker):
    random_sigma = mocker.patch("eddington.random_util.random_sigma")
    random_sigma.side_effect = [xerr, yerr]
    return random_sigma


@fixture
def random_error_mock(mocker):
    random_error = mocker.patch("eddington.random_util.random_error")
    random_error.side_effect = [real_xerr, real_yerr]
    return random_error


@fixture
@parametrize(
    idgen="args={args}",
    argnames="args",
    argvalues=[
        dict(),
        dict(measurements=50),
        dict(min_coeff=9),
        dict(max_coeff=34),
        dict(xmin=12),
        dict(xmax=53),
        dict(xsigma=2.1),
        dict(ysigma=3.9),
        dict(a=[5, 3]),
        dict(x=x),
    ],
)
def fitting_arguments(args):
    return args


@fixture
def random_fitting_data(
    mocker, fitting_arguments, random_sigma_mock, random_error_mock
):
    random_array_mock = mocker.patch("eddington.random_util.random_array")
    random_array_side_effect = []
    if "a" not in fitting_arguments:
        random_array_side_effect.append(a)
    if "x" not in fitting_arguments:
        random_array_side_effect.append(x)
    random_array_mock.side_effect = random_array_side_effect
    return (
        random_data(dummy_func1, **fitting_arguments),
        dict(
            params=fitting_arguments,
            random_array=random_array_mock,
            random_sigma=random_sigma_mock,
            random_error=random_error_mock,
        ),
    )


def test_x_data(random_fitting_data):
    data, _ = random_fitting_data
    assert data.x == pytest.approx(
        x, rel=delta
    ), "Random x value of data is different than expected"


def test_xerr_data(random_fitting_data):
    data, _ = random_fitting_data
    assert data.xerr == pytest.approx(
        xerr, rel=delta
    ), "Random x error value of data is different than expected"


def test_y_data(random_fitting_data):
    data, expected = random_fitting_data
    params = expected["params"]
    actual_a = params.get("a", a)
    y = dummy_func1(actual_a, x + real_xerr) + real_yerr
    assert data.y == pytest.approx(
        y, rel=delta
    ), "Random y value of data is different than expected"


def test_yerr_data(random_fitting_data):
    data, _ = random_fitting_data
    assert data.yerr == pytest.approx(
        yerr, rel=delta
    ), "Random y error value of data is different than expected"


def test_random_array_calls(random_fitting_data):
    _, expected = random_fitting_data
    random_array = expected["random_array"]
    params = expected["params"]
    amin = params.get("min_coeff", DEFAULT_MIN_COEFF)
    amax = params.get("max_coeff", DEFAULT_MAX_COEFF)
    xmin = params.get("xmin", DEFAULT_XMIN)
    xmax = params.get("xmax", DEFAULT_XMAX)
    measurements = params.get("measurements", DEFAULT_MEASUREMENTS)

    call_count = 0
    if "a" not in params:
        assert random_array.call_args_list[call_count] == call(
            min_val=amin, max_val=amax, size=dummy_func1.n
        )
        call_count += 1
    if "x" not in params:
        assert random_array.call_args_list[call_count] == call(
            min_val=xmin, max_val=xmax, size=measurements
        )
        call_count += 1
    assert random_array.call_count == call_count


def test_random_sigma_calls(random_fitting_data):
    _, expected = random_fitting_data
    random_sigma = expected["random_sigma"]
    params = expected["params"]
    xsigma = params.get("xsigma", DEFAULT_XSIGMA)
    ysigma = params.get("ysigma", DEFAULT_YSIGMA)
    measurements = params.get("measurements", DEFAULT_MEASUREMENTS)
    assert random_sigma.call_count == 2
    assert random_sigma.call_args_list[0] == call(
        average_sigma=xsigma, size=measurements
    )
    assert random_sigma.call_args_list[1] == call(
        average_sigma=ysigma, size=measurements
    )


def test_random_error_calls(random_fitting_data):
    _, expected = random_fitting_data
    random_error = expected["random_error"]
    assert random_error.call_count == 2
    assert random_error.call_args_list[0] == call(scales=xerr)
    assert random_error.call_args_list[1] == call(scales=yerr)
