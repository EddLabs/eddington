import numpy as np
import pytest
from pytest_cases import THIS_MODULE, parametrize_with_cases

from eddington import FitResult


def case_standard():

    kwargs = dict(
        a=[1.1, 2.98],
        aerr=[0.1, 0.76],
        acov=[[0.01, 2.3], [2.3, 0.988]],
        chi2=8.276,
        degrees_of_freedom=5,
    )
    chi2_reduced = 1.6552
    p_probability = 0.14167
    arerr = [9.09091, 25.50336]

    fit_result = FitResult(**kwargs)
    return (
        dict(
            chi2_reduced=chi2_reduced,
            p_probability=p_probability,
            arerr=arerr,
            delta=10e-5,
            **kwargs,
        ),
        fit_result,
    )


def case_negative_value():

    kwargs = dict(
        a=[1.1, -2.98],
        aerr=[0.1, 0.76],
        acov=[[0.01, 2.3], [2.3, 0.988]],
        chi2=8.276,
        degrees_of_freedom=5,
    )
    chi2_reduced = 1.6552
    p_probability = 0.14167
    arerr = [9.09091, 25.50336]
    fit_result = FitResult(**kwargs)
    return (
        dict(
            chi2_reduced=chi2_reduced,
            p_probability=p_probability,
            arerr=arerr,
            delta=10e-5,
            **kwargs,
        ),
        fit_result,
    )


def case_with_zero_error():
    kwargs = dict(
        a=[1.1, 2.98],
        aerr=[0.0, 0.0],
        acov=[[0.0, 0.0], [0.0, 0.0]],
        chi2=8.276,
        degrees_of_freedom=5,
    )
    chi2_reduced = 1.6552
    p_probability = 0.14167
    arerr = [0.0, 0.0]
    fit_result = FitResult(**kwargs)
    return (
        dict(
            chi2_reduced=chi2_reduced,
            p_probability=p_probability,
            arerr=arerr,
            delta=10e-5,
            **kwargs,
        ),
        fit_result,
    )


def case_with_zero_value():

    kwargs = dict(
        a=[0.0, 0.0],
        aerr=[0.1, 0.76],
        acov=[[0.01, 2.3], [2.3, 0.988]],
        chi2=8.276,
        degrees_of_freedom=5,
    )
    chi2_reduced = 1.6552
    p_probability = 0.14167
    arerr = [np.inf, np.inf]
    fit_result = FitResult(**kwargs)
    return (
        dict(
            chi2_reduced=chi2_reduced,
            p_probability=p_probability,
            arerr=arerr,
            delta=10e-5,
            **kwargs,
        ),
        fit_result,
    )


def case_with_small_p_probability():

    kwargs = dict(
        a=[1.1, 2.98],
        aerr=[0.1, 0.76],
        acov=[[0.01, 2.3], [2.3, 0.988]],
        chi2=43.726,
        degrees_of_freedom=5,
    )
    chi2_reduced = 8.7452
    p_probability = 2.63263e-8
    arerr = [9.09091, 25.50336]
    fit_result = FitResult(**kwargs)
    return (
        dict(
            chi2_reduced=chi2_reduced,
            p_probability=p_probability,
            arerr=arerr,
            delta=10e-5,
            **kwargs,
        ),
        fit_result,
    )


@parametrize_with_cases(argnames="expected, fit_result", cases=THIS_MODULE)
def test_a(expected, fit_result):
    assert fit_result.a == pytest.approx(
        expected["a"], rel=expected["delta"]
    ), "Calculated parameters are different than expected"


@parametrize_with_cases(argnames="expected, fit_result", cases=THIS_MODULE)
def test_aerr(expected, fit_result):
    assert fit_result.aerr == pytest.approx(
        expected["aerr"], rel=expected["delta"]
    ), "Parameters errors are different than expected"


@parametrize_with_cases(argnames="expected, fit_result", cases=THIS_MODULE)
def test_arerr(expected, fit_result):
    assert fit_result.arerr == pytest.approx(
        expected["arerr"], rel=expected["delta"]
    ), "Parameters relative errors are different than expected"


@parametrize_with_cases(argnames="expected, fit_result", cases=THIS_MODULE)
def test_acov(expected, fit_result):
    expected_acov = np.array(expected["acov"])
    actual_acov = fit_result.acov
    assert actual_acov.shape == expected_acov.shape
    for i in range(actual_acov.shape[0]):
        assert actual_acov[i, :] == pytest.approx(
            expected_acov[i, :], rel=expected["delta"]
        ), f"Parameters covariance are different than expected in row {i}"


@parametrize_with_cases(argnames="expected, fit_result", cases=THIS_MODULE)
def test_chi2(expected, fit_result):
    assert fit_result.chi2 == pytest.approx(
        expected["chi2"], rel=expected["delta"]
    ), "Chi2 is different than expected"


@parametrize_with_cases(argnames="expected, fit_result", cases=THIS_MODULE)
def test_chi2_reduced(expected, fit_result):
    assert fit_result.chi2_reduced == pytest.approx(
        expected["chi2_reduced"], rel=expected["delta"]
    ), "Chi2 reduced is different than expected"


@parametrize_with_cases(argnames="expected, fit_result", cases=THIS_MODULE)
def test_degrees_of_freedom(expected, fit_result):
    assert (
        fit_result.degrees_of_freedom == expected["degrees_of_freedom"]
    ), "Degrees of freedom are different than expected"


@parametrize_with_cases(argnames="expected, fit_result", cases=THIS_MODULE)
def test_p_probability(expected, fit_result):
    assert fit_result.p_probability == pytest.approx(
        expected["p_probability"], rel=expected["delta"]
    ), "Chi2 reduced is different than expected"
