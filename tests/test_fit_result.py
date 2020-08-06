import mock
import pytest
from pytest_cases import cases_data, THIS_MODULE
import numpy as np

from eddington import FitResult


def case_standard():

    kwargs = dict(
        a0=[1.0, 3.0],
        a=[1.1, 2.98],
        aerr=[0.1, 0.76],
        acov=[[0.01, 2.3], [2.3, 0.988]],
        chi2=8.276,
        degrees_of_freedom=5,
    )
    chi2_reduced = 1.6552
    p_probability = 0.14167
    arerr = [9.09091, 25.50336]
    repr_string = """Results:
========

Initial parameters' values:
\t1.0 3.0
Fitted parameters' values:
\ta[0] = 1.100 \u00B1 0.1000 (9.091% error)
\ta[1] = 2.980 \u00B1 0.7600 (25.503% error)
Fitted parameters covariance:
[[0.01  2.3  ]
 [2.3   0.988]]
Chi squared: 8.276
Degrees of freedom: 5
Chi squared reduced: 1.655
P-probability: 0.1417
"""
    fit_result = FitResult(**kwargs)
    return (
        dict(
            chi2_reduced=chi2_reduced,
            p_probability=p_probability,
            arerr=arerr,
            repr_string=repr_string,
            delta=10e-5,
            **kwargs,
        ),
        fit_result,
    )


def case_with_zero_error():
    kwargs = dict(
        a0=[1.0, 3.0],
        a=[1.1, 2.98],
        aerr=[0.0, 0.0],
        acov=[[0.0, 0.0], [0.0, 0.0]],
        chi2=8.276,
        degrees_of_freedom=5,
    )
    chi2_reduced = 1.6552
    p_probability = 0.14167
    arerr = [0.0, 0.0]
    repr_string = """Results:
========

Initial parameters' values:
\t1.0 3.0
Fitted parameters' values:
\ta[0] = 1.100 ± 0.000 (0.000% error)
\ta[1] = 2.980 ± 0.000 (0.000% error)
Fitted parameters covariance:
[[0. 0.]
 [0. 0.]]
Chi squared: 8.276
Degrees of freedom: 5
Chi squared reduced: 1.655
P-probability: 0.1417
"""
    fit_result = FitResult(**kwargs)
    return (
        dict(
            chi2_reduced=chi2_reduced,
            p_probability=p_probability,
            arerr=arerr,
            repr_string=repr_string,
            delta=10e-5,
            **kwargs,
        ),
        fit_result,
    )


def case_with_zero_value():

    kwargs = dict(
        a0=[1.0, 3.0],
        a=[0.0, 0.0],
        aerr=[0.1, 0.76],
        acov=[[0.01, 2.3], [2.3, 0.988]],
        chi2=8.276,
        degrees_of_freedom=5,
    )
    chi2_reduced = 1.6552
    p_probability = 0.14167
    arerr = [np.inf, np.inf]
    repr_string = """Results:
========

Initial parameters' values:
\t1.0 3.0
Fitted parameters' values:
\ta[0] = 0.000 \u00B1 0.1000 (inf% error)
\ta[1] = 0.000 \u00B1 0.7600 (inf% error)
Fitted parameters covariance:
[[0.01  2.3  ]
 [2.3   0.988]]
Chi squared: 8.276
Degrees of freedom: 5
Chi squared reduced: 1.655
P-probability: 0.1417
"""
    fit_result = FitResult(**kwargs)
    return (
        dict(
            chi2_reduced=chi2_reduced,
            p_probability=p_probability,
            arerr=arerr,
            repr_string=repr_string,
            delta=10e-5,
            **kwargs,
        ),
        fit_result,
    )


def case_with_small_p_probability():

    kwargs = dict(
        a0=[1.0, 3.0],
        a=[1.1, 2.98],
        aerr=[0.1, 0.76],
        acov=[[0.01, 2.3], [2.3, 0.988]],
        chi2=43.726,
        degrees_of_freedom=5,
    )
    chi2_reduced = 8.7452
    p_probability = 2.63263e-8
    arerr = [9.09091, 25.50336]
    repr_string = """Results:
========

Initial parameters' values:
\t1.0 3.0
Fitted parameters' values:
\ta[0] = 1.100 \u00B1 0.1000 (9.091% error)
\ta[1] = 2.980 \u00B1 0.7600 (25.503% error)
Fitted parameters covariance:
[[0.01  2.3  ]
 [2.3   0.988]]
Chi squared: 43.726
Degrees of freedom: 5
Chi squared reduced: 8.745
P-probability: 2.633e-08
"""
    fit_result = FitResult(**kwargs)
    return (
        dict(
            chi2_reduced=chi2_reduced,
            p_probability=p_probability,
            arerr=arerr,
            repr_string=repr_string,
            delta=10e-5,
            **kwargs,
        ),
        fit_result,
    )


@cases_data(module=THIS_MODULE)
def test_a0(case_data):
    expected, fit_result = case_data.get()
    assert fit_result.a0 == pytest.approx(
        expected["a0"], rel=expected["delta"]
    ), "Initial Guess is different than expected"


@cases_data(module=THIS_MODULE)
def test_a(case_data):
    expected, fit_result = case_data.get()
    assert fit_result.a == pytest.approx(
        expected["a"], rel=expected["delta"]
    ), "Calculated parameters are different than expected"


@cases_data(module=THIS_MODULE)
def test_aerr(case_data):
    expected, fit_result = case_data.get()
    assert fit_result.aerr == pytest.approx(
        expected["aerr"], rel=expected["delta"]
    ), "Parameters errors are different than expected"


@cases_data(module=THIS_MODULE)
def test_arerr(case_data):
    expected, fit_result = case_data.get()
    assert fit_result.arerr == pytest.approx(
        expected["arerr"], rel=expected["delta"]
    ), "Parameters relative errors are different than expected"


@cases_data(module=THIS_MODULE)
def test_acov(case_data):
    expected, fit_result = case_data.get()
    expected_acov = np.array(expected["acov"])
    actual_acov = fit_result.acov
    assert actual_acov.shape == expected_acov.shape
    for i in range(actual_acov.shape[0]):
        assert actual_acov[i, :] == pytest.approx(
            expected_acov[i, :], rel=expected["delta"]
        ), f"Parameters covariance are different than expected in row {i}"


@cases_data(module=THIS_MODULE)
def test_chi2(case_data):
    expected, fit_result = case_data.get()
    assert fit_result.chi2 == pytest.approx(
        expected["chi2"], rel=expected["delta"]
    ), "Chi2 is different than expected"


@cases_data(module=THIS_MODULE)
def test_chi2_reduced(case_data):
    expected, fit_result = case_data.get()
    assert fit_result.chi2_reduced == pytest.approx(
        expected["chi2_reduced"], rel=expected["delta"]
    ), "Chi2 reduced is different than expected"


@cases_data(module=THIS_MODULE)
def test_degrees_of_freedom(case_data):
    expected, fit_result = case_data.get()
    assert (
        fit_result.degrees_of_freedom == expected["degrees_of_freedom"]
    ), "Degrees of freedom are different than expected"


@cases_data(module=THIS_MODULE)
def test_p_probability(case_data):
    expected, fit_result = case_data.get()
    assert fit_result.p_probability == pytest.approx(
        expected["p_probability"], rel=expected["delta"]
    ), "Chi2 reduced is different than expected"


@cases_data(module=THIS_MODULE)
def test_representation(case_data):
    expected, fit_result = case_data.get()
    assert expected["repr_string"] == str(
        fit_result
    ), "Representation is different than expected"


@cases_data(module=THIS_MODULE)
def test_export_to_file(case_data):
    expected, fit_result = case_data.get()
    path = "/path/to/output"
    mock_open_obj = mock.mock_open()
    with mock.patch("eddington.fit_result.open", mock_open_obj):
        fit_result.print_or_export(path)
        mock_open_obj.assert_called_once_with(path, mode="w")
        mock_open_obj.return_value.write.assert_called_with(expected["repr_string"])


@cases_data(module=THIS_MODULE)
def test_print(case_data):
    expected, fit_result = case_data.get()
    with mock.patch("sys.stdout") as mock_print:
        fit_result.print_or_export()
        assert mock_print.write.call_args_list[0] == mock.call(expected["repr_string"])
