import mock
import numpy as np
import pytest
from pytest_cases import THIS_MODULE, parametrize_with_cases

from eddington import FittingResult
from tests.util import assert_calls


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
\ta[0] = 1.1000 \u00B1 0.1000 (9.091% error)
\ta[1] = 2.9800 \u00B1 0.7600 (25.50% error)
Fitted parameters covariance:
[[0.01  2.3  ]
 [2.3   0.988]]
Chi squared: 8.276
Degrees of freedom: 5
Chi squared reduced: 1.655
P-probability: 0.1417
"""
    fitting_result = FittingResult(**kwargs)
    return (
        dict(
            chi2_reduced=chi2_reduced,
            p_probability=p_probability,
            arerr=arerr,
            repr_string=repr_string,
            delta=10e-5,
            **kwargs,
        ),
        fitting_result,
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
\ta[0] = 1.100 \u00B1 0.000 (0.000% error)
\ta[1] = 2.980 \u00B1 0.000 (0.000% error)
Fitted parameters covariance:
[[0. 0.]
 [0. 0.]]
Chi squared: 8.276
Degrees of freedom: 5
Chi squared reduced: 1.655
P-probability: 0.1417
"""
    fitting_result = FittingResult(**kwargs)
    return (
        dict(
            chi2_reduced=chi2_reduced,
            p_probability=p_probability,
            arerr=arerr,
            repr_string=repr_string,
            delta=10e-5,
            **kwargs,
        ),
        fitting_result,
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
\ta[0] = 0.0000 \u00B1 0.1000 (inf% error)
\ta[1] = 0.0000 \u00B1 0.7600 (inf% error)
Fitted parameters covariance:
[[0.01  2.3  ]
 [2.3   0.988]]
Chi squared: 8.276
Degrees of freedom: 5
Chi squared reduced: 1.655
P-probability: 0.1417
"""
    fitting_result = FittingResult(**kwargs)
    return (
        dict(
            chi2_reduced=chi2_reduced,
            p_probability=p_probability,
            arerr=arerr,
            repr_string=repr_string,
            delta=10e-5,
            **kwargs,
        ),
        fitting_result,
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
\ta[0] = 1.1000 \u00B1 0.1000 (9.091% error)
\ta[1] = 2.9800 \u00B1 0.7600 (25.50% error)
Fitted parameters covariance:
[[0.01  2.3  ]
 [2.3   0.988]]
Chi squared: 43.73
Degrees of freedom: 5
Chi squared reduced: 8.745
P-probability: 2.633e-8
"""
    fitting_result = FittingResult(**kwargs)
    return (
        dict(
            chi2_reduced=chi2_reduced,
            p_probability=p_probability,
            arerr=arerr,
            repr_string=repr_string,
            delta=10e-5,
            **kwargs,
        ),
        fitting_result,
    )


@parametrize_with_cases(argnames="expected, fitting_result", cases=THIS_MODULE)
def test_a0(expected, fitting_result):
    assert fitting_result.a0 == pytest.approx(
        expected["a0"], rel=expected["delta"]
    ), "Initial Guess is different than expected"


@parametrize_with_cases(argnames="expected, fitting_result", cases=THIS_MODULE)
def test_a(expected, fitting_result):
    assert fitting_result.a == pytest.approx(
        expected["a"], rel=expected["delta"]
    ), "Calculated parameters are different than expected"


@parametrize_with_cases(argnames="expected, fitting_result", cases=THIS_MODULE)
def test_aerr(expected, fitting_result):
    assert fitting_result.aerr == pytest.approx(
        expected["aerr"], rel=expected["delta"]
    ), "Parameters errors are different than expected"


@parametrize_with_cases(argnames="expected, fitting_result", cases=THIS_MODULE)
def test_arerr(expected, fitting_result):
    assert fitting_result.arerr == pytest.approx(
        expected["arerr"], rel=expected["delta"]
    ), "Parameters relative errors are different than expected"


@parametrize_with_cases(argnames="expected, fitting_result", cases=THIS_MODULE)
def test_acov(expected, fitting_result):
    expected_acov = np.array(expected["acov"])
    actual_acov = fitting_result.acov
    assert actual_acov.shape == expected_acov.shape
    for i in range(actual_acov.shape[0]):
        assert actual_acov[i, :] == pytest.approx(
            expected_acov[i, :], rel=expected["delta"]
        ), f"Parameters covariance are different than expected in row {i}"


@parametrize_with_cases(argnames="expected, fitting_result", cases=THIS_MODULE)
def test_chi2(expected, fitting_result):
    assert fitting_result.chi2 == pytest.approx(
        expected["chi2"], rel=expected["delta"]
    ), "Chi2 is different than expected"


@parametrize_with_cases(argnames="expected, fitting_result", cases=THIS_MODULE)
def test_chi2_reduced(expected, fitting_result):
    assert fitting_result.chi2_reduced == pytest.approx(
        expected["chi2_reduced"], rel=expected["delta"]
    ), "Chi2 reduced is different than expected"


@parametrize_with_cases(argnames="expected, fitting_result", cases=THIS_MODULE)
def test_degrees_of_freedom(expected, fitting_result):
    assert (
        fitting_result.degrees_of_freedom == expected["degrees_of_freedom"]
    ), "Degrees of freedom are different than expected"


@parametrize_with_cases(argnames="expected, fitting_result", cases=THIS_MODULE)
def test_p_probability(expected, fitting_result):
    assert fitting_result.p_probability == pytest.approx(
        expected["p_probability"], rel=expected["delta"]
    ), "Chi2 reduced is different than expected"


@parametrize_with_cases(argnames="expected, fitting_result", cases=THIS_MODULE)
def test_representation(expected, fitting_result):
    expected_repr = expected["repr_string"].split("\n")
    actual_repr = str(fitting_result).split("\n")
    for i, (expected_line, actual_line) in enumerate(zip(expected_repr, actual_repr)):
        assert (
            actual_line == expected_line
        ), f"Representation is different than expected on line {i}"


@parametrize_with_cases(argnames="expected, fitting_result", cases=THIS_MODULE)
def test_export_to_text_file(expected, fitting_result):
    path = "/path/to/output.txt"
    mock_open_obj = mock.mock_open()
    with mock.patch("eddington.fitting_result.open", mock_open_obj):
        fitting_result.save_txt(path)
        mock_open_obj.assert_called_once_with(path, mode="w", encoding="utf-8")
        mock_open_obj.return_value.write.assert_called_once_with(
            expected["repr_string"]
        )


@parametrize_with_cases(argnames="expected, fitting_result", cases=THIS_MODULE)
def test_export_to_json_file(expected, fitting_result, json_dumps_mock):
    json_string = "This is a json string"
    json_dumps_mock.return_value = json_string
    path = "/path/to/output.json"
    mock_open_obj = mock.mock_open()
    with mock.patch("eddington.fitting_result.open", mock_open_obj):
        fitting_result.save_json(path)
        mock_open_obj.assert_called_once_with(path, mode="w", encoding="utf-8")
        assert_calls(
            json_dumps_mock,
            [
                (
                    [
                        dict(
                            a=expected["a"],
                            a0=expected["a0"],
                            aerr=expected["aerr"],
                            arerr=expected["arerr"],
                            acov=expected["acov"],
                            chi2=expected["chi2"],
                            chi2_reduced=expected["chi2_reduced"],
                            degrees_of_freedom=expected["degrees_of_freedom"],
                            p_probability=expected["p_probability"],
                        ),
                    ],
                    dict(indent=1),
                )
            ],
            rel=expected["delta"],
        )
        mock_open_obj.return_value.write.assert_called_once_with(json_string)
