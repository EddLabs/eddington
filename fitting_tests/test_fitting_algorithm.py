import json

import numpy as np
import pytest
from pytest_cases import cases_generator, cases_data, THIS_MODULE
from collections import OrderedDict
from pathlib import Path

from eddington_core import FitData, FitFunctionsRegistry, fit_to_data


def cases_paths():
    return (Path(__file__).parent / "cases").glob("**/*.json")


@cases_generator("case {case_path.stem}", case_path=cases_paths())
def case_fitting(case_path):
    with open(str(case_path), mode="r") as json_file:
        json_obj = json.load(json_file)
    func_name = json_obj["fit_function"]
    func = FitFunctionsRegistry.load(func_name)
    fix = json_obj.get("fix", None)
    if fix is not None:
        for index, value in fix:
            func.fix(index, value)
    data_dict = json_obj["data"]
    data = FitData(
        OrderedDict(
            [
                ("x", data_dict["x"]),
                ("xerr", data_dict["xerr"]),
                ("y", data_dict["y"]),
                ("yerr", data_dict["yerr"]),
            ]
        )
    )
    a0 = json_obj.get("a0", None)
    result = fit_to_data(data=data, func=func)
    decimal = json_obj.get("decimal", 2)
    inp = dict(actual_result=result, a0=a0, func=func, delta=np.power(10.0, -decimal))
    expected_result = json_obj["result"]
    return inp, expected_result


@cases_data(module=THIS_MODULE)
def test_a(case_data, clear_fix):  # pylint: disable=W0613
    inp, expected_result = case_data.get()
    assert inp["actual_result"].a == pytest.approx(
        expected_result["a"], rel=inp["delta"]
    ), "A error is different than expected"


@cases_data(module=THIS_MODULE)
def test_aerr(case_data, clear_fix):  # pylint: disable=W0613
    inp, expected_result = case_data.get()
    assert inp["actual_result"].aerr == pytest.approx(
        expected_result["aerr"], rel=inp["delta"]
    ), "A error is different than expected"


@cases_data(module=THIS_MODULE)
def test_arerr(case_data, clear_fix):  # pylint: disable=W0613
    inp, expected_result = case_data.get()
    assert inp["actual_result"].arerr == pytest.approx(
        expected_result["arerr"], rel=inp["delta"]
    ), "A relative error is different than expected"


@cases_data(module=THIS_MODULE)
def test_acov(case_data, clear_fix):  # pylint: disable=W0613
    inp, expected_result = case_data.get()
    actual_acov = inp["actual_result"].acov
    expected_acov = np.array(expected_result["acov"])
    assert (
        actual_acov.shape == expected_acov.shape
    ), "Covariance shapes are different than expected"
    for i in range(actual_acov.shape[0]):
        assert actual_acov[i, :] == pytest.approx(
            expected_acov[i, :], rel=inp["delta"]
        ), f"Covariance is different than expected in line {i}"


@cases_data(module=THIS_MODULE)
def test_chi2(case_data, clear_fix):  # pylint: disable=W0613
    inp, expected_result = case_data.get()
    assert inp["actual_result"].chi2 == pytest.approx(
        expected_result["chi2"], rel=inp["delta"]
    ), "A relative error is different than expected"


@cases_data(module=THIS_MODULE)
def test_degrees_of_freedom(case_data, clear_fix):  # pylint: disable=W0613
    inp, expected_result = case_data.get()
    assert inp["actual_result"].degrees_of_freedom == pytest.approx(
        expected_result["degrees_of_freedom"], rel=inp["delta"]
    ), "A relative error is different than expected"


@cases_data(module=THIS_MODULE)
def test_chi2_reduced(case_data, clear_fix):  # pylint: disable=W0613
    inp, expected_result = case_data.get()
    assert inp["actual_result"].chi2_reduced == pytest.approx(
        expected_result["chi2_reduced"], rel=inp["delta"]
    ), "A relative error is different than expected"


@cases_data(module=THIS_MODULE)
def test_p_probability(case_data, clear_fix):  # pylint: disable=W0613
    inp, expected_result = case_data.get()
    assert inp["actual_result"].p_probability == pytest.approx(
        expected_result["p_probability"], rel=inp["delta"]
    ), "A relative error is different than expected"
