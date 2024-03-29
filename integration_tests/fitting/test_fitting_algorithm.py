import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
from pytest_cases import THIS_MODULE, parametrize, parametrize_with_cases

from eddington import FittingData, FittingFunctionsRegistry, fit


def cases_paths():
    return (Path(__file__).parent.parent / "resources" / "cases").glob("**/*.json")


@parametrize(idgen="case {case_path.stem}", case_path=cases_paths())
def case_fitting(case_path):  # pylint:disable=too-many-locals
    with open(str(case_path), mode="r", encoding="utf-8") as json_file:
        json_obj = json.load(json_file)
    func_name = json_obj["fit_function"]
    func = FittingFunctionsRegistry.load(func_name)
    fix = json_obj.get("fix", None)
    if fix is not None:
        for index, value in fix:
            func.fix(index, value)
    data_dict = json_obj["data"]
    raw_data = OrderedDict()
    kwargs = {}
    for column_name in ["x", "xerr", "y", "yerr"]:
        column = data_dict.get(column_name, None)
        if column is not None:
            raw_data[column_name] = column
            kwargs[f"{column_name}_column"] = column_name
    a0 = json_obj.get("a0", None)
    fitting_data = FittingData(raw_data, search=False, **kwargs)
    result = fit(data=fitting_data, func=func, a0=a0)
    decimal = json_obj.get("decimal", 2)
    inp = dict(actual_result=result, a0=a0, func=func, delta=np.power(10.0, -decimal))
    expected_result = json_obj["result"]
    return inp, expected_result


@parametrize_with_cases(argnames="inp, expected_result", cases=THIS_MODULE)
def test_a(inp, expected_result, clear_fix):  # pylint: disable=W0613
    assert inp["actual_result"].a == pytest.approx(
        expected_result["a"], rel=inp["delta"]
    ), "A error is different than expected"


@parametrize_with_cases(argnames="inp, expected_result", cases=THIS_MODULE)
def test_aerr(inp, expected_result, clear_fix):  # pylint: disable=W0613
    assert inp["actual_result"].aerr == pytest.approx(
        expected_result["aerr"], rel=inp["delta"]
    ), "A error is different than expected"


@parametrize_with_cases(argnames="inp, expected_result", cases=THIS_MODULE)
def test_arerr(inp, expected_result, clear_fix):  # pylint: disable=W0613
    assert inp["actual_result"].arerr == pytest.approx(
        expected_result["arerr"], rel=inp["delta"]
    ), "A relative error is different than expected"


@parametrize_with_cases(argnames="inp, expected_result", cases=THIS_MODULE)
def test_acov(inp, expected_result, clear_fix):  # pylint: disable=W0613
    actual_acov = inp["actual_result"].acov
    expected_acov = np.array(expected_result["acov"])
    assert (
        actual_acov.shape == expected_acov.shape
    ), "Covariance shapes are different than expected"
    for i in range(actual_acov.shape[0]):
        assert actual_acov[i, :] == pytest.approx(
            expected_acov[i, :], rel=inp["delta"]
        ), f"Covariance is different than expected in line {i}"


@parametrize_with_cases(argnames="inp, expected_result", cases=THIS_MODULE)
def test_chi2(inp, expected_result, clear_fix):  # pylint: disable=W0613
    assert inp["actual_result"].chi2 == pytest.approx(
        expected_result["chi2"], rel=inp["delta"]
    ), "A relative error is different than expected"


@parametrize_with_cases(argnames="inp, expected_result", cases=THIS_MODULE)
def test_degrees_of_freedom(inp, expected_result, clear_fix):  # pylint: disable=W0613
    assert inp["actual_result"].degrees_of_freedom == pytest.approx(
        expected_result["degrees_of_freedom"], rel=inp["delta"]
    ), "A relative error is different than expected"


@parametrize_with_cases(argnames="inp, expected_result", cases=THIS_MODULE)
def test_chi2_reduced(inp, expected_result, clear_fix):  # pylint: disable=W0613
    assert inp["actual_result"].chi2_reduced == pytest.approx(
        expected_result["chi2_reduced"], rel=inp["delta"]
    ), "A relative error is different than expected"


@parametrize_with_cases(argnames="inp, expected_result", cases=THIS_MODULE)
def test_p_probability(inp, expected_result, clear_fix):  # pylint: disable=W0613
    assert inp["actual_result"].p_probability == pytest.approx(
        expected_result["p_probability"], rel=inp["delta"]
    ), "A relative error is different than expected"
