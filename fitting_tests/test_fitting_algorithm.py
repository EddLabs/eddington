import json
from collections import OrderedDict
from pathlib import Path
from unittest import TestCase

import numpy as np

from eddington_core import FitData, FitFunctionsRegistry
from fitting_tests.fitting_test_case import FittingAlgorithmMetaTestCase


def create_integration_test_cases():  # pylint: disable=R0914
    cases_path = Path(__file__).parent / "cases"
    tests = {}
    for json_path in cases_path.glob("**/*.json"):
        with open(str(json_path), mode="r") as json_file:
            json_obj = json.load(json_file)
        name = json_path.stem
        test_case_name = f"TestFittingIntegration{name.title().replace('_', '')}"
        func_name = json_obj["fit_function"]
        fix = json_obj.get("fix", None)
        func = FitFunctionsRegistry.load(func_name)
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
        expected_result = json_obj["result"]
        decimal = json_obj.get("decimal", 2)
        tests[test_case_name] = FittingAlgorithmMetaTestCase(
            test_case_name,
            dct=dict(
                data=data,
                expected_result=expected_result,
                fix=fix,
                a0=a0,
                func=func,
                decimal=decimal,
            ),
        )
    return tests


globals().update(create_integration_test_cases())
