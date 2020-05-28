from tests.fit_function.fit_function_test_case import (
    FitFunctionMetaTestCase,
    constant_case,
    cos_case,
    exponential_case,
    hyperbolic_case,
    linear_case,
    parabolic_case,
    sin_case,
    straight_power_2_case,
    straight_power_3_case,
    inverse_power_2_case,
)

from eddington_core import FitFunctionsRegistry
from eddington_core import (
    constant,
    exponential,
    hyperbolic,
    linear,
    parabolic,
    cos,
    sin,
    straight_power,
    inverse_power,
)


def add_test_case(test_cases, name, func, case):
    test_cases[name] = FitFunctionMetaTestCase(name, dct=dict(case=case, func=func))


def init_fit_cases(cases_list):
    test_cases = {}
    for data in cases_list:
        func = data["func"]
        case = data["case"]
        case_name = data.get("name", None)
        if case_name is None:
            case_name = func.name
        case_name = case_name.title().replace("_", "")
        add_test_case(test_cases, f"Test{case_name}Fitting", func=func, case=case)
        add_test_case(
            test_cases,
            f"TestLoaded{case_name}Fitting",
            func=FitFunctionsRegistry.load(func.name),
            case=case,
        )
    return test_cases


cases = [
    dict(func=constant, case=constant_case),
    dict(func=linear, case=linear_case),
    dict(func=parabolic, case=parabolic_case),
    dict(func=hyperbolic, case=hyperbolic_case),
    dict(func=exponential, case=exponential_case),
    dict(func=cos, case=cos_case),
    dict(func=sin, case=sin_case),
    dict(func=straight_power, case=straight_power_2_case, name="straight_power_2"),
    dict(func=straight_power, case=straight_power_3_case, name="straight_power_3"),
    # dict(func=inverse_power.fix(2, 1.0), case=hyperbolic_case, name="inverse_power_1"), # noqa: E501
    dict(func=inverse_power, case=inverse_power_2_case, name="inverse_power_2"),
]
globals().update(init_fit_cases(cases))
