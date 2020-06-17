import pytest

import numpy as np
from eddington_core import FitFunctionRuntimeError

from tests.fit_function.dummy_functions import dummy_func1, dummy_func2

delta = 10e-5


@pytest.fixture
def func_fixture():
    yield dummy_func1
    dummy_func1.clear_fixed()


def test_name(func_fixture):
    assert "dummy_func1" == func_fixture.name, "Name is different than expected"


def test_signature(func_fixture):
    assert (
        "dummy_func1" == func_fixture.signature
    ), "Signature is different than expected"


def test_title_name(func_fixture):
    assert (
        "Dummy Func1" == func_fixture.title_name
    ), "Title name is different than expected"


def test_call_success(func_fixture):
    a = np.array([1, 2])
    x = 3
    result = func_fixture(a, x)
    assert 19 == pytest.approx(
        result, rel=delta
    ), "Execution result is different than expected"


def test_call_failure_because_of_not_enough_parameters(func_fixture):
    a = np.array([1])
    x = 3
    with pytest.raises(
        FitFunctionRuntimeError, match="^Input length should be 2, got 1$"
    ):
        func_fixture(a, x)


def test_call_failure_because_of_too_many_parameters(func_fixture):
    a = np.array([1, 2, 3])
    x = 4
    with pytest.raises(
        FitFunctionRuntimeError, match="^Input length should be 2, got 3$"
    ):
        func_fixture(a, x)


def test_assign(func_fixture):
    a = np.array([1, 2])
    x = 3
    func_fixture.assign(a)
    result = func_fixture(x)
    assert 19 == pytest.approx(
        result, rel=delta
    ), "Execution result is different than expected"


def test_call_func_with_fix_value(func_fixture):
    a = np.array([7, 2, 1])
    x = 2
    dummy_func2.fix(1, 3)
    result = dummy_func2(a, x)
    assert 29 == pytest.approx(
        result, rel=delta
    ), "Execution result is different than expected"


def test_call_x_derivative_with_fix_value(func_fixture):
    a = np.array([7, 2, 1])
    x = 2
    dummy_func2.fix(1, 3)
    result = dummy_func2.x_derivative(a, x)
    assert 23 == pytest.approx(
        result, rel=delta
    ), "x derivative execution result is different than expected"


def test_call_a_derivative_with_fix_value(func_fixture):
    a = np.array([7, 2, 1])
    x = 2
    dummy_func2.fix(1, 3)
    result = dummy_func2.a_derivative(a, x)
    assert result == pytest.approx(
        np.array([1, 4, 8]), rel=delta
    ), "a derivative execution result is different than expected"


def test_override_fix_value(func_fixture):
    a = np.array([7, 2, 1])
    x = 2
    dummy_func2.fix(1, 9)
    dummy_func2.fix(1, 3)
    result = dummy_func2(a, x)
    assert 29 == pytest.approx(
        result, rel=delta
    ), "Execution result is different than expected"


def test_unfix_value(func_fixture):
    dummy_func2.fix(1, 5)
    dummy_func2.unfix(1)
    a = np.array([7, 3, 2, 1])
    x = 2
    result = dummy_func2(a, x)
    assert 29 == pytest.approx(
        result, rel=delta
    ), "Execution result is different than expected"


def test_clear_fixed(func_fixture):
    dummy_func2.fix(1, 5)
    dummy_func2.fix(3, 2)
    dummy_func2.clear_fixed()
    a = np.array([7, 3, 2, 1])
    x = 2
    result = dummy_func2(a, x)
    assert 29 == pytest.approx(
        result, rel=delta
    ), "Execution result is different than expected"


def test_assign_failure_because_of_not_enough_parameters(func_fixture):
    a = np.array([1])
    with pytest.raises(
        FitFunctionRuntimeError, match="^Input length should be 2, got 1$"
    ):
        func_fixture.assign(a)


def test_assign_failure_because_of_too_many_parameters(func_fixture):
    a = np.array([1, 2, 3])
    with pytest.raises(
        FitFunctionRuntimeError, match="^Input length should be 2, got 3$"
    ):
        func_fixture.assign(a)


def test_fix_failure_when_trying_to_fix_negative_index(func_fixture):
    with pytest.raises(
        FitFunctionRuntimeError,
        match="^Cannot fix index -1. Indices should be between 0 and 1$",
    ):
        func_fixture.fix(
            -1, 10,
        )


def test_fix_failure_when_trying_to_fix_too_big_index(func_fixture):
    with pytest.raises(
        FitFunctionRuntimeError,
        match="^Cannot fix index 2. Indices should be between 0 and 1$",
    ):
        func_fixture.fix(2, 10)


def test_call_failure_because_not_enough_parameters_after_fix(func_fixture):
    a = np.array([7, 2])
    x = 2
    dummy_func2.fix(1, 3)

    with pytest.raises(
        FitFunctionRuntimeError, match="^Input length should be 4, got 3$"
    ):
        dummy_func2(a, x)


def test_call_failure_because_too_much_parameters_after_fix(func_fixture):
    a = np.array([7, 2, 8, 2])
    x = 2
    dummy_func2.fix(1, 3)

    with pytest.raises(
        FitFunctionRuntimeError, match="^Input length should be 4, got 5$"
    ):
        dummy_func2(a, x)


def test_call_failure_when_trying_to_run_without_arguments(func_fixture):
    with pytest.raises(
        FitFunctionRuntimeError, match='^No parameters has been given to "dummy_func2"$'
    ):
        dummy_func2()


def test_fit_function_representation(func_fixture):
    assert "FitFunction(name='dummy_func1', syntax='a[0] + a[1] * x ** 2')" == str(
        dummy_func1
    ), "Representation is different than expected"
