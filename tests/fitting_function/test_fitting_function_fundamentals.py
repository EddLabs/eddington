import numpy as np
import pytest

from eddington import FittingFunctionRuntimeError
from tests.dummy_functions import dummy_func1, dummy_func2

delta = 10e-5


@pytest.fixture
def dummy_func1_fixture():
    yield dummy_func1
    dummy_func1.clear_fixed()


@pytest.fixture
def dummy_func2_fixture():
    yield dummy_func2
    dummy_func2.clear_fixed()


def test_name(dummy_func1_fixture):
    assert dummy_func1_fixture.name == "dummy_func1", "Name is different than expected"


def test_title_name(dummy_func1_fixture):
    assert (
        dummy_func1_fixture.title_name == "Dummy Func1"
    ), "Title name is different than expected"


def test_call_success(dummy_func1_fixture):
    a = np.array([1, 2])
    x = 3
    result = dummy_func1_fixture(a, x)
    assert (
        pytest.approx(result, rel=delta) == 19
    ), "Execution result is different than expected"


def test_call_failure_because_of_not_enough_parameters(dummy_func1_fixture):
    a = np.array([1])
    x = 3
    with pytest.raises(
        FittingFunctionRuntimeError, match="^Input length should be 2, got 1$"
    ):
        dummy_func1_fixture(a, x)


def test_call_failure_because_of_too_many_parameters(dummy_func1_fixture):
    a = np.array([1, 2, 3])
    x = 4
    with pytest.raises(
        FittingFunctionRuntimeError, match="^Input length should be 2, got 3$"
    ):
        dummy_func1_fixture(a, x)


def test_assign(dummy_func1_fixture):
    a = np.array([1, 2])
    x = 3
    dummy_func1_fixture.assign(a)
    result = dummy_func1_fixture(x)
    assert (
        pytest.approx(result, rel=delta) == 19
    ), "Execution result is different than expected"


def test_call_func_with_fix_value(dummy_func2_fixture):
    a = np.array([7, 2, 1])
    x = 2
    dummy_func2_fixture.fix(1, 3)
    result = dummy_func2_fixture(a, x)
    assert (
        pytest.approx(result, rel=delta) == 29
    ), "Execution result is different than expected"


def test_call_x_derivative_with_fix_value(dummy_func2_fixture):
    a = np.array([7, 2, 1])
    x = 2
    dummy_func2_fixture.fix(1, 3)
    result = dummy_func2_fixture.x_derivative(a, x)
    assert (
        pytest.approx(result, rel=delta) == 23
    ), "x derivative execution result is different than expected"


def test_call_a_derivative_with_fix_value(dummy_func2_fixture):
    a = np.array([7, 2, 1])
    x = 2
    dummy_func2_fixture.fix(1, 3)
    result = dummy_func2_fixture.a_derivative(a, x)
    assert result == pytest.approx(
        np.array([1, 4, 8]), rel=delta
    ), "a derivative execution result is different than expected"


def test_override_fix_value(dummy_func2_fixture):
    a = np.array([7, 2, 1])
    x = 2
    dummy_func2_fixture.fix(1, 9)
    dummy_func2_fixture.fix(1, 3)
    result = dummy_func2(a, x)
    assert (
        pytest.approx(result, rel=delta) == 29
    ), "Execution result is different than expected"


def test_unfix_value(dummy_func2_fixture):
    dummy_func2_fixture.fix(1, 5)
    dummy_func2_fixture.unfix(1)
    a = np.array([7, 3, 2, 1])
    x = 2
    result = dummy_func2(a, x)
    assert (
        pytest.approx(result, rel=delta) == 29
    ), "Execution result is different than expected"


def test_clear_fixed(dummy_func2_fixture):
    dummy_func2_fixture.fix(1, 5)
    dummy_func2_fixture.fix(3, 2)
    dummy_func2_fixture.clear_fixed()
    a = np.array([7, 3, 2, 1])
    x = 2
    result = dummy_func2(a, x)
    assert (
        pytest.approx(result, rel=delta) == 29
    ), "Execution result is different than expected"


def test_assign_failure_because_of_not_enough_parameters(dummy_func1_fixture):
    a = np.array([1])
    with pytest.raises(
        FittingFunctionRuntimeError, match="^Input length should be 2, got 1$"
    ):
        dummy_func1_fixture.assign(a)


def test_assign_failure_because_of_too_many_parameters(dummy_func1_fixture):
    a = np.array([1, 2, 3])
    with pytest.raises(
        FittingFunctionRuntimeError, match="^Input length should be 2, got 3$"
    ):
        dummy_func1_fixture.assign(a)


def test_fix_failure_when_trying_to_fix_negative_index(dummy_func1_fixture):
    with pytest.raises(
        FittingFunctionRuntimeError,
        match="^Cannot fix index -1. Indices should be between 0 and 1$",
    ):
        dummy_func1_fixture.fix(-1, 10)


def test_fix_failure_when_trying_to_fix_too_big_index(dummy_func1_fixture):
    with pytest.raises(
        FittingFunctionRuntimeError,
        match="^Cannot fix index 2. Indices should be between 0 and 1$",
    ):
        dummy_func1_fixture.fix(2, 10)


def test_call_failure_because_not_enough_parameters_after_fix(dummy_func2_fixture):
    a = np.array([7, 2])
    x = 2
    dummy_func2_fixture.fix(1, 3)

    with pytest.raises(
        FittingFunctionRuntimeError, match="^Input length should be 3, got 2"
    ):
        dummy_func2_fixture(a, x)


def test_call_failure_because_too_much_parameters_after_fix(dummy_func2_fixture):
    a = np.array([7, 2, 8, 2])
    x = 2
    dummy_func2_fixture.fix(1, 3)

    with pytest.raises(
        FittingFunctionRuntimeError, match="^Input length should be 3, got 4$"
    ):
        dummy_func2_fixture(a, x)


def test_call_failure_when_trying_to_run_without_arguments(dummy_func2_fixture):
    with pytest.raises(
        FittingFunctionRuntimeError,
        match='^No parameters has been given to "dummy_func2"$',
    ):
        dummy_func2_fixture()


def test_fitting_function_representation(dummy_func1_fixture):
    assert str(dummy_func1_fixture) == (
        "FittingFunction(name='dummy_func1', syntax='a[0] + a[1] * x ** 2')"
    ), "Representation is different than expected"
