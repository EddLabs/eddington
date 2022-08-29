import numpy as np
import pytest

from eddington.exceptions import FittingFunctionParsingError
from eddington.fitting_function_class import FittingFunction
from eddington.fitting_function_parser import parse_fitting_function
from eddington.fitting_functions_registry import FittingFunctionsRegistry


def test_fitting_function_parse_linear_returns_fitting_function(
    clear_functions_registry,
):
    fitting_func = parse_fitting_function(name="linear", syntax="a0 + a1 * x")
    assert isinstance(fitting_func, FittingFunction)


def test_fitting_function_parse_linear_on_float_x(clear_functions_registry):
    fitting_func = parse_fitting_function(name="linear", syntax="a0 + a1 * x")
    res = fitting_func(np.array([1.3, 4.2]), 4)
    assert isinstance(res, float), f"Type of result is {type(res)}"
    np.testing.assert_almost_equal(res, 18.1)


def test_fitting_function_parse_linear_on_array_x(clear_functions_registry):
    fitting_func = parse_fitting_function(name="linear", syntax="a0 + a1 * x")
    res = fitting_func(np.array([1.3, 4.2]), np.array([4, 6]))
    assert isinstance(res, np.ndarray), f"Type of result is {type(res)}"
    np.testing.assert_almost_equal(res, np.array([18.1, 26.5]))


def test_fitting_function_parse_linear_is_saved_to_registry(clear_functions_registry):
    assert not FittingFunctionsRegistry.exists("linear")
    fitting_func = parse_fitting_function(name="linear", syntax="a0 + a1 * x")
    assert FittingFunctionsRegistry.exists("linear")
    fitting_func_from_registry = FittingFunctionsRegistry.load("linear")

    a = np.array([9, 6])
    x = np.linspace(1, 10, num=100)
    np.testing.assert_almost_equal(fitting_func(a, x), fitting_func_from_registry(a, x))


def test_fitting_function_parse_linear_is_not_saved_to_registry(
    clear_functions_registry,
):
    assert not FittingFunctionsRegistry.exists("linear")
    parse_fitting_function(name="linear", syntax="a0 + a1 * x", save=False)
    assert not FittingFunctionsRegistry.exists("linear")


def test_fitting_function_parse_linear_x_derivative(clear_functions_registry):
    fitting_func = parse_fitting_function(name="linear", syntax="a0 + a1 * x")
    res = fitting_func.x_derivative(np.array([1.3, 4.7]), np.linspace(0, 100, num=100))
    np.testing.assert_almost_equal(res, [4.7] * 100)


def test_fitting_function_parse_linear_a_derivative(clear_functions_registry):
    fitting_func = parse_fitting_function(name="linear", syntax="a0 + a1 * x")
    res = fitting_func.a_derivative(np.array([1.3, 4.7]), np.linspace(0, 100, num=100))
    expected = np.array([np.ones(shape=100), np.linspace(0, 100, num=100)])
    np.testing.assert_almost_equal(res, expected)


def test_fitting_function_parse_with_syntax_error(clear_functions_registry):
    with pytest.raises(
        FittingFunctionParsingError,
        match=r'^Could not parse "a0 \+ " into fitting function$',
    ):
        parse_fitting_function(name="bla", syntax="a0 + ")
    assert not FittingFunctionsRegistry.exists("bla")


def test_fitting_function_parse_with_unidentified_variable(clear_functions_registry):
    with pytest.raises(
        FittingFunctionParsingError,
        match=(
            r'^"b" is an invalid variable name\. '
            r'only "x", "a0", "a1",...,"a\{n\}" are excepted\.$'
        ),
    ):
        parse_fitting_function(name="bla", syntax="a0 + b * x")
    assert not FittingFunctionsRegistry.exists("bla")


def test_fitting_function_parse_with_a_without_index(clear_functions_registry):
    with pytest.raises(
        FittingFunctionParsingError,
        match=(
            r'^"a" is an invalid variable name\. '
            r'only "x", "a0", "a1",...,"a\{n\}" are excepted\.$'
        ),
    ):
        parse_fitting_function(name="bla", syntax="a0 + a * x")
    assert not FittingFunctionsRegistry.exists("bla")


def test_fitting_function_parse_without_x(clear_functions_registry):
    with pytest.raises(
        FittingFunctionParsingError,
        match=(
            '^"x" variable was not found. '
            "If you wish to use a constant fitting function, "
            'please use the out-of-the-box "constant" function.$'
        ),
    ):
        parse_fitting_function(name="bla", syntax="a0 + a1 * a2")
    assert not FittingFunctionsRegistry.exists("bla")


def test_fitting_function_parse_with_invalid_a_indices(clear_functions_registry):
    with pytest.raises(
        FittingFunctionParsingError,
        match=(
            r'^"a" variable coordinates should be between 0 and 2 \(included\)\. '
            "the following indices are invalid: 3, 5$"
        ),
    ):
        parse_fitting_function(name="bla", syntax="a0 + a3 * x + a5")
    assert not FittingFunctionsRegistry.exists("bla")
