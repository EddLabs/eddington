"""Module for build fitting functions from syntax string."""
import re
from numbers import Number
from typing import Dict, Union

import numpy as np
from sympy import Expr, Symbol, diff
from sympy.parsing import parse_expr

from eddington.exceptions import FittingFunctionParsingError
from eddington.fitting_function_class import FittingFunction, fitting_function


def parse_fitting_function(
    name: str, syntax: str, save: bool = True
) -> FittingFunction:
    """
    Parse a syntax string into a :class:`FittingFunction` object.

    :param name: Name of the fitting function
    :type name: str
    :param syntax: Syntax of the fitting function
    :type syntax: str
    :param save: Should the created fitting function be saved into registry.
        True by default.
    :type save: bool
    :return: A fitting function base on the syntax string.
    :rtype: FittingFunction
    :raises FittingFunctionParsingError: Raised when there is an error build a fitting
        function from the given syntax.
    """
    try:
        expr = parse_expr(syntax)
    except SyntaxError as error:
        raise FittingFunctionParsingError(
            f'Could not parse "{syntax}" into fitting function'
        ) from error
    variables_map = {var.name: var for var in expr.free_symbols}
    n = len(variables_map) - 1
    _validate_variables(variable_names=list(variables_map.keys()), n=n)
    x_var = variables_map.pop("x")

    actual_func = _make_function(expr, x_var=x_var, variables_map=variables_map)
    x_derivative = _make_function(
        diff(expr, x_var), x_var=x_var, variables_map=variables_map
    )
    a_derivatives = [
        _make_function(
            diff(expr, variables_map[f"a{i}"]), x_var=x_var, variables_map=variables_map
        )
        for i in range(n)
    ]
    return fitting_function(
        n=n,
        name=name,
        syntax=syntax,
        a_derivative=lambda a, x: np.stack([a_der(a, x) for a_der in a_derivatives]),
        x_derivative=x_derivative,
        save=save,
    )(actual_func)


def _validate_variables(variable_names, n):
    found_x = False
    invalid_indices = []
    for name in variable_names:
        if name == "x":
            found_x = True
            continue
        search_result = re.search(r"^a(?P<index>0|[1-9]\d*)$", name)
        if search_result is None:
            raise FittingFunctionParsingError(
                f'"{name}" is an invalid variable name. only "x", '
                f'"a0", "a1",...,"a{{n}}" are excepted.'
            )
        index = int(search_result.groupdict()["index"])
        if index >= n:
            invalid_indices.append(str(index))
            continue
    if not found_x:
        raise FittingFunctionParsingError(
            '"x" variable was not found. '
            "If you wish to use a constant fitting function, "
            'please use the out-of-the-box "constant" function.'
        )
    if len(invalid_indices) > 0:
        raise FittingFunctionParsingError(
            f'"a" variable coordinates should be between 0 and {n - 1} (included). '
            f"the following indices are invalid: {', '.join(invalid_indices)}"
        )


def _make_function(expr: Expr, x_var: Symbol, variables_map: Dict[str, Symbol]):
    def returned_function(
        a: np.ndarray, x: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        subs = [(variables_map[f"a{i}"], a_val) for i, a_val in enumerate(a)]
        x_expr = expr.subs(subs)
        if isinstance(x, Number):
            return float(x_expr.subs([(x_var, x)]))
        return np.vectorize(lambda x_val: x_expr.subs([(x_var, x_val)]))(x)

    return returned_function
