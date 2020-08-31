"""Printing functions for displaying numbers with given precision."""
import math
from typing import List, Optional, Tuple, Union

import numpy as np

from eddington import FitResult


def fit_result_pretty_string(  # pylint: disable=invalid-name
    fit_result: FitResult, a0: Optional[Union[List[float], np.ndarray]] = None
) -> str:
    """
    Create a pretty representation string for a fit result.

    :param fit_result: the fit result to create it's pretty string
    :type fit_result: FitResult
    :param a0: Optional initial guess for the result.
    :type a0: list of ``float`` or ``numpy.ndarray``
    :returns: str
    """
    old_precision = np.get_printoptions()["precision"]
    precision = fit_result.precision
    np.set_printoptions(precision=precision)
    a_value_string = "\n".join(
        [
            __a_value_string(i=i, a=a, aerr=aerr, arerr=arerr, precision=precision)
            for i, (a, aerr, arerr) in enumerate(
                zip(fit_result.a, fit_result.aerr, fit_result.arerr)
            )
        ]
    )
    repr_string = f"""Results:
========

{__initial_parameters_string(a0)}Fitted parameters' values:
{a_value_string}
Fitted parameters covariance:
{fit_result.acov}
Chi squared: {__to_precise_string(fit_result.chi2, precision)}
Degrees of freedom: {fit_result.degrees_of_freedom}
Chi squared reduced: {__to_precise_string(fit_result.chi2_reduced, precision)}
P-probability: {__to_precise_string(fit_result.p_probability, precision)}
"""
    np.set_printoptions(precision=old_precision)
    return repr_string


def __initial_parameters_string(  # pylint: disable=invalid-name
    a0: Optional[Union[List[float], np.ndarray]]
):
    if a0 is None:
        return ""
    return f"""Initial parameters' values:
\t{" ".join(str(i) for i in a0)}
"""


def __to_relevant_precision(decimal: float) -> Tuple[float, int]:
    """
    Get relevant precision of a decimal number.

    :param decimal: a floating point number.
    :return: a tuple of a and n such that: decimal = a * 10^(-b).
    """
    if decimal == 0:
        return 0, 0
    precision = 0
    abs_a = math.fabs(decimal)
    while abs_a < 1.0:
        abs_a *= 10
        precision += 1
    if decimal < 0:
        return -abs_a, precision
    return abs_a, precision


def __to_precise_string(decimal: float, precision: int) -> str:
    """
    Returns a decimal as string with desired precision.

    :param decimal: a floating point number.
    :param precision: the desired precision to return the number.
    :return: a string representing the decimal with given precision.
    """
    new_decimal, relevant_precision = __to_relevant_precision(decimal)
    if relevant_precision < 3:
        return f"{decimal:.{precision + relevant_precision}f}"
    return f"{new_decimal:.{precision}f}e-0{relevant_precision}"


def __a_value_string(  # pylint: disable=invalid-name
    i: int, a: float, aerr: float, arerr: float, precision: int
) -> str:
    a_string = __to_precise_string(a, precision)
    aerr_string = __to_precise_string(aerr, precision)
    arerr_string = __to_precise_string(arerr, precision)
    return f"\ta[{i}] = {a_string} \u00B1 {aerr_string} ({arerr_string}% error)"
