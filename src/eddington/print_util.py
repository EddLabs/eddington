"""Printing functions for displaying numbers with given precision."""
import math
from typing import Tuple

import numpy as np

from eddington.consts import DEFAULT_PRECISION


def to_precise_string(decimal: float, precision: int = DEFAULT_PRECISION) -> str:
    """
    Returns a decimal as string with desired precision.

    :param decimal: a floating point number.
    :param precision: the desired precision to return the number.
    :return: a string representing the decimal with given precision.
    """
    new_decimal, relevant_precision = __to_relevant_precision(decimal)
    if -precision <= relevant_precision <= precision:
        return f"{decimal:.{precision}f}"
    sign = "-" if relevant_precision < 0 else "+"
    return f"{new_decimal:.{precision}f}e{sign}{abs(relevant_precision)}"


def __to_relevant_precision(decimal: float) -> Tuple[float, int]:
    """
    Get relevant precision of a decimal number.

    :param decimal: a floating point number.
    :return: a tuple of a and n such that: decimal = a * 10^(-b).
    """
    if decimal in [0, np.inf, np.nan, -np.inf]:
        return decimal, 0
    precision = 0
    abs_a = math.fabs(decimal)
    while abs_a < 1.0:
        abs_a *= 10
        precision -= 1
    while abs_a >= 10.0:
        abs_a /= 10
        precision += 1
    if decimal < 0:
        return -abs_a, precision
    return abs_a, precision
