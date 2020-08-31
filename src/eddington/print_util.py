"""Printing functions for displaying numbers with given precision."""
import math
from typing import Tuple


def to_relevant_precision(decimal: float) -> Tuple[float, int]:
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


def to_precise_string(decimal: float, precision: int) -> str:
    """
    Returns a decimal as string with desired precision.

    :param decimal: a floating point number.
    :param precision: the desired precision to return the number.
    :return: a string representing the decimal with given precision.
    """
    new_decimal, relevant_precision = to_relevant_precision(decimal)
    if relevant_precision < 3:
        return f"{decimal:.{precision + relevant_precision}f}"
    return f"{new_decimal:.{precision}f}e-0{relevant_precision}"
