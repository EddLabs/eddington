"""Printing functions for displaying numbers with given precision."""
import numpy as np

from eddington.consts import DEFAULT_MAX_STRING_LENGTH, DEFAULT_PRECISION


def to_relevant_precision_string(
    decimal: float,
    relevant_digits: int = DEFAULT_PRECISION,
    max_string_length: int = DEFAULT_MAX_STRING_LENGTH,
) -> str:
    """
    Convert a decimal into string while preserving relevant digits.

    :param decimal: The float to convert to string.
    :type decimal: float
    :param relevant_digits: number of relevant digits to preserve while converting to
        string.
    :type relevant_digits: int
    :param max_string_length: maximum string length when decimal is not normalized.
    :type max_string_length: int
    :returns: decimal as string
    :rtype: str
    """
    if is_nan_or_infinity(decimal):
        return str(decimal)
    order = order_of_magnitude(decimal)
    digit = order - relevant_digits
    return to_digit_string(decimal, digit, max_string_length)


def to_digit_string(
    decimal: float, digit: int, max_string_length: int = DEFAULT_MAX_STRING_LENGTH
) -> str:
    """
    Convert a decimal into string while preserving given digit.

    :param decimal: The float to convert to string.
    :type decimal: float
    :param digit: The lowest digit to preserve while converting to string.
    :type digit: int
    :param max_string_length: maximum string length when decimal is not normalized.
    :type max_string_length: int
    :returns: decimal as string
    :rtype: str
    """
    if decimal == 0 and digit < 0:
        return f"{0:.{-digit}f}"
    if decimal in [np.inf, np.nan, -np.inf]:
        return str(decimal)
    order = order_of_magnitude(decimal)
    number_of_digits = order - digit
    if number_of_digits < max_string_length and -DEFAULT_MAX_STRING_LENGTH < digit < 0:
        return f"{decimal:.{-digit}f}"
    order_sign = "+" if order > 0 else "-"
    normalized_value = decimal * np.power(10.0, -order)
    return f"{normalized_value:.{number_of_digits}f}e{order_sign}{np.abs(order)}"


def order_of_magnitude(decimal: float) -> int:
    """
    Get the order of magnitude of the given number.

    If 0, nan or infinity is provided, returns 0

    :param decimal: The float to check ordered of magnitude.
    :type decimal: float
    :return: Order of magnitude of the given float
    :rtype: int
    """
    if decimal == 0 or is_nan_or_infinity(decimal):
        return 0
    return int(np.floor(np.log10(np.abs(decimal))))


def is_nan_or_infinity(decimal: float) -> bool:
    """
    Whether a floating number is infinity or NaN (Not a Number).

    :param decimal: float to check
    :type decimal: float
    :return: Is nan or infinity
    :rtype: bool
    """
    return np.isnan(decimal) or np.isinf(decimal)
