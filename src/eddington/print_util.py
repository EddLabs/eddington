"""Printing functions for displaying numbers with given precision."""
import numpy as np

from eddington.consts import DEFAULT_PRECISION, DEFAULT_MAX_STRING_LENGTH


def to_relevant_precision_string(
        decimal: float,
        relevant_digits: int = DEFAULT_PRECISION,
        max_string_length: int = DEFAULT_MAX_STRING_LENGTH
) -> str:
    if decimal in [np.inf, np.nan, -np.inf]:
        return str(decimal)
    order = order_of_magnitude(decimal)
    digit = order - relevant_digits
    return to_digit_string(decimal, digit, max_string_length)


def to_digit_string(
        decimal: float,
        digit: int,
        max_string_length: int = DEFAULT_MAX_STRING_LENGTH
) -> str:
    if decimal == 0 and digit < 0:
        return f"{0:.{-digit}f}"
    if decimal in [np.inf, np.nan, -np.inf]:
        return str(decimal)
    order = order_of_magnitude(decimal)
    number_of_digits = order - digit
    if number_of_digits < max_string_length and -DEFAULT_MAX_STRING_LENGTH < digit < 0:
        return f"{decimal:.{-digit}f}"
    order_sign = "+" if order > 0 else "-"
    return f"{(decimal * np.power(10.0, -order)):.{number_of_digits}f}e{order_sign}{np.abs(order)}"


def order_of_magnitude(decimal: float) -> int:
    if decimal in [0, np.inf, np.nan, -np.inf]:
        return 0
    return int(np.floor(np.log10(np.abs(decimal))))
