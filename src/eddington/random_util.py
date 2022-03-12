"""Functions for data randomization."""
from collections import OrderedDict
from typing import Optional

import numpy as np

from eddington.consts import (
    DEFAULT_MAX_COEFF,
    DEFAULT_MEASUREMENTS,
    DEFAULT_MIN_COEFF,
    DEFAULT_XMAX,
    DEFAULT_XMIN,
    DEFAULT_XSIGMA,
    DEFAULT_YSIGMA,
)
from eddington.fitting_data import FittingData


def random_data(  # pylint: disable=too-many-arguments,too-many-locals
    fit_func,  # type: ignore
    x: Optional[np.ndarray] = None,
    a: Optional[np.ndarray] = None,
    xmin: float = DEFAULT_XMIN,
    xmax: float = DEFAULT_XMAX,
    min_coeff: float = DEFAULT_MIN_COEFF,
    max_coeff: float = DEFAULT_MAX_COEFF,
    xsigma: float = DEFAULT_XSIGMA,
    ysigma: float = DEFAULT_YSIGMA,
    measurements: Optional[int] = None,
    x_column: str = "x",
    y_column: str = "y",
    xerr_column: Optional[str] = "xerr",
    yerr_column: Optional[str] = "yerr",
) -> FittingData:
    """
    Generate a random fit data.

    :param fit_func: :class:`FittingFunction` to evaluate with the fit data
    :type fit_func: ``FittingFunction``
    :param x: Optional. The input for the fitting algorithm.
        If not given, generated randomly.
    :type x: ``numpy.ndarray``
    :param a: Optional. the actual parameters that should be returned by the
        fitting algorithm. If not given, generated randomly.
    :type a: ``numpy.ndarray``
    :param xmin: Minimum value for x.
    :type xmin: float
    :param xmax: Maximum value for x.
    :type xmax: float
    :param min_coeff: Minimum value for `a` coefficient.
    :type min_coeff: float
    :param max_coeff: Maximum value for `a` coefficient.
    :type max_coeff: float
    :param xsigma: Standard deviation for x.
    :type xsigma: float
    :param ysigma: Standard deviation for y.
    :type ysigma: float
    :param measurements: Optional. Number of measurements. If :paramref:`x` is
        given, take as length of x
    :type measurements: int
    :param x_column: Column name for the x values
    :type x_column: str
    :param y_column: Column name for the y values
    :type y_column: str
    :param xerr_column: Column name for the x error values. If None, do not generate
        x errors
    :type xerr_column: str or None
    :param yerr_column: Column name for the y error values. If None, do not generate
        y errors
    :type yerr_column: str or None
    :returns: a random fitting data, according to parameters
    :rtype: FittingData
    """
    if a is None:
        a = random_array(min_val=min_coeff, max_val=max_coeff, size=fit_func.n)
    if x is None:
        if measurements is None:
            measurements = DEFAULT_MEASUREMENTS
        x = random_array(min_val=xmin, max_val=xmax, size=measurements)
    else:
        measurements = x.shape[0]
    raw_data = OrderedDict()
    actual_xerr, xerr = __generate_errors(
        column_name=xerr_column, sigma=xsigma, measurements=measurements
    )
    actual_yerr, yerr = __generate_errors(
        column_name=yerr_column, sigma=ysigma, measurements=measurements
    )
    y = fit_func(a, x + actual_xerr) + actual_yerr
    raw_data[x_column] = x
    if xerr_column is not None:
        raw_data[xerr_column] = xerr
    raw_data[y_column] = y
    if yerr_column is not None:
        raw_data[yerr_column] = yerr

    return FittingData(
        data=raw_data,
        x_column=x_column,
        xerr_column=xerr_column,
        y_column=y_column,
        yerr_column=yerr_column,
        search=False,
    )


def random_array(min_val: float, max_val: float, size: int) -> np.ndarray:
    """
    Creates a random array.

    :param min_val: Minimum value for array item.
    :param max_val: Maximum value for array item.
    :param size: Size of the array.
    :return: array
    """
    return np.random.uniform(min_val, max_val, size=size)


def random_sigma(average_sigma: float, size: int) -> np.ndarray:
    """
    Creates random standard deviation (aka. sigma).

    :param average_sigma: average value for sigma.
    :param size: Size of the returned array.
    :return: array of sigma values.
    """
    return np.random.exponential(average_sigma, size=size)


def random_error(scales: np.ndarray) -> np.ndarray:
    """
    Generates random errors based on an array of scales.

    :param scales: array of standard deviations.
    :return: errors array.
    """
    return np.random.normal(scale=scales)


def __generate_errors(column_name, sigma, measurements):
    if column_name is not None:
        average_error = random_sigma(average_sigma=sigma, size=measurements)
        actual_xerr = random_error(scales=average_error)
        return actual_xerr, average_error
    actual_xerr = np.zeros(shape=measurements)
    return actual_xerr, 0
