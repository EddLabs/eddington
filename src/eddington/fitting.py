"""Implementation of the fitting algorithm."""
import numpy as np
from scipy.odr import ODR, Model, RealData

from eddington import FitData, FitFunction, FitResult


def fit_to_data(
    data: FitData,
    func: FitFunction,
    a0: np.ndarray = None,
    use_x_derivative=True,
    use_a_derivative=True,
):  # pylint: disable=C0103
    """
    Implementation of the fitting algorithm using scipy ODR algorithm.

    :param data: :class:`FitData`. Fitting data to optimize
    :param func: :class:`FitFunction`. a function to fit the data according to.
    :param a0: nd.array. initail guess for the parameters
    :param use_x_derivative: Boolean. indicates whether to use x derviative or not.
    :param use_a_derivative: Boolean. indicates whether to use a derviative or not.
    :return: :class:`FitResult`
    """
    model = Model(
        **__get_odr_model_kwargs(
            func, use_x_derivative=use_x_derivative, use_a_derivative=use_a_derivative,
        )
    )
    a0 = __get_a0(n=func.active_parameters, a0=a0)
    real_data = RealData(x=data.x, y=data.y, sx=data.xerr, sy=data.yerr)
    odr = ODR(data=real_data, model=model, beta0=a0)
    output = odr.run()
    a = output.beta
    chi2 = output.sum_square  # pylint: disable=E1101
    degrees_of_freedom = len(data.x) - func.active_parameters
    return FitResult(
        a0=a0,
        a=a,
        aerr=output.sd_beta,
        acov=output.cov_beta,
        degrees_of_freedom=degrees_of_freedom,
        chi2=chi2,
    )


def __get_odr_model_kwargs(
    func, use_x_derivative=True, use_a_derivative=True,
):
    kwargs = dict(fcn=func)
    if use_a_derivative and func.a_derivative is not None:
        kwargs["fjacb"] = func.a_derivative
    if use_x_derivative and func.x_derivative is not None:
        kwargs["fjacd"] = func.x_derivative
    return kwargs


def __get_a0(n, a0=None):  # pylint: disable=invalid-name
    """
    Returns initial parameters for fitting algorithm.

    :param n: Number of parameters
    :param a0: Initial parameters value. Optional
    :return: nd.array
    """
    if a0 is not None:
        return a0
    return np.full(shape=n, fill_value=1.0)
