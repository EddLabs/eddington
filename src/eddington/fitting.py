"""Implementation of the fitting algorithm."""
from typing import Any, Dict, Optional

import numpy as np
from scipy.odr import ODR, Model, RealData

from eddington.fitting_data import FittingData
from eddington.fitting_function_class import FittingFunction
from eddington.fitting_result import FittingResult


def fit(  # pylint: disable=invalid-name
    data: FittingData,
    func: FittingFunction,
    a0: np.ndarray = None,
    use_x_derivative: bool = True,
    use_a_derivative: bool = True,
) -> FittingResult:
    """
    Implementation of the fitting algorithm.

    This functions wraps *scipy*'s
    `ODR <https://docs.scipy.org/doc/scipy/reference/odr.html>`_ algorithm.

    :param data: Fitting data to optimize
    :type data: :class:`FittingData`
    :param func: a function to fit the data according to.
    :type func: :class:`FittingFunction`
    :param a0: initial guess for the parameters
    :type a0: ``np.ndarray``
    :param use_x_derivative: indicates whether to use x derivative or not.
    :type use_x_derivative: ``bool``
    :param use_a_derivative: indicates whether to use a derivative or not.
    :type use_a_derivative: ``bool``
    :returns: :class:`FittingResult`
    """
    model = Model(
        **__get_odr_model_kwargs(
            func,
            use_x_derivative=use_x_derivative,
            use_a_derivative=use_a_derivative,
        )
    )
    a0 = __get_a0(n=func.active_parameters, a0=a0)
    real_data = RealData(x=data.x, y=data.y, sx=data.xerr, sy=data.yerr)
    odr = ODR(data=real_data, model=model, beta0=a0)
    output = odr.run()
    a = output.beta  # pylint: disable=invalid-name
    chi2 = output.sum_square  # pylint: disable=no-member
    degrees_of_freedom = len(data.x) - func.active_parameters
    return FittingResult(
        a0=a0,
        a=a,
        aerr=output.sd_beta,
        acov=output.cov_beta,
        degrees_of_freedom=degrees_of_freedom,
        chi2=chi2,
    )


def __get_odr_model_kwargs(
    func: FittingFunction,
    use_x_derivative: bool = True,
    use_a_derivative: bool = True,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = dict(fcn=func)
    if use_a_derivative and func.a_derivative is not None:
        kwargs["fjacb"] = func.a_derivative
    if use_x_derivative and func.x_derivative is not None:
        kwargs["fjacd"] = func.x_derivative
    return kwargs


def __get_a0(  # pylint: disable=invalid-name
    n: int, a0: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Returns initial parameters for fitting algorithm.

    :param n: Number of parameters
    :param a0: Initial parameters value. Optional
    :return: nd.array
    """
    if a0 is not None:
        return a0
    return np.full(shape=n, fill_value=1.0)
