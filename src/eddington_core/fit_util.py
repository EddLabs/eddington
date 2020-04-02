from scipy.odr import ODR, RealData, Model
import numpy as np

from eddington_core.fit_functions.fit_function import FitFunction
from eddington_core.fit_data import FitData
from eddington_core.fit_result import FitResult


def fit_to_data(data: FitData, func: FitFunction, a0: np.ndarray):
    model = Model(**__get_odr_model_kwargs(func))
    real_data = RealData(x=data.x, y=data.y, sx=data.xerr, sy=data.yerr)
    odr = ODR(data=real_data, model=model, beta0=a0)
    output = odr.run()
    a = output.beta
    chi2 = output.sum_square
    degrees_of_freedom = len(data.x) - func.n
    return FitResult(
        a0=a0,
        a=a,
        aerr=output.sd_beta,
        acov=output.cov_beta,
        degrees_of_freedom=degrees_of_freedom,
        chi2=chi2,
    )


def __get_odr_model_kwargs(func):
    kwargs = dict(fcn=func)
    if func.a_derivative is not None:
        kwargs["fjacb"] = func.a_derivative
    if func.x_derivative is not None:
        kwargs["fjacd"] = func.x_derivative
    return kwargs
