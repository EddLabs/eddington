from dataclasses import dataclass
import numpy as np

from eddington_core.consts import (
    DEFAULT_MIN_COEFF,
    DEFAULT_MAX_COEFF,
    DEFAULT_XMIN,
    DEFAULT_XMAX,
    DEFAULT_MEASUREMENTS,
    DEFAULT_XSIGMA,
    DEFAULT_YSIGMA,
)
from eddington_core.random_util import random_array, random_sigma, random_error


@dataclass
class FitData:

    x: np.ndarray
    xerr: np.ndarray
    y: np.ndarray
    yerr: np.ndarray

    @classmethod
    def random(
        cls,
        fit_func,
        a=None,
        xmin=DEFAULT_XMIN,
        xmax=DEFAULT_XMAX,
        min_coeff=DEFAULT_MIN_COEFF,
        max_coeff=DEFAULT_MAX_COEFF,
        xsigma=DEFAULT_XSIGMA,
        ysigma=DEFAULT_YSIGMA,
        measurements=DEFAULT_MEASUREMENTS,
    ):
        if a is None:
            a = random_array(min_val=min_coeff, max_val=max_coeff, n=fit_func.n)
        x = random_array(min_val=xmin, max_val=xmax, n=measurements)
        xerr = random_sigma(average_sigma=xsigma, n=measurements)
        yerr = random_sigma(average_sigma=ysigma, n=measurements)
        y = fit_func(a, x + random_error(scales=xerr)) + random_error(scales=yerr)
        return FitData(x=x, xerr=xerr, y=y, yerr=yerr)
