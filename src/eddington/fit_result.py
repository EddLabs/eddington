"""Fitting result class that will be returned by the fitting algorithm."""
from dataclasses import dataclass, field
from typing import List, Union

import numpy as np
import scipy.stats as stats


@dataclass  # pylint: disable=too-many-instance-attributes
class FitResult:
    """
    Dataclass that contains the relevant parameters returned by a fitting algorithm.

    :param a0: The initial guess for the fit function parameters.
    :param a: The result for the fitting parameters.
    :param aerr: Estimated errors of a.
    :param arerr: Estimated relative errors of a (equivalent to aerr/a).
    :param acov: Covariance matrix of a.
    :param degrees_of_freedom: How many degrees of freedom of the fittings.
    :param chi2: Optimization evaluation for the fit.
    :param chi2_reduced: Reduced chi2.
    :param p_probability: P-probability (p-value) of the fitting, evaluated from
     chi2_reduced.
    """

    a: Union[List[float], np.ndarray]  # pylint: disable=invalid-name
    aerr: Union[List[float], np.ndarray]
    arerr: Union[List[float], np.ndarray] = field(init=False)
    acov: Union[List[List[float]], np.ndarray]
    degrees_of_freedom: int
    chi2: float
    chi2_reduced: float = field(init=False)
    p_probability: float = field(init=False)
    precision: int = field(default=3, repr=False)

    def __post_init__(self) -> None:
        """Post init methods."""
        self.a = np.array(self.a)
        self.aerr = np.array(self.aerr)
        self.acov = np.array(self.acov)
        with np.errstate(divide="ignore"):
            self.arerr = np.abs(self.aerr / self.a) * 100
        self.chi2_reduced = self.chi2 / self.degrees_of_freedom
        self.p_probability = stats.chi2.sf(self.chi2, self.degrees_of_freedom)
