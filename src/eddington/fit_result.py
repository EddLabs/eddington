"""Fitting result class that will be returned by the fitting algorithm."""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.stats as stats

from eddington.print_util import to_precise_string


@dataclass(repr=False)  # pylint: disable=too-many-instance-attributes
class FitResult:
    """
    Dataclass that contains the relevant parameters returned by a fitting algorithm.

    :param a0: The initial guess for the fit function parameters.
    :param a: The result for the fitting parameters.
    :param aerr: Estimated errors of a.
    :param arerr: Estimated relative errors of a (equivilant to aerr/a).
    :param acov: Covarance matrix of a.
    :param degrees_of_freedom: How many degrees of freedom of the fittings.
    :param chi2: Optimization evaluation for the fit.
    :param chi2_reduced: Reduced chi2.
    :param p_probability: P-probability (p-value) of the fitting, evaluated from
     chi2_reduced.
    """

    a0: np.ndarray  # pylint: disable=invalid-name
    a: np.ndarray  # pylint: disable=invalid-name
    aerr: np.ndarray
    arerr: np.ndarray = field(init=False)
    acov: np.ndarray
    degrees_of_freedom: int
    chi2: float
    chi2_reduced: float = field(init=False)
    p_probability: float = field(init=False)
    precision: int = field(default=3)
    __pretty_string: Optional[str] = field(default=None, init=False)

    def __post_init__(self):
        """Post init methods."""
        self.aerr = np.array(self.aerr)
        self.acov = np.array(self.acov)
        self.a0 = np.array(self.a0)
        self.a = np.array(self.a)
        with np.errstate(divide="ignore"):
            self.arerr = np.abs(self.aerr / self.a) * 100
        self.chi2_reduced = self.chi2 / self.degrees_of_freedom
        self.p_probability = stats.chi2.sf(self.chi2, self.degrees_of_freedom)

    def print_or_export(self, file_path=None):
        """
        Write the result to a file or print it to console.

        :param file_path: str ot None. Path to write the result in. if None, prints
         to console.
        """
        if file_path is None:
            print(self.pretty_string)
            return
        with open(file_path, mode="w") as output_file:
            output_file.write(self.pretty_string)

    @property
    def pretty_string(self):
        """Pretty representation string."""
        if self.__pretty_string is None:
            self.__pretty_string = self.__build_pretty_string()
        return self.__pretty_string

    def __repr__(self):
        """Representation string."""
        return self.pretty_string

    def __build_pretty_string(self):
        old_precision = np.get_printoptions()["precision"]
        np.set_printoptions(precision=self.precision)
        a_value_string = "\n".join(
            [
                self.__a_value_string(i, a, aerr, arerr)
                for i, (a, aerr, arerr) in enumerate(zip(self.a, self.aerr, self.arerr))
            ]
        )
        repr_string = f"""Results:
========

Initial parameters' values:
\t{" ".join(str(i) for i in self.a0)}
Fitted parameters' values:
{a_value_string}
Fitted parameters covariance:
{self.acov}
Chi squared: {to_precise_string(self.chi2, self.precision)}
Degrees of freedom: {self.degrees_of_freedom}
Chi squared reduced: {to_precise_string(self.chi2_reduced, self.precision)}
P-probability: {to_precise_string(self.p_probability, self.precision)}
"""
        np.set_printoptions(precision=old_precision)
        return repr_string

    def __a_value_string(self, i, a, aerr, arerr):  # pylint: disable=invalid-name
        a_string = to_precise_string(a, self.precision)
        aerr_string = to_precise_string(aerr, self.precision)
        arerr_string = to_precise_string(arerr, self.precision)
        return f"\ta[{i}] = {a_string} \u00B1 {aerr_string} ({arerr_string}% error)"
