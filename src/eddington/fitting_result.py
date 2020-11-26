"""Fitting result class that will be returned by the fitting algorithm."""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import scipy.stats as stats

from eddington.consts import DEFAULT_PRECISION
from eddington.print_util import (
    order_of_magnitude,
    to_digit_string,
    to_relevant_precision_string,
)


@dataclass(repr=False)
class FittingResult:  # pylint: disable=too-many-instance-attributes
    """
    Dataclass that contains the relevant parameters returned by a fitting algorithm.

    :param a0: The initial guess for the fitting function parameters.
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

    a0: Union[List[float], np.ndarray]  # pylint: disable=invalid-name
    a: Union[List[float], np.ndarray]  # pylint: disable=invalid-name
    aerr: Union[List[float], np.ndarray]
    arerr: Union[List[float], np.ndarray] = field(init=False)
    acov: Union[List[List[float]], np.ndarray]
    degrees_of_freedom: int
    chi2: float
    chi2_reduced: float = field(init=False)
    p_probability: float = field(init=False)
    precision: int = field(default=DEFAULT_PRECISION)
    __pretty_string: Optional[str] = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Post init methods."""
        self.aerr = np.array(self.aerr)
        self.acov = np.array(self.acov)
        self.a0 = np.array(self.a0)
        self.a = np.array(self.a)
        with np.errstate(divide="ignore"):
            self.arerr = np.abs(self.aerr / self.a) * 100
        self.chi2_reduced = self.chi2 / self.degrees_of_freedom
        self.p_probability = stats.chi2.sf(self.chi2, self.degrees_of_freedom)

    def save_txt(self, file_path: Union[str, Path]) -> None:
        """
        Write the result to a text file.

        :param file_path: Path to write the result in. if None, prints to
         console.
        :type file_path: ``str`` or ``Path``
        """
        with open(file_path, mode="w") as output_file:
            output_file.write(self.pretty_string)

    def save_json(self, file_path: Union[str, Path]) -> None:
        """
        Write the result to a json file.

        :param file_path: Path to write the result in. if None, prints to
         console.
        :type file_path: ``str`` or ``Path``
        """
        with open(file_path, mode="w") as output_file:
            json.dump(
                dict(
                    a0=self.a0.tolist(),  # type: ignore
                    a=self.a.tolist(),  # type: ignore
                    aerr=self.aerr.tolist(),  # type: ignore
                    arerr=self.arerr.tolist(),  # type: ignore
                    acov=self.acov.tolist(),  # type: ignore
                    degrees_of_freedom=self.degrees_of_freedom,
                    chi2=self.chi2,
                    chi2_reduced=self.chi2_reduced,
                    p_probability=self.p_probability,
                ),
                output_file,
                indent=1,
            )

    @property
    def pretty_string(self) -> str:
        """Pretty representation string."""
        if self.__pretty_string is None:
            self.__pretty_string = self.__build_pretty_string()
        return self.__pretty_string

    def __repr__(self) -> str:
        """Representation string."""
        return self.pretty_string

    def __build_pretty_string(self) -> str:
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
Chi squared: {to_relevant_precision_string(self.chi2, self.precision)}
Degrees of freedom: {self.degrees_of_freedom}
Chi squared reduced: {to_relevant_precision_string(self.chi2_reduced, self.precision)}
P-probability: {to_relevant_precision_string(self.p_probability, self.precision)}
"""
        np.set_printoptions(precision=old_precision)
        return repr_string

    def __a_value_string(  # pylint: disable=invalid-name
        self, i: int, a: float, aerr: float, arerr: float
    ) -> str:
        order = min(order_of_magnitude(a), order_of_magnitude(aerr))
        digit = order - self.precision
        a_string = to_digit_string(a, digit)
        aerr_string = to_digit_string(aerr, digit)
        arerr_string = to_relevant_precision_string(arerr, self.precision)
        return f"\ta[{i}] = {a_string} \u00B1 {aerr_string} ({arerr_string}% error)"
