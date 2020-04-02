from dataclasses import dataclass, field
import numpy as np
import scipy.stats as stats

from eddington_core.print_util import to_precise_string


@dataclass(repr=False)
class FitResult:

    a0: np.ndarray
    a: np.ndarray
    aerr: np.ndarray
    arerr: np.ndarray = field(init=False)
    acov: np.ndarray
    degrees_of_freedom: int
    chi2: float
    chi2_reduced: float = field(init=False)
    p_probability: float = field(init=False)
    precision: int = 3

    __repr_string: str = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.arerr = np.abs(self.aerr / self.a) * 100
        self.chi2_reduced = self.chi2 / self.degrees_of_freedom
        self.p_probability = stats.chi2.sf(self.chi2, self.degrees_of_freedom)

    def __repr__(self):
        if self.__repr_string is None:
            self.__repr_string = self.build_repr_string()
        return self.__repr_string

    def build_repr_string(self):
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

    def __a_value_string(self, i, a, aerr, arerr):
        a_string = to_precise_string(a, self.precision)
        aerr_string = to_precise_string(aerr, self.precision)
        arerr_string = to_precise_string(arerr, self.precision)
        return f"\ta[{i}] = {a_string} \u00B1 {aerr_string} ({arerr_string}% error)"
