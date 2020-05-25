"""Fitting function to evaluate with the fitting algorithm."""
from dataclasses import InitVar, dataclass, field
from typing import Callable, Optional

import numpy as np

from eddington_core.exceptions import FitFunctionRuntimeError
from eddington_core.fit_functions_registry import FitFunctionsRegistry


@dataclass(unsafe_hash=True)
class FitFunction:  # pylint: disable=invalid-name,too-many-instance-attributes
    """
    Fitting function class.

    :param fit_func: Callable. The actual fitting function.
     The function gets 2 parameters:
     a - an array with the parameters of the function.
     x - the sample data to be fit.
    :param n: Number of parameters. the length of "a" in fit_func.
    :param name: The name of the function.
    :param syntax: The syntax of the fitting function
    :param a_derivative: a function representing the derivative of fit_func according
     to the "a" array
    :param x_derivative: a function representing the derivative of fit_func according
     to x
    :param title_name: same as "name" but in title case
    :param costumed: Is this fit functioned made from a string.
     This will be deprecated soon.
    :param save: Boolean. Should this function be saved in the
     :class:`FitFunctionsRegistry`
    """

    fit_func: Callable = field(repr=False)
    n: int = field(repr=False)
    name: Optional[str] = field(default=None)
    syntax: Optional[str] = field(default=None)
    a_derivative: np.ndarray = field(default=None, repr=False)
    x_derivative: np.ndarray = field(default=None, repr=False)
    title_name: str = field(init=False, repr=False)
    save: InitVar[bool] = True

    def __post_init__(self, save):
        """Post init methods."""
        self.title_name = self.__get_title_name()
        if save:
            FitFunctionsRegistry.add(self)

    def __get_title_name(self):
        if self.name is None:
            return None
        return self.name.title().replace("_", " ")

    def __validate_parameters_number(self, a):
        a_length = len(a)
        if a_length != self.n:
            raise FitFunctionRuntimeError(
                f"input length should be {self.n}, got {a_length}"
            )

    def __call__(self, a, x):
        """Call the fit function as a regular callable."""
        self.__validate_parameters_number(a)
        return self.fit_func(a, x)

    def assign(self, a):
        """Assign the function parameters."""
        self.__validate_parameters_number(a)
        return lambda x: self(a, x)

    @classmethod
    def is_generator(cls):
        """Indicates that this is not a :class:`FitFunctionGenerator`."""
        return False

    @property
    def signature(self):
        """Same as name."""
        return self.name


def fit_function(  # pylint: disable=invalid-name,too-many-arguments
    n, name=None, syntax=None, a_derivative=None, x_derivative=None, save=True
):
    """
    Wrapper making a simple callable into a :class:`FitFunction`.

    :param n: Number of parameters. the length of "a" in fit_func.
    :param name: The name of the function.
    :param syntax: The syntax of the fitting function
    :param a_derivative: a function representing the derivative of fit_func according
     to the "a" array
    :param x_derivative: a function representing the derivative of fit_func according
     to x
    :param save: Boolean. Should this function be saved in the
     :class:`FitFunctionsRegistry`
    :return: :class:`FitFunction` instance.
    """

    def wrapper(func):
        func_name = func.__name__ if name is None else name
        return FitFunction(
            fit_func=func,
            n=n,
            name=func_name,
            syntax=syntax,
            a_derivative=a_derivative,
            x_derivative=x_derivative,
            save=save,
        )

    return wrapper
