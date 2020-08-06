"""Fitting function to evaluate with the fitting algorithm."""
import functools
from dataclasses import InitVar, dataclass, field
from typing import Callable, Optional, Dict

import numpy as np

from eddington.exceptions import FitFunctionRuntimeError
from eddington.fit_functions_registry import FitFunctionsRegistry


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
    name: Optional[str] = field()
    syntax: Optional[str] = field(default=None)
    a_derivative: Optional[Callable] = field(default=None, repr=False)
    x_derivative: Optional[Callable] = field(default=None, repr=False)
    title_name: str = field(init=False, repr=False)
    fixed: Dict[int, float] = field(init=False, repr=False)
    save: InitVar[bool] = True

    def __post_init__(self, save):
        """Post init methods."""
        self.title_name = self.__get_title_name()
        self.fixed = dict()
        self.x_derivative = self.__wrap_x_derivative(self.x_derivative)
        self.a_derivative = self.__wrap_a_derivative(self.a_derivative)
        if save:
            FitFunctionsRegistry.add(self)

    def __get_title_name(self):
        return self.name.title().replace("_", " ")

    def __validate_parameters_number(self, a):
        a_length = len(a)
        if a_length != self.n:
            raise FitFunctionRuntimeError(
                f"Input length should be {self.active_parameters}, "
                f"got {a_length - len(self.fixed)}"
            )

    def __call__(self, *args):
        """Call the fit function as a regular callable."""
        a, x = self.__extract_a_and_x(args)
        self.__validate_parameters_number(a)
        return self.fit_func(a, x)

    def assign(self, a):
        """Assign the function parameters."""
        a = self.__add_fixed_values(a)
        self.__validate_parameters_number(a)
        self.fixed = dict(enumerate(a))
        return self

    def fix(self, index, value):
        """
        Fix parameter with predefined value.

        :param index: The index of the parameter to fix. Starting from 0
        :param value: The value to fix
        :return: self
        """
        if index < 0 or index >= self.n:
            raise FitFunctionRuntimeError(
                f"Cannot fix index {index}. "
                f"Indices should be between 0 and {self.n - 1}"
            )
        self.fixed[index] = value
        return self

    def unfix(self, index):
        """
        Unfix a fixed parameter.

        :param index: The index of the parameter to unfix
        :return: self
        """
        del self.fixed[index]
        return self

    def clear_fixed(self):
        """
        Clear all fixed parameters.

        :return: self
        """
        self.fixed.clear()

    @property
    def signature(self):
        """Same as name."""
        return self.name

    @property
    def active_parameters(self):
        """Number of active parameters (aka, unfixed)."""
        return self.n - len(self.fixed)

    def __wrap_x_derivative(self, method):
        if method is None:
            return None

        @functools.wraps(method)
        def wrapper(*args):
            a, x = self.__extract_a_and_x(args)
            self.__validate_parameters_number(a)
            return method(a, x)

        return wrapper

    def __wrap_a_derivative(self, method):
        if method is None:
            return None

        @functools.wraps(method)
        def wrapper(*args):
            a, x = self.__extract_a_and_x(args)
            self.__validate_parameters_number(a)
            result = method(a, x)
            if len(self.fixed) == 0:
                return result
            return np.delete(result, list(self.fixed.keys()), axis=0)

        return wrapper

    def __extract_a_and_x(self, args):
        if len(args) == 0:
            raise FitFunctionRuntimeError(
                f'No parameters has been given to "{self.name}"'
            )
        if len(args) == 1:
            a = [self.fixed[i] for i in sorted(self.fixed.keys())]
            x = args[0]
        else:
            a = self.__add_fixed_values(args[0])
            x = args[1]
        return a, x

    def __add_fixed_values(self, a):
        for i in sorted(self.fixed.keys()):
            a = np.insert(a, i, self.fixed[i])
        return a


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
        return functools.wraps(func)(
            FitFunction(
                fit_func=func,
                n=n,
                name=func_name,
                syntax=syntax,
                a_derivative=a_derivative,
                x_derivative=x_derivative,
                save=save,
            )
        )

    return wrapper
