"""Fitting function to evaluate with the fitting algorithm."""
import re
import uuid
from dataclasses import InitVar, dataclass, field
from typing import Callable, Optional

import numpy as np
import scipy
from eddington_core.exceptions import FitFunctionLoadError, FitFunctionRuntimeError
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
    :param x_derivative: a function representing the derivative of fit_func according
     to x
     to the "a" array
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
    costumed: InitVar[bool] = False
    save: InitVar[bool] = True

    def __post_init__(self, costumed, save):
        """Post init methods."""
        self.title_name = self.__get_title_name(costumed=costumed)
        self.__costumed = costumed
        if save:
            FitFunctionsRegistry.add(self)

    def __get_title_name(self, costumed):
        if costumed:
            return "Costumed Function"
        if self.name is None:
            return None
        return self.name.title().replace("_", " ")

    @classmethod
    def anonymous_function(cls, fit_func, n):
        """Creates a function without a name."""
        return FitFunction(fit_func=fit_func, n=n, save=False)

    @classmethod
    def from_string(cls, syntax_string, name=None, save=True):
        """Creates a :class:`FitFunction` from a string."""
        if name is None:
            name = f"dummy-{uuid.uuid4()}"
        n = max([int(a) for a in re.findall(r"a\[(\d+?)\]", syntax_string)]) + 1
        locals_dict = {}
        globals_dict = cls.__get_costumed_globals()
        try:
            exec(  # pylint: disable=exec-used
                f"func = lambda a, x: {syntax_string}", globals_dict, locals_dict
            )
        except SyntaxError:
            raise FitFunctionLoadError(f'"{syntax_string}" has invalid syntax')
        func = locals_dict["func"]
        return FitFunction(
            fit_func=func,
            n=n,
            name=name,
            syntax=syntax_string,
            save=save,
            costumed=True,
        )

    @classmethod
    def __get_costumed_globals(cls):
        globals_dict = globals().copy()
        globals_dict["math"] = np.math
        globals_dict["np"] = np
        globals_dict["numpy"] = np
        globals_dict.update(vars(np))
        globals_dict.update(vars(scipy.special))
        return globals_dict

    def __call__(self, a, x):
        """Call the fit function as a regular callable."""
        a_length = len(a)
        if a_length != self.n:
            raise FitFunctionRuntimeError(
                f"input length should be {self.n}, got {a_length}"
            )
        return self.fit_func(a, x)

    def assign(self, a):
        """Assign the function parameters."""
        return lambda x: self(a, x)

    @classmethod
    def is_generator(cls):
        """Indicates that this is not a :class:`FitFunctionGenerator`."""
        return False

    def is_costumed(self):
        """Indicates if this function is costumed."""
        return self.__costumed

    @property
    def signature(self):
        """Same as name."""
        return self.name

    # Arithmetic Methods

    def __neg__(self):
        """Implementing negative operator."""
        return FitFunction.anonymous_function(lambda a, x: -self.fit_func(a, x), self.n)

    def __add__(self, other):
        """Implementing add operator."""
        if isinstance(other, FitFunction):
            n = max(self.n, other.n)
            return FitFunction.anonymous_function(
                lambda a, x: self.fit_func(a, x) + other.fit_func(a, x), n
            )
        return FitFunction.anonymous_function(lambda a, x: self(a, x) + other, self.n)

    def __radd__(self, other):
        """Implementing reverse add operator."""
        return self + other

    def __sub__(self, other):
        """Implementing subtract operator."""
        return self + (-other)

    def __rsub__(self, other):
        """Implementing reverse subtract operator."""
        return (-self) + other

    def __mul__(self, other):
        """Implementing multiplication operator."""
        if isinstance(other, FitFunction):
            n = max(self.n, other.n)
            return FitFunction.anonymous_function(
                lambda a, x: self.fit_func(a, x) * other.fit_func(a, x), n
            )
        return FitFunction.anonymous_function(
            lambda a, x: other * self.fit_func(a, x), self.n
        )

    def __rmul__(self, other):
        """Implementing reverse multiplication operator."""
        return self * other

    def __truediv__(self, other):
        """Implementing division operator."""
        if isinstance(other, FitFunction):
            n = max(self.n, other.n)
            return FitFunction.anonymous_function(
                lambda a, x: self.fit_func(a, x) / other.fit_func(a, x), n
            )
        return FitFunction.anonymous_function(
            lambda a, x: self.fit_func(a, x) / other, self.n
        )

    def __rtruediv__(self, other):
        """Implementing reverse division operator."""
        return FitFunction.anonymous_function(
            lambda a, x: other / self.fit_func(a, x), self.n
        )

    def __pow__(self, power, modulo=None):
        """Implementing power operator."""
        if isinstance(power, FitFunction):
            n = max(self.n, power.n)
            return FitFunction.anonymous_function(
                lambda a, x: np.power(self.fit_func(a, x), power.fit_func(a, x)), n
            )
        return FitFunction.anonymous_function(
            lambda a, x: np.power(self.fit_func(a, x), power), self.n
        )


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
