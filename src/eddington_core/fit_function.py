import re
import uuid
from typing import Callable
from dataclasses import dataclass, InitVar, field
import numpy as np
import scipy

from eddington_core.exceptions import FitFunctionLoadError, FitFunctionRuntimeError
from eddington_core.fit_functions_registry import FitFunctionsRegistry


@dataclass(unsafe_hash=True)
class FitFunction:
    fit_func: Callable = field(repr=False)
    n: int = field(repr=False)
    name: str = field(default=None)
    syntax: str = field(default=None)
    a_derivative: np.ndarray = field(default=None, repr=False)
    x_derivative: np.ndarray = field(default=None, repr=False)
    title_name: str = field(init=False, repr=False)
    costumed: InitVar[bool] = False
    save: InitVar[bool] = True

    def __post_init__(self, costumed, save):
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
        return FitFunction(fit_func=fit_func, n=n, save=False)

    @classmethod
    def from_string(cls, syntax_string, name=None, save=True):
        if name is None:
            name = f"dummy-{uuid.uuid4()}"
        n = max([int(a) for a in re.findall(r"a\[(\d+?)\]", syntax_string)]) + 1
        locals_dict = {}
        globals_dict = cls.__get_costumed_globals()
        try:
            exec(f"func = lambda a, x: {syntax_string}", globals_dict, locals_dict)
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
        a_length = len(a)
        if a_length != self.n:
            raise FitFunctionRuntimeError(
                f"input length should be {self.n}, got {a_length}"
            )
        return self.fit_func(a, x)

    def assign(self, a):
        return lambda x: self(a, x)

    @classmethod
    def is_generator(cls):
        return False

    def is_costumed(self):
        return self.__costumed

    @property
    def signature(self):
        return self.name

    # Arithmetic Methods

    def __neg__(self):
        return FitFunction.anonymous_function(lambda a, x: -self(a, x), self.n)

    def __add__(self, other):
        if isinstance(other, FitFunction):
            n = max(self.n, other.n)
            return FitFunction.anonymous_function(
                lambda a, x: self.fit_func(a, x) + other.fit_func(a, x), n
            )
        return FitFunction.anonymous_function(lambda a, x: self(a, x) + other, self.n)

    def __radd__(self, other):
        return self + other


def fit_function(
    n, name=None, syntax=None, a_derivative=None, x_derivative=None, save=True
):
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
