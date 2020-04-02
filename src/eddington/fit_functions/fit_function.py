import re
import uuid
from typing import Callable
from dataclasses import dataclass, InitVar, field
import numpy as np

from eddington.exceptions import FitFunctionLoadError
from eddington.fit_functions.fit_functions_registry import FitFunctionsRegistry


@dataclass(repr=False, unsafe_hash=True)
class FitFunction:
    fit_func: Callable
    n: int
    name: str
    syntax: str
    a_derivative: np.ndarray = field(default=None)
    x_derivative: np.ndarray = field(default=None)
    title_name: str = field(init=False)
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
        return self.name.title().replace("_", " ")

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
        return globals_dict

    def __call__(self, a, x):
        a_length = len(a)
        if a_length != self.n:
            raise ValueError(f"input length should be {self.n}, got {a_length}")
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

    def __repr__(self):
        return f"FitFunction(name={self.name})"


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
