from typing import Callable, Union, List
from dataclasses import dataclass, field, InitVar

from eddington_core.fit_functions.fit_functions_registry import FitFunctionsRegistry


@dataclass(repr=False, unsafe_hash=True)
class FitFunctionGenerator:
    generator_func: Callable
    name: str
    syntax: str
    parameters: Union[List[str], str]
    signature: str = field(init=False)
    save: InitVar[bool] = True

    def __post_init__(self, save):
        self.signature = f"{self.name}({self.__param_string()})"
        if save:
            FitFunctionsRegistry.add(self)

    def __call__(self, *args, **kwargs):
        return self.generator_func(*args, **kwargs)

    @classmethod
    def is_generator(cls):
        return True

    def __param_string(self):
        if isinstance(self.parameters, str):
            return self.parameters
        return ", ".join(self.parameters)


def fit_function_generator(parameters, name=None, syntax=None, save=True):
    def wrapper(generator):
        generator_name = generator.__name__ if name is None else name
        return FitFunctionGenerator(
            generator_func=generator,
            name=generator_name,
            syntax=syntax,
            parameters=parameters,
            save=save,
        )

    return wrapper
