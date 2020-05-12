"""Generator creating fitting functions based on parameters."""
from dataclasses import InitVar, dataclass, field
from typing import Callable, List, Union

from eddington_core.fit_functions_registry import FitFunctionsRegistry


@dataclass(repr=False, unsafe_hash=True)
class FitFunctionGenerator:
    """
    Fit function generator creates a :class:`FitFunction` based on input parameters.

    :param generator_func: Callable. The actual generator function
    :param name: The name of the generator.
    :param syntax: The syntax of the generator.
    :param parameters: The parameters the generator gets in order to create the fit
     function
    :param signature: The signature of the genrator. based o the name and parameters of
     the generator.
    :param save: Boolean. Indicates whether to save the generator in the
     :class:`FitFunctionsRegistry` or not.
    """

    generator_func: Callable
    name: str
    syntax: str
    parameters: Union[List[str], str]
    signature: str = field(init=False)
    save: InitVar[bool] = True

    def __post_init__(self, save):
        """Post init methods."""
        self.signature = f"{self.name}({self.__param_string()})"
        if save:
            FitFunctionsRegistry.add(self)

    def __call__(self, *args, **kwargs):
        """Call the generator to create fit function."""
        return self.generator_func(*args, **kwargs)

    @classmethod
    def is_generator(cls):
        """Inidicates this is a :class:`FitFunctionGenerator`."""
        return True

    def __param_string(self):
        if isinstance(self.parameters, str):
            return self.parameters
        return ", ".join(self.parameters)


def fit_function_generator(parameters, name=None, syntax=None, save=True):
    """
    A wrapper making a callable into a :class:`FitFunctionGenerator`.

    :param name: The name of the generator.
    :param syntax: The syntax of the generator.
    :param parameters: The parameters the generator gets in order to create the fit
     function
    :param save: Boolean. Indicates whether to save the generator in the
     :class:`FitFunctionsRegistry` or not.
    """

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
