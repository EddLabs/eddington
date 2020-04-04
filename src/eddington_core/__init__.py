from eddington_core.fit_functions.fit_functions_list import (
    constant,
    linear,
    parabolic,
    hyperbolic,
    exponential,
)

from eddington_core.fit_functions.fit_function_generators_list import (
    straight_power,
    polynom,
    inverse_power,
)
from eddington_core.fit_functions.fit_function import FitFunction, fit_function
from eddington_core.fit_functions.fit_function_generator import (
    FitFunctionGenerator,
    fit_function_generator,
)
from eddington_core.fit_functions.fit_functions_registry import FitFunctionsRegistry

from eddington_core.exceptions import (
    FitFunctionLoadError,
    EddingtonException,
    InvalidGeneratorInitialization,
)

from eddington_core.fit_data import FitData
from eddington_core.fit_result import FitResult
from eddington_core.fit_util import fit_to_data


def get_fit_functions(module):
    return [
        key
        for key, value in vars(module).items()
        if isinstance(value, FitFunction) or isinstance(value, FitFunctionGenerator)
    ]


__all__ = [
    # Fit functions infrastructure
    "FitFunction",
    "fit_function",
    "FitFunctionGenerator",
    "fit_function_generator",
    "FitFunctionsRegistry",
    # Fit functions
    "constant",
    "linear",
    "parabolic",
    "hyperbolic",
    "exponential",
    # Fit functions generators
    "straight_power",
    "polynom",
    "inverse_power",
    # Exceptions
    "FitFunctionLoadError",
    "EddingtonException",
    "InvalidGeneratorInitialization",
    # Data structures
    "FitData",
    "FitResult",
    # Fitting functions
    "fit_to_data",
]
