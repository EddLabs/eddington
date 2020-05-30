"""Core functionalities of the Eddington platform."""
from eddington_core.exceptions import (
    EddingtonException,
    FitDataColumnExistenceError,
    FitDataColumnIndexError,
    FitDataColumnsLengthError,
    FitDataError,
    FitDataInvalidFile,
    FitDataInvalidFileSyntax,
    FitFunctionLoadError,
    FitFunctionRuntimeError,
)
from eddington_core.fit_data import FitData
from eddington_core.fit_function_class import FitFunction, fit_function
from eddington_core.fit_functions_registry import FitFunctionsRegistry
from eddington_core.fit_result import FitResult
from eddington_core.fit_functions_list import (
    constant,
    exponential,
    hyperbolic,
    linear,
    parabolic,
    cos,
    sin,
    straight_power,
    inverse_power,
)
from eddington_core.fitting import fit_to_data

__all__ = [
    # Fit functions infrastructure
    "FitFunction",
    "fit_function",
    "FitFunctionsRegistry",
    # Fit functions
    "constant",
    "exponential",
    "hyperbolic",
    "linear",
    "parabolic",
    "cos",
    "sin",
    "straight_power",
    "inverse_power",
    # Fitting algorithm
    "fit_to_data",
    # Exceptions
    "EddingtonException",
    "FitFunctionRuntimeError",
    "FitFunctionLoadError",
    "FitDataError",
    "FitDataColumnExistenceError",
    "FitDataColumnIndexError",
    "FitDataInvalidFile",
    "FitDataColumnsLengthError",
    "FitDataInvalidFileSyntax",
    # Data structures
    "FitData",
    "FitResult",
]
