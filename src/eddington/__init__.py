"""Core functionalities of the Eddington platform."""
from eddington.exceptions import (
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
from eddington.fit_data import FitData
from eddington.fit_function_class import FitFunction, fit_function
from eddington.fit_functions_list import (
    constant,
    cos,
    exponential,
    hyperbolic,
    inverse_power,
    linear,
    normal,
    parabolic,
    poisson,
    polynomial,
    sin,
    straight_power,
)
from eddington.fit_functions_registry import FitFunctionsRegistry
from eddington.fit_result import FitResult
from eddington.fitting import fit_to_data
from eddington.plot import plot_data, plot_fitting, plot_residuals, show_or_export

__version__ = "0.0.16.dev0"

__all__ = [
    "__version__",
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
    "polynomial",
    "cos",
    "sin",
    "straight_power",
    "inverse_power",
    "normal",
    "poisson",
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
    # Plot
    "plot_data",
    "plot_fitting",
    "plot_residuals",
    "show_or_export",
]
