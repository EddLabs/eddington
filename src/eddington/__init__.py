"""Core functionalities of the Eddington platform."""
from eddington.exceptions import (
    EddingtonException,
    FittingDataColumnExistenceError,
    FittingDataColumnIndexError,
    FittingDataColumnsLengthError,
    FittingDataError,
    FittingDataInvalidFile,
    FittingDataInvalidFileSyntax,
    FittingFunctionLoadError,
    FittingFunctionRuntimeError,
)
from eddington.fitting import fit
from eddington.fitting_data import FittingData
from eddington.fitting_function_class import FittingFunction, fitting_function
from eddington.fitting_functions_list import (
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
from eddington.fitting_functions_registry import FittingFunctionsRegistry
from eddington.fitting_result import FittingResult
from eddington.plot import plot_data, plot_fitting, plot_residuals, show_or_export

__version__ = "0.0.16"

__all__ = [
    "__version__",
    # Fit functions infrastructure
    "FittingFunction",
    "fitting_function",
    "FittingFunctionsRegistry",
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
    "fit",
    # Exceptions
    "EddingtonException",
    "FittingFunctionRuntimeError",
    "FittingFunctionLoadError",
    "FittingDataError",
    "FittingDataColumnExistenceError",
    "FittingDataColumnIndexError",
    "FittingDataInvalidFile",
    "FittingDataColumnsLengthError",
    "FittingDataInvalidFileSyntax",
    # Data structures
    "FittingData",
    "FittingResult",
    # Plot
    "plot_data",
    "plot_fitting",
    "plot_residuals",
    "show_or_export",
]
