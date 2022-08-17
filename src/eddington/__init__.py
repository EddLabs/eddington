"""Core functionalities of the Eddington platform."""
from eddington.exceptions import (
    EddingtonException,
    FittingDataColumnExistenceError,
    FittingDataColumnIndexError,
    FittingDataColumnsLengthError,
    FittingDataError,
    FittingDataInvalidFile,
    FittingDataRecordsSelectionError,
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
from eddington.plot.figure_builder import FigureBuilder
from eddington.plot.plot_legacy import (
    add_errorbar,
    add_grid,
    add_legend,
    add_plot,
    get_figure,
    horizontal_line,
    label_axes,
    plot_data,
    plot_fitting,
    plot_residuals,
    title,
)
from eddington.plot.plot_util import build_repr_string, show_or_export
from eddington.print_util import to_relevant_precision_string
from eddington.random_util import random_data

__version__ = "0.0.24.dev1"

__all__ = [
    "__version__",
    # Fitting functions infrastructure
    "FittingFunction",
    "fitting_function",
    "FittingFunctionsRegistry",
    # Fitting functions
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
    "FittingDataRecordsSelectionError",
    # Data structures
    "FittingData",
    "FittingResult",
    # Plot
    "FigureBuilder",
    "show_or_export",
    "build_repr_string",
    # Plot legacy
    "plot_data",
    "plot_fitting",
    "plot_residuals",
    "get_figure",
    "add_errorbar",
    "add_legend",
    "add_grid",
    "add_plot",
    "horizontal_line",
    "label_axes",
    "title",
    # Print
    "to_relevant_precision_string",
    # Miscellaneous
    "random_data",
]
