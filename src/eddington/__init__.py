from eddington.fit_functions.fit_functions_list import *
from eddington.fit_functions.fit_function_generators_list import *
from eddington.exceptions import *

from eddington.fit_functions.fit_function import FitFunction, fit_function
from eddington.fit_functions.fit_function_generator import (
    FitFunctionGenerator,
    fit_function_generator,
)
from eddington.fit_functions.fit_functions_registry import FitFunctionsRegistry

from eddington.fit_data import FitData
from eddington.fit_result import FitResult

from eddington.fit_util import fit_to_data

__version__ = "v0.0.1"
