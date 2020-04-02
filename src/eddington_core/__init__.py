from eddington_core.fit_functions.fit_functions_list import *
from eddington_core.fit_functions.fit_function_generators_list import *
from eddington_core.exceptions import *

from eddington_core.fit_functions.fit_function import FitFunction, fit_function
from eddington_core.fit_functions.fit_function_generator import (
    FitFunctionGenerator,
    fit_function_generator,
)
from eddington_core.fit_functions.fit_functions_registry import FitFunctionsRegistry

from eddington_core.fit_data import FitData
from eddington_core.fit_result import FitResult

from eddington_core.fit_util import fit_to_data

__version__ = "v0.0.1"
