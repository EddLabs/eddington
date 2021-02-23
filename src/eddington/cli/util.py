"""Utility methods for the Eddington CLI."""
import re
from pathlib import Path
from typing import Optional

import numpy as np

from eddington import FittingDataInvalidFile, FittingFunction
from eddington.exceptions import EddingtonCLIError
from eddington.fitting_data import FittingData
from eddington.fitting_functions_list import linear, polynomial
from eddington.fitting_functions_registry import FittingFunctionsRegistry


def load_data_file(data_file: Path, **kwargs) -> FittingData:
    """
    Load data file with any suffix.

    :param data_file: The type of the file to be loaded.
    :type data_file: Path
    :param kwargs: Keyword arguments for the actual reading method
    :type kwargs: dict
    :return: FittingData
    :raises FittingDataInvalidFile: Given an unknown file suffix, raise exception.
    """
    suffix = data_file.suffix
    if suffix == ".xlsx":
        return FittingData.read_from_excel(filepath=data_file, **kwargs)
    if "sheet" in kwargs:
        del kwargs["sheet"]
    if suffix == ".csv":
        return FittingData.read_from_csv(filepath=data_file, **kwargs)
    if suffix == ".json":
        return FittingData.read_from_json(filepath=data_file, **kwargs)
    raise FittingDataInvalidFile(f"Cannot read fitting data from a {suffix} file.")


def load_fitting_function(
    func_name: Optional[str], polynomial_degree: Optional[int]
) -> FittingFunction:
    """
    Load fitting function.

    If function name is given, get from registry.
    If polynomial degree is given, get polynomial with given degree.
    If none are given, returns linear.

    :param func_name: Function name to get from registry
    :type func_name: Optional[str]
    :param polynomial_degree: Degree of the polynomial.
    :type polynomial_degree: Optional[int]
    :return: Matching fitting function.
    :rtype: FittingFunction
    :raises EddingtonCLIError: If both function name and polynomial degree are given
        raise an exception.

    """
    if func_name is None or func_name.strip() == "":
        if polynomial_degree is not None:
            return polynomial(polynomial_degree)
        return linear
    if polynomial_degree is not None:
        raise EddingtonCLIError("Cannot accept both polynomial and fitting function")
    return FittingFunctionsRegistry.load(func_name)


def extract_array_from_string(  # pylint: disable=invalid-name
    a0: Optional[str],
) -> Optional[np.ndarray]:
    """
    Split a0 string to an array.

    :param a0: Initial guess values separated by commas.
    :type a0: Optional[str]
    :return: Parsed a0 or None.
    :rtype: Optional[np.ndarray]
    """
    if a0 is None:
        return None
    return np.array(list(map(float, re.split(",[ \t]*", a0))))
