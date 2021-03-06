"""Module for handling statistical values."""
from dataclasses import dataclass, fields
from typing import List, Union

import numpy as np


@dataclass
class Statistics:
    """Common statistics fields for data analysis."""

    mean: float
    median: float
    variance: float
    standard_deviation: float
    maximum_value: float
    minimum_value: float

    @classmethod
    def from_array(cls, values_array: Union[List[float], np.ndarray]) -> "Statistics":
        """
        Build statistics object for a given values array.

        :param values_array: Values to calculate statistics for.
        :type values_array: numpy.ndarray or list
        :return: Statistics of the given array
        :rtype: Statistics
        :raises ValueError: Cannot calculate statistics of empty data
        """
        if not isinstance(values_array, np.ndarray):
            values_array = np.array(values_array)
        if values_array.shape[0] == 0:
            raise ValueError("Cannot calculate statistics of no values.")
        return Statistics(
            mean=np.average(values_array),
            median=float(np.median(values_array)),
            variance=float(np.var(values_array)),
            standard_deviation=float(np.std(values_array)),
            maximum_value=np.max(values_array),
            minimum_value=np.min(values_array),
        )

    @classmethod
    def parameters(cls):
        """
        Get list of available statistics parameters.

        :return: List of the field names of the statistics
        :rtype: List[str]
        """
        return [field.name for field in fields(cls)]
