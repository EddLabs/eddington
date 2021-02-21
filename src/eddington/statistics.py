"""Module for handling statistical values."""
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from eddington import io_util


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

    @classmethod
    def save_as_csv(
        cls,
        statistics_map: Dict[str, "Statistics"],
        output_directory: Union[str, Path],
        name: Optional[str] = None,
    ):
        """
        Save the fitting data statistics to csv file.

        :param statistics_map: dictionary from a value to statistics object to be saved
        :type statistics_map: Dict[str, "Statistics"]
        :param output_directory:
            Path to the directory for the new excel file to be saved.
        :type output_directory: ``Path`` or ``str``
        :param name: Optional. The name of the file, without the .csv suffix.
            "fitting_data" by default.
        :type name: str
        """
        if name is None:
            name = "fitting_data_statistics"
        io_util.save_as_csv(
            content=cls.__statistics_as_records(statistics_map),
            output_directory=output_directory,
            file_name=name,
        )

    @classmethod
    def save_as_excel(
        cls,
        statistics_map: Dict[str, "Statistics"],
        output_directory: Union[str, Path],
        name: Optional[str] = None,
        sheet: Optional[str] = None,
    ):
        """
        Save the fitting data statistics to xlsx file.

        :param statistics_map: dictionary from a value to statistics object to be saved
        :type statistics_map: Dict[str, "Statistics"]
        :param output_directory: Path to the directory for the new excel file to be
            saved.
        :type output_directory: ``Path`` or ``str``
        :param name: Optional. The name of the file, without the .xlsx suffix.
            "fitting_data_statistics" by default.
        :type name: str
        :param sheet: Optional. Name of the sheet that the data will be saved to.
        :type sheet: str
        """
        if name is None:
            name = "fitting_data_statistics"
        io_util.save_as_excel(
            content=cls.__statistics_as_records(statistics_map),
            output_directory=output_directory,
            file_name=name,
            sheet=sheet,
        )

    @classmethod
    def __statistics_as_records(cls, statistics_map):
        columns = list(statistics_map.keys())
        records = [["Parameters", *columns]]
        for parameter in cls.parameters():
            records.append(
                [
                    parameter.replace("_", " ").title(),
                    *[getattr(statistics_map[column], parameter) for column in columns],
                ]
            )
        return records
