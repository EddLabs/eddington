"""Fitting data class insert the fitting algorithm."""
import csv
import json
from collections import OrderedDict, namedtuple
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import openpyxl

from eddington.consts import (
    DEFAULT_MAX_COEFF,
    DEFAULT_MEASUREMENTS,
    DEFAULT_MIN_COEFF,
    DEFAULT_XMAX,
    DEFAULT_XMIN,
    DEFAULT_XSIGMA,
    DEFAULT_YSIGMA,
)
from eddington.exceptions import (
    FitDataColumnExistenceError,
    FitDataColumnIndexError,
    FitDataColumnsLengthError,
    FitDataColumnsSelectionError,
    FitDataInvalidFileSyntax,
    FitDataSetError,
)
from eddington.random_util import random_array, random_error, random_sigma

Columns = namedtuple("ColumnsResult", ["x", "y", "xerr", "yerr"])


class FitData:  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """Fitting data class."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        data: Union[OrderedDict, Dict[str, np.ndarray]],
        x_column: Optional[Union[str, int]] = None,
        xerr_column: Optional[Union[str, int]] = None,
        y_column: Optional[Union[str, int]] = None,
        yerr_column: Optional[Union[str, int]] = None,
    ):
        """
        Constructor.

        :param data: Dictionary from a column name to its values
        :type data: ``dict`` or ``OrderedDict`` from ``str`` to ``numpy.ndarray``
        :param x_column: Indicates which column should be used as the x parameter
        :type x_column: ``str`` or ``numpy.ndarray``
        :param xerr_column: Indicates which column should be used as the x error
         parameter
        :type xerr_column: ``str`` or ``numpy.ndarray``
        :param y_column: Indicates which column should be used as the y parameter
        :type y_column: ``str`` or ``numpy.ndarray``
        :param yerr_column: Indicates which column should be used as the y error
         parameter
        :type yerr_column: ``str`` or ``numpy.ndarray``
        """
        self._data = OrderedDict(
            [(key, np.array(value)) for key, value in data.items()]
        )
        lengths = {value.size for value in self.data.values()}
        if len(lengths) != 1:
            raise FitDataColumnsLengthError()
        self._length = next(iter(lengths))
        self.select_all_records()
        self._all_columns = list(self.data.keys())
        self.x_column = x_column
        self.xerr_column = xerr_column
        self.y_column = y_column
        self.yerr_column = yerr_column

    # Data properties are read-only

    @property
    def length(self):
        """Number of records."""
        return self._length

    @property
    def data(self):
        """Data matrix."""
        return self._data

    @property
    def all_columns(self):
        """Columns list."""
        return self._all_columns

    @property
    def used_columns(self):
        """Dictionary of columns in use."""
        return Columns(
            x=self.x_column,
            xerr=self.xerr_column,
            y=self.y_column,
            yerr=self.yerr_column,
        )

    @property
    def x(self):  # pylint: disable=invalid-name
        """X values."""
        return self.data[self.x_column][self.records_indices]

    @property
    def xerr(self):
        """X error values."""
        return self.data[self.xerr_column][self.records_indices]

    @property
    def y(self):  # pylint: disable=invalid-name
        """Y values."""
        return self.data[self.y_column][self.records_indices]

    @property
    def yerr(self):
        """Y error values."""
        return self.data[self.yerr_column][self.records_indices]

    # Records indices methods

    def select_record(self, index: int):
        """
        Select a record to be used in fitting.

        :param index: index of the desired record **starting from 1**.
        :type index: int
        """
        self.records_indices[index - 1] = True

    def unselect_record(self, index: int):
        """
        Unselect a record to be used in fitting.

        :param index: index of the desired record **starting from 1**.
        :type index: int
        """
        self.records_indices[index - 1] = False

    def select_all_records(self):
        """Select all records to be used in fitting."""
        self.records_indices = [True] * self.length

    def unselect_all_records(self):
        """Unselect all records from being used in fitting."""
        self.records_indices = [False] * self.length

    def is_selected(self, index):
        """
        Checks if a record is selected or not.

        :param index: index of the desired record **starting from 1**.
        :type index: int
        :returns: ``bool``
        """
        return self.records_indices[index - 1]

    @property
    def records_indices(self):
        """List of booleans indicating which records are selected."""
        return self._records_indices

    @records_indices.setter
    def records_indices(self, records_indices):
        if len(records_indices) != self.length:
            raise FitDataColumnsSelectionError(
                f"Should select {self.length} records,"
                f" only {len(records_indices)} selected."
            )
        if not all([isinstance(element, bool) for element in records_indices]):
            raise FitDataColumnsSelectionError(
                "When setting record indices, all values should be booleans."
            )
        self._records_indices = records_indices

    # Columns can be set

    @property
    def x_column(self):
        """Name of the x column."""
        return self._x_column

    @x_column.setter
    def x_column(self, x_column):
        if x_column is None:
            self._x_column_index = 0
        elif x_column in self.all_columns:
            self._x_column_index = self.all_columns.index(x_column)
        else:
            self._x_column_index = self.__covert_to_index(x_column)
        self.__validate_index(self._x_column_index, x_column)
        self._x_column = self.all_columns[self._x_column_index]

    @property
    def xerr_column(self):
        """Name of the x error column."""
        return self._xerr_column

    @xerr_column.setter
    def xerr_column(self, xerr_column):
        if xerr_column is None:
            self._xerr_column_index = self._x_column_index + 1
        elif xerr_column in self.all_columns:
            self._xerr_column_index = self.all_columns.index(xerr_column)
        else:
            self._xerr_column_index = self.__covert_to_index(xerr_column)
        self.__validate_index(self._xerr_column_index, xerr_column)
        self._xerr_column = self.all_columns[self._xerr_column_index]

    @property
    def y_column(self):
        """Name of the y column."""
        return self._y_column

    @y_column.setter
    def y_column(self, y_column):
        if y_column is None:
            self._y_column_index = self._xerr_column_index + 1
        elif y_column in self.all_columns:
            self._y_column_index = self.all_columns.index(y_column)
        else:
            self._y_column_index = self.__covert_to_index(y_column)
        self.__validate_index(self._y_column_index, y_column)
        self._y_column = self.all_columns[self._y_column_index]

    @property
    def yerr_column(self):
        """Name of the y error column."""
        return self._yerr_column

    @yerr_column.setter
    def yerr_column(self, yerr_column):
        if yerr_column is None:
            self._yerr_column_index = self._y_column_index + 1
        elif yerr_column in self.all_columns:
            self._yerr_column_index = self.all_columns.index(yerr_column)
        else:
            self._yerr_column_index = self.__covert_to_index(yerr_column)
        self.__validate_index(self._yerr_column_index, yerr_column)
        self._yerr_column = self.all_columns[self._yerr_column_index]

    @classmethod
    def random(  # pylint: disable=invalid-name,too-many-arguments
        cls,
        fit_func,  # type: ignore
        x: Optional[np.ndarray] = None,
        a: Optional[np.ndarray] = None,
        xmin: float = DEFAULT_XMIN,
        xmax: float = DEFAULT_XMAX,
        min_coeff: float = DEFAULT_MIN_COEFF,
        max_coeff: float = DEFAULT_MAX_COEFF,
        xsigma: float = DEFAULT_XSIGMA,
        ysigma: float = DEFAULT_YSIGMA,
        measurements: int = DEFAULT_MEASUREMENTS,
    ):
        """
        Generate a random fit data.

        :param fit_func: :class:`FitFunction` to evaluate with the fit data
        :type fit_func: ``FitFunction``
        :param x: Optional. The input for the fitting algorithm.
         If not given, generated randomly.
        :type x: ``numpy.ndarray``
        :param a: Optional. the actual parameters that should be returned by the
         fitting algorithm. If not given, generated randomly.
        :type a: ``numpy.ndarray``
        :param xmin: Minimum value for x.
        :type xmin: float
        :param xmax: Maximum value for x.
        :type xmax: float
        :param min_coeff: Minimum value for `a` coefficient.
        :type min_coeff: float
        :param max_coeff: Maximum value for `a` coefficient.
        :type max_coeff: float
        :param xsigma: Standard deviation for x.
        :type xsigma: int
        :param ysigma: Standard deviation for y.
        :type ysigma: int
        :param measurements: Number of measurements
        :type measurements: int
        :returns: random :class:`FitData`
        """
        if a is None:
            a = random_array(min_val=min_coeff, max_val=max_coeff, size=fit_func.n)
        if x is None:
            x = random_array(min_val=xmin, max_val=xmax, size=measurements)
        xerr = random_sigma(average_sigma=xsigma, size=measurements)
        yerr = random_sigma(average_sigma=ysigma, size=measurements)
        y = fit_func(a, x + random_error(scales=xerr)) + random_error(scales=yerr)
        return FitData(
            data=OrderedDict([("x", x), ("xerr", xerr), ("y", y), ("yerr", yerr)])
        )

    @classmethod
    def read_from_excel(  # pylint: disable=too-many-arguments
        cls,
        filepath: Union[str, Path],
        sheet: str,
        x_column: Optional[Union[str, int]] = None,
        xerr_column: Optional[Union[str, int]] = None,
        y_column: Optional[Union[str, int]] = None,
        yerr_column: Optional[Union[str, int]] = None,
    ):
        """
        Read :class:`FitData` from excel file.

        :param filepath: str or Path. Path to location of excel file
        :param sheet: str. The name of the sheet to extract the data from.
        :param x_column: Indicates which column should be used as the
         x parameter
        :type x_column: ``str`` or ``numpy.ndarray``
        :param xerr_column: Indicates which column should be used as the x error
         parameter
        :type xerr_column: ``str`` or ``numpy.ndarray``
        :param y_column: Indicates which column should be used as the x parameter
        :type y_column: ``str`` or ``numpy.ndarray``
        :param yerr_column: Indicates which column should be used as the y error
         parameter
        :type xerr_column: ``str`` or ``numpy.ndarray``
        :returns: :class:`FitData` read from the excel file.
        """
        if isinstance(filepath, str):
            filepath = Path(filepath)

        workbook = openpyxl.load_workbook(filepath, data_only=True)
        rows = [list(row) for row in workbook[sheet].values]

        return cls.__extract_data_from_rows(
            rows=rows,
            file_name=filepath.name,
            sheet=sheet,
            x_column=x_column,
            xerr_column=xerr_column,
            y_column=y_column,
            yerr_column=yerr_column,
        )

    @classmethod
    def read_from_csv(  # pylint: disable=too-many-arguments
        cls,
        filepath: Union[str, Path],
        x_column: Optional[Union[str, int]] = None,
        xerr_column: Optional[Union[str, int]] = None,
        y_column: Optional[Union[str, int]] = None,
        yerr_column: Optional[Union[str, int]] = None,
    ):
        """
        Read :class:`FitData` from csv file.

        :param filepath: str or Path. Path to location of csv file
        :param x_column: Indicates which column should be used as the x parameter
        :type x_column: ``str`` or ``numpy.ndarray``
        :param xerr_column: Indicates which column should be used as the x error
         parameter
        :type xerr_column: ``str`` or ``numpy.ndarray``
        :param y_column: Indicates which column should be used as the x parameter
        :type y_column: ``str`` or ``numpy.ndarray``
        :param yerr_column: Indicates which column should be used as the y error
         parameter
        :type xerr_column: ``str`` or ``numpy.ndarray``
        :returns: :class:`FitData` read from the csv file.
        """
        if isinstance(filepath, str):
            filepath = Path(filepath)
        with open(filepath, mode="r") as csv_file:
            csv_obj = csv.reader(csv_file)
            rows = list(csv_obj)
        return cls.__extract_data_from_rows(
            rows=rows,
            file_name=filepath.name,
            x_column=x_column,
            xerr_column=xerr_column,
            y_column=y_column,
            yerr_column=yerr_column,
        )

    @classmethod
    def read_from_json(  # pylint: disable=too-many-arguments
        cls,
        filepath: Union[str, Path],
        x_column: Optional[Union[str, int]] = None,
        xerr_column: Optional[Union[str, int]] = None,
        y_column: Optional[Union[str, int]] = None,
        yerr_column: Optional[Union[str, int]] = None,
    ):
        """
        Read :class:`FitData` from json file.

        :param filepath: str or Path. Path to location of csv file
        :param x_column: Indicates which column should be used as the x parameter
        :type x_column: ``str`` or ``numpy.ndarray``
        :param xerr_column: Indicates which column should be used as the x error
         parameter
        :type xerr_column: ``str`` or ``numpy.ndarray``
        :param y_column: Indicates which column should be used as the x parameter
        :type y_column: ``str`` or ``numpy.ndarray``
        :param yerr_column: Indicates which column should be used as the y error
         parameter
        :type xerr_column: ``str`` or ``numpy.ndarray``
        :returns: :class:`FitData` read from the json file.
        """
        if isinstance(filepath, str):
            filepath = Path(filepath)
        with open(filepath, mode="r") as file:
            data = json.load(file, object_pairs_hook=OrderedDict)
        try:
            return FitData(
                OrderedDict(
                    [(key, list(map(float, row))) for key, row in data.items()]
                ),
                x_column=x_column,
                xerr_column=xerr_column,
                y_column=y_column,
                yerr_column=yerr_column,
            )
        except (ValueError, TypeError) as error:
            raise FitDataInvalidFileSyntax(filepath) from error

    def save_excel(
        self,
        output_directory: Union[str, Path],
        name: Optional[str] = "fit_data",
        sheet: Optional[str] = None,
    ):
        """
        Save :class:`FitData` to xlsx file.

        :param output_directory: Path to the directory for the new excel file to be
         saved.
        :type output_directory: ``Path`` or ``str``
        :param name: Optional. The name of the file, without the .xlsx suffix.
         "fit_data" by default.
        :type name: str
        :param sheet: Optional. Name of the sheet that the data will be saved to.
        :type sheet: str
        :returns: :class:`FitData` read from the excel file.
        """
        workbook = openpyxl.Workbook()
        worksheet = workbook.active

        if sheet:
            worksheet.title = sheet

        headers = list(self.data.keys())
        columns = list(self.data.values())

        worksheet.append(headers)

        for row in zip(*columns):
            worksheet.append(row)

        path = Path(output_directory / Path(f"{name}.xlsx"))

        workbook.save(path)

    def save_csv(
        self, output_directory: Union[str, Path], name: Optional[str] = "fit_data"
    ):
        """
        Save :class:`FitData` to csv file.

        :param output_directory:
         Path to the directory for the new excel file to be saved.
        :type output_directory: ``Path`` or ``str``
        :param name: Optional. The name of the file, without the .csv suffix.
         "fit_data" by default.
        :type name: str
        """
        headers = list(self.data.keys())
        columns = list(self.data.values())

        path = Path(output_directory / Path(f"{name}.csv"))

        with open(path, mode="w+", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(headers)
            writer.writerows(zip(*columns))

    def set_header(self, old, new):
        """
        Rename header.

        :param old: The old columns name
        :type old: str
        :param new: The new value to set for the header
        :type new: str
        """
        if new == old:
            return
        if new == "":
            raise FitDataSetError("Cannot set new header to be empty")
        if new in self.all_columns:
            raise FitDataSetError(f'The column name:"{new}" is already used.')
        self._data[new] = self._data.pop(old)
        self._all_columns = list(self.data.keys())

    def set_cell(self, record_number, column_name, value):
        """
        Set new value to a cell.

        :param record_number: The number of the record to set, starting from 1
        :type record_number: int
        :param column_name: The column name
        :type column_name: str
        :param value: The new value to set for the cell
        :type value: float
        """
        if not self.__is_number(value):
            raise FitDataSetError(
                f'The cell at record number:"{record_number}", '
                f'column:"{column_name}" has invalid syntax: {value}.'
            )
        try:
            self._data[column_name][record_number - 1] = value
        except KeyError as error:
            raise FitDataSetError(
                f'Column name "{column_name}" does not exists'
            ) from error
        except IndexError as error:
            raise FitDataSetError(
                f"Record number {record_number} does not exists"
            ) from error

    @classmethod
    def __covert_to_index(cls, column):
        try:
            return int(column) - 1
        except ValueError:
            return None

    def __validate_index(self, index, column):
        if index is None:
            raise FitDataColumnExistenceError(column)
        max_index = len(self._all_columns)
        if index < 0 or index >= max_index:
            raise FitDataColumnIndexError(index + 1, max_index)

    @classmethod
    def __extract_data_from_rows(  # pylint: disable=too-many-arguments
        cls,
        rows,
        file_name,
        sheet=None,
        x_column=None,
        xerr_column=None,
        y_column=None,
        yerr_column=None,
    ):
        headers: List[str] = rows[0]
        if cls.__is_headers(headers):
            content = rows[1:]
        else:
            headers = [str(i) for i in range(len(headers))]
            content = rows
        try:
            content = [list(map(float, row)) for row in content]
        except (ValueError, TypeError) as error:
            raise FitDataInvalidFileSyntax(file_name, sheet=sheet) from error
        columns = [np.array(column) for column in zip(*content)]
        return FitData(
            OrderedDict(zip(headers, columns)),
            x_column=x_column,
            xerr_column=xerr_column,
            y_column=y_column,
            yerr_column=yerr_column,
        )

    @classmethod
    def __is_headers(cls, headers):
        return all([cls.__is_header(header) for header in headers])

    @classmethod
    def __is_header(cls, string):
        if not isinstance(string, str):
            return False
        if string == "":
            return False
        return not cls.__is_number(string)

    @classmethod
    def __is_number(cls, string):
        try:
            float(string)
            return True
        except ValueError:
            return False
