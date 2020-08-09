"""Fitting data class insert the fitting algorithm."""
import csv
from collections import OrderedDict, namedtuple

import numpy as np
import xlrd

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
)
from eddington.random_util import random_array, random_error, random_sigma


Columns = namedtuple("ColumnsResult", ["x", "y", "xerr", "yerr"])


class FitData:  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """Fitting data class."""

    def __init__(  # pylint: disable=too-many-arguments
        self, data, x_column=None, xerr_column=None, y_column=None, yerr_column=None,
    ):
        """
        Constructor.

        :param data: Numpy array which its rows are the available data records.
        :param x_column: int or string. Indicates the which column should be used as the
         x parameter
        :param xerr_column: int or string. Indicates the which column should be used as
         the x error parameter
        :param y_column: int or string. Indicates the which column should be used as the
         x parameter
        :param yerr_column: int or string. Indicates the which column should be used as
         the x error parameter
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

    def select_record(self, record_index):
        """Select a record to be used in fitting."""
        self.records_indices[record_index - 1] = True

    def unselect_record(self, record_index):
        """Unselect a record to be used in fitting."""
        self.records_indices[record_index - 1] = False

    def select_all_records(self):
        """Select all records to be used in fitting."""
        self.records_indices = [True] * self.length

    def unselect_all_records(self):
        """Unselect all recrods from being used in fitting."""
        self.records_indices = [False] * self.length

    def is_selected(self, records_index):
        """Checks if a record is selected or not."""
        return self.records_indices[records_index - 1]

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
        fit_func,
        a=None,
        xmin=DEFAULT_XMIN,
        xmax=DEFAULT_XMAX,
        min_coeff=DEFAULT_MIN_COEFF,
        max_coeff=DEFAULT_MAX_COEFF,
        xsigma=DEFAULT_XSIGMA,
        ysigma=DEFAULT_YSIGMA,
        measurements=DEFAULT_MEASUREMENTS,
    ):
        """
        Generate a random fit data.

        :param fit_func: Fit function to evaluate with the fit data
        :param a: Optional. the actual parameters that should be returned by the fitting
         algorithm. If not given, generated randomly.
        :param xmin: Minimum value for x.
        :param xmax: Maximum value for x.
        :param min_coeff: Minimum value for :ref:`a` coefficient.
        :param max_coeff: Maximum value for :ref:`a` coefficient.
        :param xsigma: Standard deviation for x.
        :param ysigma: Standard deviation for y.
        :param measurements: Number of measurements
        :return: random :class:`FitData` object
        """
        if a is None:
            a = random_array(min_val=min_coeff, max_val=max_coeff, size=fit_func.n)
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
        filepath,
        sheet,
        x_column=None,
        xerr_column=None,
        y_column=None,
        yerr_column=None,
    ):
        """
        Read :class:`FitData` from excel file.

        :param filepath: str or Path. Path to location of excel file
        :param sheet: str. The name of the seet to exctract the data from.
        :param x_column: int or str. Column for the x values.
        :param xerr_column: int or str. Column for the x error values.
        :param y_column: int or str. Column for the y values.
        :param yerr_column: int or str. Column for the y error values.
        :return: :class:`FitData` read from the excel file.
        """
        excel_obj = xlrd.open_workbook(filepath)
        sheet_obj = excel_obj.sheet_by_name(sheet)
        rows = [sheet_obj.row(i) for i in range(sheet_obj.nrows)]
        rows = [list(map(lambda element: element.value, row)) for row in rows]
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
        cls, filepath, x_column=None, xerr_column=None, y_column=None, yerr_column=None,
    ):
        """
        Read :class:`FitData` from csv file.

        :param filepath: str or Path. Path to location of csv file
        :param x_column: int or str. Column for the x values.
        :param xerr_column: int or str. Column for the x error values.
        :param y_column: int or str. Column for the y values.
        :param yerr_column: int or str. Column for the y error values.
        :return: :class:`FitData` read from the csv file.
        """
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
        headers = rows[0]
        if cls.__is_headers(headers):
            content = rows[1:]
        else:
            headers = range(len(headers))
            content = rows
        try:
            content = [list(map(float, row)) for row in content]
        except ValueError:
            raise FitDataInvalidFileSyntax(file_name, sheet=sheet)
        columns = zip(*content)
        return FitData(
            OrderedDict(zip(headers, columns)),
            x_column=x_column,
            xerr_column=xerr_column,
            y_column=y_column,
            yerr_column=yerr_column,
        )

    @classmethod
    def __is_headers(cls, headers):
        return all([header != "" and not cls.__is_number(header) for header in headers])

    @classmethod
    def __is_number(cls, string):
        try:
            float(string)
            return True
        except ValueError:
            return False
