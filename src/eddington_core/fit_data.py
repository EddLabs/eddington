import csv
from collections import OrderedDict

import numpy as np
import xlrd

from eddington_core.consts import (
    DEFAULT_MIN_COEFF,
    DEFAULT_MAX_COEFF,
    DEFAULT_XMIN,
    DEFAULT_XMAX,
    DEFAULT_MEASUREMENTS,
    DEFAULT_XSIGMA,
    DEFAULT_YSIGMA,
)
from eddington_core.exceptions import (
    FitDataColumnExistenceError,
    FitDataColumnIndexError,
    FitDataInvalidFileSyntax,
    FitDataColumnsLengthError,
)
from eddington_core.random_util import random_array, random_sigma, random_error


class FitData:
    def __init__(
        self, data, x_column=None, xerr_column=None, y_column=None, yerr_column=None
    ):
        self._data = OrderedDict(
            [(key, np.array(value)) for key, value in data.items()]
        )
        lengths = set([value.size for value in self.data.values()])
        if len(lengths) != 1:
            raise FitDataColumnsLengthError()
        self._all_columns = list(self.data.keys())
        self.x_column = x_column
        self.xerr_column = xerr_column
        self.y_column = y_column
        self.yerr_column = yerr_column

    # Data is read-only

    @property
    def data(self):
        return self._data

    @property
    def all_columns(self):
        return self._all_columns

    @property
    def x_column(self):
        return self._x_column

    @x_column.setter
    def x_column(self, x_column):
        if x_column is None:
            self._x_column_index = 0
        elif x_column in self.all_columns:
            self._x_column_index = self.all_columns.index(x_column)
        else:
            self._x_column_index = self._covert_to_index(x_column)
        self._validate_index(self._x_column_index, x_column)
        self._x_column = self.all_columns[self._x_column_index]

    @property
    def xerr_column(self):
        return self._xerr_column

    @xerr_column.setter
    def xerr_column(self, xerr_column):
        if xerr_column is None:
            self._xerr_column_index = self._x_column_index + 1
        elif xerr_column in self.all_columns:
            self._xerr_column_index = self.all_columns.index(xerr_column)
        else:
            self._xerr_column_index = self._covert_to_index(xerr_column)
        self._validate_index(self._xerr_column_index, xerr_column)
        self._xerr_column = self.all_columns[self._xerr_column_index]

    @property
    def y_column(self):
        return self._y_column

    @y_column.setter
    def y_column(self, y_column):
        if y_column is None:
            self._y_column_index = self._xerr_column_index + 1
        elif y_column in self.all_columns:
            self._y_column_index = self.all_columns.index(y_column)
        else:
            self._y_column_index = self._covert_to_index(y_column)
        self._validate_index(self._y_column_index, y_column)
        self._y_column = self.all_columns[self._y_column_index]

    @property
    def yerr_column(self):
        return self._yerr_column

    @yerr_column.setter
    def yerr_column(self, yerr_column):
        if yerr_column is None:
            self._yerr_column_index = self._y_column_index + 1
        elif yerr_column in self.all_columns:
            self._yerr_column_index = self.all_columns.index(yerr_column)
        else:
            self._yerr_column_index = self._covert_to_index(yerr_column)
        self._validate_index(self._yerr_column_index, yerr_column)
        self._yerr_column = self.all_columns[self._yerr_column_index]

    @property
    def x(self):
        return self.data[self.x_column]

    @property
    def xerr(self):
        return self.data[self.xerr_column]

    @property
    def y(self):
        return self.data[self.y_column]

    @property
    def yerr(self):
        return self.data[self.yerr_column]

    @classmethod
    def random(
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
        if a is None:
            a = random_array(min_val=min_coeff, max_val=max_coeff, n=fit_func.n)
        x = random_array(min_val=xmin, max_val=xmax, n=measurements)
        xerr = random_sigma(average_sigma=xsigma, n=measurements)
        yerr = random_sigma(average_sigma=ysigma, n=measurements)
        y = fit_func(a, x + random_error(scales=xerr)) + random_error(scales=yerr)
        return FitData(
            data=OrderedDict([("x", x), ("xerr", xerr), ("y", y), ("yerr", yerr)])
        )

    @classmethod
    def read_from_excel(
        cls,
        filepath,
        sheet,
        x_column=None,
        xerr_column=None,
        y_column=None,
        yerr_column=None,
    ):
        excel_obj = xlrd.open_workbook(filepath)
        sheet_obj = excel_obj.sheet_by_name(sheet)
        rows = [sheet_obj.row(i) for i in range(sheet_obj.nrows)]
        rows = [list(map(lambda element: element.value, row)) for row in rows]
        return cls._extract_data_from_rows(
            rows=rows,
            file_name=filepath.name,
            sheet=sheet,
            x_column=x_column,
            xerr_column=xerr_column,
            y_column=y_column,
            yerr_column=yerr_column,
        )

    @classmethod
    def read_from_csv(
        cls, filepath, x_column=None, xerr_column=None, y_column=None, yerr_column=None,
    ):
        with open(filepath, mode="r") as csv_file:
            csv_obj = csv.reader(csv_file)
            rows = list(csv_obj)
        return cls._extract_data_from_rows(
            rows=rows,
            file_name=filepath.name,
            x_column=x_column,
            xerr_column=xerr_column,
            y_column=y_column,
            yerr_column=yerr_column,
        )

    @classmethod
    def _covert_to_index(cls, column):
        try:
            return int(column) - 1
        except ValueError:
            return None

    def _validate_index(self, index, column):
        if index is None:
            raise FitDataColumnExistenceError(column)
        max_index = len(self._all_columns)
        if index < 0 or index >= max_index:
            raise FitDataColumnIndexError(index + 1, max_index)

    @classmethod
    def _extract_data_from_rows(
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
        if cls._is_headers(headers):
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
    def _is_headers(cls, headers):
        return all([header != "" and not cls._is_number(header) for header in headers])

    @classmethod
    def _is_number(cls, string):
        try:
            float(string)
            return True
        except ValueError:
            return False
