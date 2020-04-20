from collections import OrderedDict

import numpy as np

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
    ColumnExistenceError,
    ColumnIndexError,
)
from eddington_core.random_util import random_array, random_sigma, random_error


class FitData:
    def __init__(
        self, data, x_column=None, xerr_column=None, y_column=None, yerr_column=None
    ):
        self._data = OrderedDict(
            [(key, np.array(value)) for key, value in data.items()]
        )
        self._data_keys = list(self.data.keys())
        self.x_column = x_column
        self.xerr_column = xerr_column
        self.y_column = y_column
        self.yerr_column = yerr_column

    # Data is read-only

    @property
    def data(self):
        return self._data

    @property
    def x_column(self):
        return self._x_column

    @x_column.setter
    def x_column(self, x_column):
        if x_column is None:
            self._x_column_index = 0
        elif x_column in self._data_keys:
            self._x_column_index = list(self._data_keys).index(x_column)
        else:
            self._x_column_index = self._covert_to_index(x_column)
        self._validate_index(self._x_column_index, x_column)
        self._x_column = self._data_keys[self._x_column_index]

    @property
    def xerr_column(self):
        return self._xerr_column

    @xerr_column.setter
    def xerr_column(self, xerr_column):
        if xerr_column is None:
            self._xerr_column_index = self._x_column_index + 1
        elif xerr_column in self._data_keys:
            self._xerr_column_index = list(self._data_keys).index(xerr_column)
        else:
            self._xerr_column_index = self._covert_to_index(xerr_column)
        self._validate_index(self._xerr_column_index, xerr_column)
        self._xerr_column = self._data_keys[self._xerr_column_index]

    @property
    def y_column(self):
        return self._y_column

    @y_column.setter
    def y_column(self, y_column):
        if y_column is None:
            self._y_column_index = self._xerr_column_index + 1
        elif y_column in self._data_keys:
            self._y_column_index = list(self._data_keys).index(y_column)
        else:
            self._y_column_index = self._covert_to_index(y_column)
        self._validate_index(self._y_column_index, y_column)
        self._y_column = self._data_keys[self._y_column_index]

    @property
    def yerr_column(self):
        return self._yerr_column

    @yerr_column.setter
    def yerr_column(self, yerr_column):
        if yerr_column is None:
            self._yerr_column_index = self._y_column_index + 1
        elif yerr_column in self._data_keys:
            self._yerr_column_index = list(self._data_keys).index(yerr_column)
        else:
            self._yerr_column_index = self._covert_to_index(yerr_column)
        self._validate_index(self._yerr_column_index, yerr_column)
        self._yerr_column = self._data_keys[self._yerr_column_index]

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
    def _covert_to_index(cls, column):
        try:
            return int(column) - 1
        except ValueError:
            return None

    def _validate_index(self, index, column):
        if index is None:
            raise ColumnExistenceError(column)
        max_index = len(self._data_keys)
        if index < 0 or index >= max_index:
            raise ColumnIndexError(index + 1, max_index)

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
