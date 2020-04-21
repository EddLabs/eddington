from unittest import TestCase

import numpy as np

from eddington_core import FitData, ColumnIndexError, ColumnExistenceError
from tests.fit_data import COLUMNS, COLUMNS_NAMES


class DataColumnsBaseTestCase:
    fit_data: FitData

    def test_x(self):
        np.testing.assert_equal(
            COLUMNS[self.x], self.fit_data.x, err_msg="X is different than expected",
        )

    def test_x_err(self):
        np.testing.assert_equal(
            COLUMNS[self.xerr],
            self.fit_data.xerr,
            err_msg="X error is different than expected",
        )

    def test_y(self):
        np.testing.assert_equal(
            COLUMNS[self.y], self.fit_data.y, err_msg="Y is different than expected",
        )

    def test_y_err(self):
        np.testing.assert_equal(
            COLUMNS[self.yerr],
            self.fit_data.yerr,
            err_msg="Y error is different than expected",
        )

    def test_all_columns(self):
        self.assertEqual(
            COLUMNS_NAMES,
            self.fit_data.all_columns,
            msg="Columns are different than expected",
        )

    def test_data(self):
        self.assertEqual(
            COLUMNS_NAMES,
            list(self.fit_data.data.keys()),
            msg="Data keys are different than expected",
        )
        for key, item in self.fit_data.data.items():
            np.testing.assert_equal(
                item,
                COLUMNS[key],
                err_msg=f"Value of {key} is different than expected.",
            )


class TestDataColumnsWithoutArgs(TestCase, DataColumnsBaseTestCase):

    x = "a"
    xerr = "b"
    y = "c"
    yerr = "d"
    fit_data = FitData(COLUMNS)


class TestDataColumnsWithIntX(TestCase, DataColumnsBaseTestCase):

    x = "c"
    xerr = "d"
    y = "e"
    yerr = "f"
    fit_data = FitData(COLUMNS, x_column=3)


class TestDataColumnsWithStringX(TestCase, DataColumnsBaseTestCase):

    x = "c"
    xerr = "d"
    y = "e"
    yerr = "f"
    fit_data = FitData(COLUMNS, x_column="c")


class TestDataColumnsWithIntY(TestCase, DataColumnsBaseTestCase):

    x = "a"
    xerr = "b"
    y = "e"
    yerr = "f"
    fit_data = FitData(COLUMNS, y_column=5)


class TestDataColumnsWithStringY(TestCase, DataColumnsBaseTestCase):

    x = "a"
    xerr = "b"
    y = "e"
    yerr = "f"
    fit_data = FitData(COLUMNS, y_column="e")


class TestDataColumnsWithIntXerr(TestCase, DataColumnsBaseTestCase):

    x = "a"
    xerr = "d"
    y = "e"
    yerr = "f"
    fit_data = FitData(COLUMNS, xerr_column=4)


class TestDataColumnsWithStringXerr(TestCase, DataColumnsBaseTestCase):

    x = "a"
    xerr = "d"
    y = "e"
    yerr = "f"
    fit_data = FitData(COLUMNS, xerr_column="d")


class TestDataColumnsWithIntYerr(TestCase, DataColumnsBaseTestCase):

    x = "a"
    xerr = "b"
    y = "c"
    yerr = "f"
    fit_data = FitData(COLUMNS, yerr_column=6)


class TestDataColumnsWithStringYerr(TestCase, DataColumnsBaseTestCase):

    x = "a"
    xerr = "b"
    y = "c"
    yerr = "f"
    fit_data = FitData(COLUMNS, yerr_column="f")


class TestDataColumnsWithXAndY(TestCase, DataColumnsBaseTestCase):

    x = "c"
    xerr = "d"
    y = "h"
    yerr = "i"
    fit_data = FitData(COLUMNS, x_column=3, y_column="h")


class TestDataColumnsWithJumbledColumns(TestCase, DataColumnsBaseTestCase):

    x = "c"
    xerr = "a"
    y = "b"
    yerr = "i"
    fit_data = FitData(COLUMNS, x_column=3, xerr_column=1, y_column="b", yerr_column=9,)


class DataColumnsRaiseExceptionBaseTestCase:
    def check(self):
        self.assertRaisesRegex(
            self.exception_class, self.error_message, FitData, COLUMNS, **self.kwargs
        )

    def test_raise_exception_on_x_being_not_existing(self):
        self.exception_class = ColumnExistenceError
        self.error_message = '^Could not find column "r" in data$'
        self.kwargs = {self.column: "r"}

        self.check()

    def test_raise_exception_on_x_being_zero_index(self):
        self.exception_class = ColumnIndexError
        self.error_message = (
            "^No column number 0 in data. index should be between 1 and 10$"
        )
        self.kwargs = {self.column: 0}

        self.check()

    def test_raise_exception_on_x_being_larger_than_size(self):
        self.exception_class = ColumnIndexError
        self.error_message = (
            "^No column number 11 in data. index should be between 1 and 10$"
        )
        self.kwargs = {self.column: 11}

        self.check()


class TesDataColumnsRaiseExceptionBaseXColumnByXColumn(
    TestCase, DataColumnsRaiseExceptionBaseTestCase
):

    column = "x_column"


class TesDataColumnsRaiseExceptionBaseXErrColumnByXColumn(
    TestCase, DataColumnsRaiseExceptionBaseTestCase
):

    column = "xerr_column"


class TesDataColumnsRaiseExceptionBaseYColumnByXColumn(
    TestCase, DataColumnsRaiseExceptionBaseTestCase
):

    column = "y_column"


class TesDataColumnsRaiseExceptionBaseYErrColumnByXColumn(
    TestCase, DataColumnsRaiseExceptionBaseTestCase
):

    column = "yerr_column"
