from unittest import TestCase
from copy import deepcopy
import numpy as np

from eddington_core import (
    FitData,
    FitDataColumnIndexError,
    FitDataColumnExistenceError,
    FitDataColumnsLengthError,
)
from tests.fit_data import COLUMNS, COLUMNS_NAMES


class FitDataConstructorBaseTestCase:
    fit_data: FitData

    def setUp(self):
        self.fit_data = FitData(COLUMNS, **self.kwargs)

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


class TestFitDataConstructorWithoutArgs(TestCase, FitDataConstructorBaseTestCase):

    x = "a"
    xerr = "b"
    y = "c"
    yerr = "d"
    kwargs = dict()

    setUp = FitDataConstructorBaseTestCase.setUp


class TestFitDataConstructorWithIntX(TestCase, FitDataConstructorBaseTestCase):

    x = "c"
    xerr = "d"
    y = "e"
    yerr = "f"
    kwargs = dict(x_column=3)

    setUp = FitDataConstructorBaseTestCase.setUp


class TestFitDataConstructorWithStringX(TestCase, FitDataConstructorBaseTestCase):

    x = "c"
    xerr = "d"
    y = "e"
    yerr = "f"
    kwargs = dict(x_column="c")

    setUp = FitDataConstructorBaseTestCase.setUp


class TestFitDataConstructorWithIntY(TestCase, FitDataConstructorBaseTestCase):

    x = "a"
    xerr = "b"
    y = "e"
    yerr = "f"
    kwargs = dict(y_column=5)

    setUp = FitDataConstructorBaseTestCase.setUp


class TestFitDataConstructorWithStringY(TestCase, FitDataConstructorBaseTestCase):

    x = "a"
    xerr = "b"
    y = "e"
    yerr = "f"
    kwargs = dict(y_column="e")

    setUp = FitDataConstructorBaseTestCase.setUp


class TestFitDataConstructorWithIntXerr(TestCase, FitDataConstructorBaseTestCase):

    x = "a"
    xerr = "d"
    y = "e"
    yerr = "f"
    kwargs = dict(xerr_column=4)

    setUp = FitDataConstructorBaseTestCase.setUp


class TestFitDataConstructorWithStringXerr(TestCase, FitDataConstructorBaseTestCase):

    x = "a"
    xerr = "d"
    y = "e"
    yerr = "f"
    kwargs = dict(xerr_column="d")

    setUp = FitDataConstructorBaseTestCase.setUp


class TestFitDataConstructorWithIntYerr(TestCase, FitDataConstructorBaseTestCase):

    x = "a"
    xerr = "b"
    y = "c"
    yerr = "f"
    kwargs = dict(yerr_column=6)

    setUp = FitDataConstructorBaseTestCase.setUp


class TestFitDataConstructorWithStringYerr(TestCase, FitDataConstructorBaseTestCase):

    x = "a"
    xerr = "b"
    y = "c"
    yerr = "f"
    kwargs = dict(yerr_column="f")

    setUp = FitDataConstructorBaseTestCase.setUp


class TestFitDataConstructorWithXAndY(TestCase, FitDataConstructorBaseTestCase):

    x = "c"
    xerr = "d"
    y = "h"
    yerr = "i"
    kwargs = dict(x_column=3, y_column="h")

    setUp = FitDataConstructorBaseTestCase.setUp


class TestFitDataConstructorColumnsWithJumbled(
    TestCase, FitDataConstructorBaseTestCase
):

    x = "c"
    xerr = "a"
    y = "b"
    yerr = "i"
    kwargs = dict(x_column=3, xerr_column=1, y_column="b", yerr_column=9,)

    setUp = FitDataConstructorBaseTestCase.setUp


class FitDataConstructorRaiseColumnExceptionBaseTestCase:
    def check(self):
        self.assertRaisesRegex(
            self.exception_class, self.error_message, FitData, COLUMNS, **self.kwargs
        )

    def test_raise_exception_on_x_being_not_existing(self):
        self.exception_class = FitDataColumnExistenceError
        self.error_message = '^Could not find column "r" in data$'
        self.kwargs = {self.column: "r"}

        self.check()

    def test_raise_exception_on_x_being_zero_index(self):
        self.exception_class = FitDataColumnIndexError
        self.error_message = (
            "^No column number 0 in data. index should be between 1 and 10$"
        )
        self.kwargs = {self.column: 0}

        self.check()

    def test_raise_exception_on_x_being_larger_than_size(self):
        self.exception_class = FitDataColumnIndexError
        self.error_message = (
            "^No column number 11 in data. index should be between 1 and 10$"
        )
        self.kwargs = {self.column: 11}

        self.check()


class TesFitDataConstructorRaiseColumnExceptionBaseXColumnByXColumn(
    TestCase, FitDataConstructorRaiseColumnExceptionBaseTestCase
):

    column = "x_column"


class TesFitDataConstructorRaiseColumnExceptionBaseXErrColumnByXColumn(
    TestCase, FitDataConstructorRaiseColumnExceptionBaseTestCase
):

    column = "xerr_column"


class TesFitDataConstructorRaiseColumnExceptionBaseYColumnByXColumn(
    TestCase, FitDataConstructorRaiseColumnExceptionBaseTestCase
):

    column = "y_column"


class TesFitDataConstructorRaiseColumnExceptionBaseYErrColumnByXColumn(
    TestCase, FitDataConstructorRaiseColumnExceptionBaseTestCase
):

    column = "yerr_column"


class TestFitDataConstructorGeneralExceptions(TestCase):
    def test_exception_risen_because_of_columns_length(self):
        data = deepcopy(COLUMNS)
        data["a"] = data["a"][:-2]
        self.assertRaisesRegex(
            FitDataColumnsLengthError,
            "^All columns in FitData should have the same length$",
            FitData,
            data,
        )
