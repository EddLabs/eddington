from collections import namedtuple
from pathlib import Path
from unittest import TestCase
from copy import deepcopy
import numpy as np
from mock import patch, mock_open, PropertyMock

from eddington_core import FitData
from eddington_core.exceptions import FitDataInvalidFileSyntax
from tests.fit_data import COLUMNS, VALUES, ROWS, CONTENT


class FitDataReadFromFileBaseTestCase:
    @classmethod
    def check_data_by_keys(cls, actual_fit_data):
        for key, value in actual_fit_data.data.items():
            np.testing.assert_equal(
                actual_fit_data.data[key],
                COLUMNS[key],
                err_msg="Data is different than expected",
            )

    @classmethod
    def check_data_by_indexes(cls, actual_fit_data):
        for key, value in actual_fit_data.data.items():
            np.testing.assert_equal(
                actual_fit_data.data[key],
                VALUES[key],
                err_msg="Data is different than expected",
            )

    @classmethod
    def check_columns(
        cls, actual_fit_data, x_column=0, xerr_column=1, y_column=2, yerr_column=3
    ):
        np.testing.assert_equal(
            actual_fit_data.x, VALUES[x_column], err_msg="X is different than expected",
        )
        np.testing.assert_equal(
            actual_fit_data.xerr,
            VALUES[xerr_column],
            err_msg="X Error is different than expected",
        )
        np.testing.assert_equal(
            actual_fit_data.y, VALUES[y_column], err_msg="Y is different than expected",
        )
        np.testing.assert_equal(
            actual_fit_data.yerr,
            VALUES[yerr_column],
            err_msg="Y Error is different than expected",
        )

    def test_read_with_headers_successful(self):
        self.rows = ROWS

        actual_fit_data = self.read()

        self.check_data_by_keys(actual_fit_data)
        self.check_columns(actual_fit_data)

    def test_read_without_headers_successful(self):
        self.rows = CONTENT

        actual_fit_data = self.read()

        self.check_data_by_indexes(actual_fit_data)
        self.check_columns(actual_fit_data)

    def test_read_without_headers_unsuccessful(self):
        self.rows = deepcopy(CONTENT)
        self.rows[1][0] = "f"

        self.assertRaises(FitDataInvalidFileSyntax, self.read)

    def test_read_with_x_column(self):
        self.rows = ROWS

        actual_fit_data = self.read(x_column=3)

        self.check_columns(
            actual_fit_data, x_column=2, xerr_column=3, y_column=4, yerr_column=5
        )

    def test_read_with_xerr_column(self):
        self.rows = ROWS

        actual_fit_data = self.read(xerr_column=3)

        self.check_columns(
            actual_fit_data, x_column=0, xerr_column=2, y_column=3, yerr_column=4
        )

    def test_read_with_y_column(self):
        self.rows = ROWS

        actual_fit_data = self.read(y_column=5)

        self.check_columns(
            actual_fit_data, x_column=0, xerr_column=1, y_column=4, yerr_column=5
        )

    def test_read_with_yerr_column(self):
        self.rows = ROWS

        actual_fit_data = self.read(yerr_column=5)

        self.check_columns(
            actual_fit_data, x_column=0, xerr_column=1, y_column=2, yerr_column=4
        )


class TestReadFitDataFromCSV(TestCase, FitDataReadFromFileBaseTestCase):

    file_name = "file"
    filepath = Path("path/to") / file_name

    def setUp(self):
        csv_reader_patcher = patch("csv.reader")
        self.reader = csv_reader_patcher.start()
        self.addCleanup(csv_reader_patcher.stop)

    def read(self, **kwargs):
        self.reader.return_value = self.rows
        m_open = mock_open()
        with patch("eddington_core.fit_data.open", m_open):
            actual_fit_data = FitData.read_from_csv(self.filepath, **kwargs)
        return actual_fit_data


class TestReadFitDataFromExcel(TestCase, FitDataReadFromFileBaseTestCase):
    DummyCell = namedtuple("DummyCell", "value")
    file_name = "file"
    filepath = Path("path/to") / file_name
    sheet_name = "sheet"

    def setUp(self):
        open_workbook_patcher = patch("xlrd.open_workbook")
        self.open_workbook = open_workbook_patcher.start()
        self.sheet = self.open_workbook.return_value.sheet_by_name.return_value
        self.addCleanup(open_workbook_patcher.stop)

        type(self.sheet).nrows = PropertyMock(side_effect=self.nrows)
        self.sheet.row.side_effect = self.get_row

    def nrows(self):
        return len(self.rows)

    def get_row(self, i):
        return [
            TestReadFitDataFromExcel.DummyCell(value=element)
            for element in self.rows[i]
        ]

    def read(self, **kwargs):
        return FitData.read_from_excel(self.filepath, self.sheet_name, **kwargs)
