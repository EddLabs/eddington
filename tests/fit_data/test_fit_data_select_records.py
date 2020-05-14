from unittest import TestCase

import numpy as np
from eddington_core import FitData
from eddington_core.exceptions import FitDataColumnsSelectionError

from tests.fit_data import COLUMNS, NUMBER_OF_RECORDS, VALUES


class BaseFitDataSelectRecordTestCase:
    def setUp(self):
        self.fit_data = FitData(COLUMNS)
        self.select_records()
        self.expected_x = self.extract_values(VALUES[0])
        self.expected_xerr = self.extract_values(VALUES[1])
        self.expected_y = self.extract_values(VALUES[2])
        self.expected_yerr = self.extract_values(VALUES[3])

    @classmethod
    def extract_values(cls, column):
        return [column[i - 1] for i in cls.selected]

    def test_x(self):
        np.testing.assert_equal(
            self.fit_data.x, self.expected_x, err_msg="X is different than expected"
        )

    def test_xerr(self):
        np.testing.assert_equal(
            self.fit_data.xerr,
            self.expected_xerr,
            err_msg="X error is different than expected",
        )

    def test_y(self):
        np.testing.assert_equal(
            self.fit_data.y, self.expected_y, err_msg="Y is different than expected"
        )

    def test_yerr(self):
        np.testing.assert_equal(
            self.fit_data.yerr,
            self.expected_yerr,
            err_msg="Y error is different than expected",
        )

    def test_is_selected(self):
        for i in range(1, self.fit_data.length + 1):
            if i in self.selected:
                self.assertTrue(
                    self.fit_data.is_selected(i), msg=f"Record {i} was not selected"
                )
            else:
                self.assertFalse(
                    self.fit_data.is_selected(i), msg=f"Record {i} was selected"
                )


class TestFitDataUnselectOneRecord(TestCase, BaseFitDataSelectRecordTestCase):
    setUp = BaseFitDataSelectRecordTestCase.setUp
    selected = [1] + list(range(3, NUMBER_OF_RECORDS + 1))

    def select_records(self):
        self.fit_data.unselect_record(2)


class TestFitDataUnselectTwoRecord(TestCase, BaseFitDataSelectRecordTestCase):
    setUp = BaseFitDataSelectRecordTestCase.setUp
    selected = [1, 3, 4] + list(range(6, NUMBER_OF_RECORDS + 1))

    def select_records(self):
        self.fit_data.unselect_record(2)
        self.fit_data.unselect_record(5)


class TestFitDataUnselectMultipleRecord(TestCase, BaseFitDataSelectRecordTestCase):
    setUp = BaseFitDataSelectRecordTestCase.setUp
    selected = [1, 4, 6, 7, 8, 9] + list(range(11, NUMBER_OF_RECORDS + 1))

    def select_records(self):
        self.fit_data.unselect_record(2)
        self.fit_data.unselect_record(5)
        self.fit_data.unselect_record(3)
        self.fit_data.unselect_record(10)


class TestFitDataUnselectAllRecords(TestCase, BaseFitDataSelectRecordTestCase):
    setUp = BaseFitDataSelectRecordTestCase.setUp
    selected = []

    def select_records(self):
        self.fit_data.unselect_all_records()


class TestFitDataSelectRecord(TestCase, BaseFitDataSelectRecordTestCase):
    setUp = BaseFitDataSelectRecordTestCase.setUp
    selected = [2]

    def select_records(self):
        self.fit_data.unselect_all_records()
        self.fit_data.select_record(2)


class TestFitDataSelectTwoRecords(TestCase, BaseFitDataSelectRecordTestCase):
    setUp = BaseFitDataSelectRecordTestCase.setUp
    selected = [2, 5]

    def select_records(self):
        self.fit_data.unselect_all_records()
        self.fit_data.select_record(2)
        self.fit_data.select_record(5)


class TestFitDataSelectMultipleRecords(TestCase, BaseFitDataSelectRecordTestCase):
    setUp = BaseFitDataSelectRecordTestCase.setUp
    selected = [2, 3, 5, 10]

    def select_records(self):
        self.fit_data.unselect_all_records()
        self.fit_data.select_record(2)
        self.fit_data.select_record(5)
        self.fit_data.select_record(3)
        self.fit_data.select_record(10)


class TestFitDataReselectRecord(TestCase, BaseFitDataSelectRecordTestCase):
    setUp = BaseFitDataSelectRecordTestCase.setUp
    selected = list(range(1, NUMBER_OF_RECORDS + 1))

    def select_records(self):
        self.fit_data.unselect_record(2)
        self.fit_data.select_record(2)


class TestFitDataSelectAllRecords(TestCase, BaseFitDataSelectRecordTestCase):
    setUp = BaseFitDataSelectRecordTestCase.setUp
    selected = list(range(1, NUMBER_OF_RECORDS + 1))

    def select_records(self):
        self.fit_data.unselect_all_records()
        self.fit_data.select_record(2)
        self.fit_data.select_record(5)
        self.fit_data.select_all_records()


class TestFitDataSetSelectedRecordsIndices(TestCase, BaseFitDataSelectRecordTestCase):
    setUp = BaseFitDataSelectRecordTestCase.setUp
    selected = [2, 5]

    def select_records(self):
        # fmt: off
        self.fit_data.records_indices = \
            [False, True, False, False, True] + [False] * (NUMBER_OF_RECORDS - 5)
        # fmt: on


class TestFitDataSelectRaiseError(TestCase):
    fit_data = FitData(COLUMNS)

    def test_set_selection_with_different_size(self):
        def set_records_indices():
            # fmt: off
            self.fit_data.records_indices = \
                [False, False, True]
            # fmt: on

        self.assertRaisesRegex(
            FitDataColumnsSelectionError,
            f"^Should select {NUMBER_OF_RECORDS} records, only 3 selected.$",
            set_records_indices,
        )

    def test_set_selection_with_non_boolean_values(self):
        def set_records_indices():
            # fmt: off
            self.fit_data.records_indices = \
                [False, False, "dummy"] + [True] * (NUMBER_OF_RECORDS - 3)
            # fmt: on

        self.assertRaisesRegex(
            FitDataColumnsSelectionError,
            "^When setting record indices, all values should be booleans.$",
            set_records_indices,
        )
