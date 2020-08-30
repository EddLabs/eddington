from collections import OrderedDict
from pathlib import Path

import numpy as np

from eddington import FitData

DECIMAL = 5

RAW_DATA = OrderedDict(
    [
        ("a", [1, 2, 3]),
        ("b", [4, 5, 6]),
        ("c", [7, 8, 9]),
        ("d", [10, 11, 12]),
        ("e", [13, 14, 15]),
        ("f", [16, 17, 18]),
    ]
)
FILENAME = "save_test"
SHEET_NAME = "sheet1"


def assert_fit_data_are_equal(fit_data1: FitData, fit_data2: FitData):
    assert fit_data1.all_columns == fit_data2.all_columns, "The columns are different."
    for column in fit_data1.all_columns:
        np.testing.assert_almost_equal(
            fit_data1.data[column],
            fit_data2.data[column],
            decimal=DECIMAL,
            err_msg=f'Values does not match for "{column}"',
        )


def test_save_and_read_excel_with_name(tmpdir):
    original_fit_data = FitData(data=RAW_DATA)
    original_fit_data.save_excel(tmpdir, FILENAME, SHEET_NAME)
    loaded_fit_data = FitData.read_from_excel(
        Path(tmpdir / f"{FILENAME}.xlsx"), SHEET_NAME
    )
    assert_fit_data_are_equal(original_fit_data, loaded_fit_data)


def test_save_and_read_csv_with_name(tmpdir):
    original_fit_data = FitData(data=RAW_DATA)
    original_fit_data.save_csv(tmpdir, FILENAME)
    loaded_fit_data = FitData.read_from_csv(Path(tmpdir / f"{FILENAME}.csv"))

    assert_fit_data_are_equal(original_fit_data, loaded_fit_data)


def test_save_and_read_excel_without_name(tmpdir):
    original_fit_data = FitData(data=RAW_DATA)
    original_fit_data.save_excel(tmpdir, sheet=SHEET_NAME)
    loaded_fit_data = FitData.read_from_excel(
        Path(tmpdir / "fit_data.xlsx"), SHEET_NAME
    )

    assert_fit_data_are_equal(original_fit_data, loaded_fit_data)


def test_save_and_read_csv_without_name(tmpdir):
    original_fit_data = FitData(data=RAW_DATA)
    original_fit_data.save_csv(tmpdir)
    loaded_fit_data = FitData.read_from_csv(Path(tmpdir / "fit_data.csv"))

    assert_fit_data_are_equal(original_fit_data, loaded_fit_data)
