from collections import OrderedDict
from pathlib import Path

import numpy as np

from eddington import FittingData

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


def assert_fitting_data_are_equal(
    fitting_data1: FittingData, fitting_data2: FittingData
):
    assert (
        fitting_data1.all_columns == fitting_data2.all_columns
    ), "The columns are different."
    for column in fitting_data1.all_columns:
        np.testing.assert_almost_equal(
            fitting_data1.data[column],
            fitting_data2.data[column],
            decimal=DECIMAL,
            err_msg=f'Values does not match for "{column}"',
        )


def test_save_and_read_excel_with_name(tmpdir):
    original_fitting_data = FittingData(data=RAW_DATA)
    original_fitting_data.save_excel(tmpdir, FILENAME, SHEET_NAME)
    loaded_fitting_data = FittingData.read_from_excel(
        Path(tmpdir / f"{FILENAME}.xlsx"), SHEET_NAME
    )
    assert_fitting_data_are_equal(original_fitting_data, loaded_fitting_data)


def test_save_and_read_csv_with_name(tmpdir):
    original_fitting_data = FittingData(data=RAW_DATA)
    original_fitting_data.save_csv(tmpdir, FILENAME)
    loaded_fitting_data = FittingData.read_from_csv(Path(tmpdir / f"{FILENAME}.csv"))

    assert_fitting_data_are_equal(original_fitting_data, loaded_fitting_data)


def test_save_and_read_excel_without_name(tmpdir):
    original_fitting_data = FittingData(data=RAW_DATA)
    original_fitting_data.save_excel(tmpdir, sheet=SHEET_NAME)
    loaded_fitting_data = FittingData.read_from_excel(
        Path(tmpdir / "fitting_data.xlsx"), SHEET_NAME
    )

    assert_fitting_data_are_equal(original_fitting_data, loaded_fitting_data)


def test_save_and_read_csv_without_name(tmpdir):
    original_fitting_data = FittingData(data=RAW_DATA)
    original_fitting_data.save_csv(tmpdir)
    loaded_fitting_data = FittingData.read_from_csv(Path(tmpdir / "fitting_data.csv"))

    assert_fitting_data_are_equal(original_fitting_data, loaded_fitting_data)
