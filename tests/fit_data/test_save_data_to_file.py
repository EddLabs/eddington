from pathlib import Path

import numpy as np

from eddington import FitData
from tests.fit_data import COLUMNS

FILENAME = "save_test"
SHEET_NAME = "sheet1"


def test_save_and_read_excel_with_name(tmp_path):
    original_fit_data = FitData(COLUMNS)
    original_fit_data.save_excel(tmp_path, FILENAME, SHEET_NAME)
    loaded_fit_data = FitData.read_from_excel(
        Path(tmp_path / f"{FILENAME}.xlsx"), SHEET_NAME
    )
    original_keys, loaded_keys = (
        original_fit_data.data.keys(),
        loaded_fit_data.data.keys(),
    )
    original_values, loaded_values = (
        list(original_fit_data.data.values()),
        list(loaded_fit_data.data.values()),
    )
    assert original_keys == loaded_keys
    np.testing.assert_array_almost_equal(original_values, loaded_values, decimal=10)


def test_save_and_read_csv_with_name(tmp_path):
    original_fit_data = FitData(COLUMNS)
    original_fit_data.save_csv(tmp_path, FILENAME)
    loaded_fit_data = FitData.read_from_csv(Path(tmp_path / f"{FILENAME}.csv"))
    original_keys, loaded_keys = (
        original_fit_data.data.keys(),
        loaded_fit_data.data.keys(),
    )
    original_values, loaded_values = (
        list(original_fit_data.data.values()),
        list(loaded_fit_data.data.values()),
    )
    assert original_keys == loaded_keys
    np.testing.assert_array_almost_equal(original_values, loaded_values, decimal=10)


def test_save_and_read_excel_without_name(tmp_path):
    original_fit_data = FitData(COLUMNS)
    original_fit_data.save_excel(tmp_path, sheet=SHEET_NAME)
    loaded_fit_data = FitData.read_from_excel(
        Path(tmp_path / "FitData.xlsx"), SHEET_NAME
    )
    original_keys, loaded_keys = (
        original_fit_data.data.keys(),
        loaded_fit_data.data.keys(),
    )
    original_values, loaded_values = (
        list(original_fit_data.data.values()),
        list(loaded_fit_data.data.values()),
    )
    assert original_keys == loaded_keys
    np.testing.assert_array_almost_equal(original_values, loaded_values, decimal=10)


def test_save_and_read_csv_without_name(tmp_path):
    original_fit_data = FitData(COLUMNS)
    original_fit_data.save_csv(tmp_path)
    loaded_fit_data = FitData.read_from_csv(Path(tmp_path / "FitData.csv"))
    original_keys, loaded_keys = (
        original_fit_data.data.keys(),
        loaded_fit_data.data.keys(),
    )
    original_values, loaded_values = (
        list(original_fit_data.data.values()),
        list(loaded_fit_data.data.values()),
    )
    assert original_keys == loaded_keys
    np.testing.assert_array_almost_equal(original_values, loaded_values, decimal=10)
