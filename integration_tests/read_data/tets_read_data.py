from pytest_cases import parametrize_with_cases, THIS_MODULE
from pathlib import Path
import pytest

from eddington import FitData

SOURCES_DIRECTORY = Path(__file__).parent.parent / "resources"

EXCEL_FILE = SOURCES_DIRECTORY / "data.xlsx"
SHEET_NAME = "data1"

CSV_FILE = SOURCES_DIRECTORY / "data.csv"


def case_read_excel():
    def read_method(**kwargs):
        return FitData.read_from_excel(EXCEL_FILE, SHEET_NAME, **kwargs)

    return read_method


def case_read_csv():
    def read_method(**kwargs):
        return FitData.read_from_csv(CSV_FILE, **kwargs)

    return read_method


@parametrize_with_cases(argnames="read_method", cases=THIS_MODULE)
def test_simple_read(read_method):
    fit_data: FitData = read_method()
    assert fit_data.x == pytest.approx(
        [10, 20, 30, 40, 50, 60, 70]
    ), "x is different than expected"
    assert fit_data.xerr == pytest.approx(
        [0.5, 1.0, 1.2, 0.3, 0.4, 1.1, 1.3]
    ), "x error is different than expected"
    assert fit_data.y == pytest.approx(
        [16.0, 29.0, 47.0, 56.0, 70.0, 92.0, 100.0]
    ), "y is different than expected"
    assert fit_data.yerr == pytest.approx(
        [1.0, 1.3, 0.8, 2.0, 1.1, 0.2, 2.0]
    ), "y error is different than expected"


@parametrize_with_cases(argnames="read_method", cases=THIS_MODULE)
def test_read_with_y_column(read_method):
    fit_data: FitData = read_method(y_column=5)
    assert fit_data.x == pytest.approx(
        [10, 20, 30, 40, 50, 60, 70]
    ), "x is different than expected"
    assert fit_data.xerr == pytest.approx(
        [0.5, 1.0, 1.2, 0.3, 0.4, 1.1, 1.3]
    ), "x error is different than expected"
    assert fit_data.y == pytest.approx(
        [100.0, 401.0, 910.0, 1559.0, 2480.0, 3623.0, 4910.0]
    ), "y is different than expected"
    assert fit_data.yerr == pytest.approx(
        [14.0, 10.0, 11.0, 8.0, 10.0, 5.0, 16.0]
    ), "y error is different than expected"
