import pytest

from eddington import FittingData
from tests.fitting_data import COLUMNS
from tests.util import assert_calls

EPSILON = 1e-3


@pytest.fixture
def mock_save_as_excel(mocker):
    return mocker.patch("eddington.io_util.save_as_excel")


@pytest.fixture
def mock_save_as_csv(mocker):
    return mocker.patch("eddington.io_util.save_as_csv")


def test_default_save_as_excel(mock_save_as_excel):
    output_directory = "/path/to/directory"
    data = FittingData(COLUMNS)
    data.save_excel(output_directory=output_directory)
    content = [data.all_columns, *data.all_records]
    assert_calls(
        mock_save_as_excel,
        [
            (
                [],
                dict(
                    content=content,
                    output_directory=output_directory,
                    file_name="fitting_data",
                    sheet=None,
                ),
            )
        ],
        rel=EPSILON,
    )


def test_save_as_excel_with_file_and_sheet(mock_save_as_excel):
    output_directory = "/path/to/directory"
    file_name = "data"
    sheet = "sheet"
    data = FittingData(COLUMNS)
    data.save_excel(output_directory=output_directory, name=file_name, sheet=sheet)
    content = [data.all_columns, *data.all_records]
    assert_calls(
        mock_save_as_excel,
        [
            (
                [],
                dict(
                    content=content,
                    output_directory=output_directory,
                    file_name=file_name,
                    sheet=sheet,
                ),
            )
        ],
        rel=EPSILON,
    )


def test_default_save_as_csv(mock_save_as_csv):
    output_directory = "/path/to/directory"
    data = FittingData(COLUMNS)
    data.save_csv(output_directory=output_directory)
    content = [data.all_columns, *data.all_records]
    assert_calls(
        mock_save_as_csv,
        [
            (
                [],
                dict(
                    content=content,
                    output_directory=output_directory,
                    file_name="fitting_data",
                ),
            )
        ],
        rel=EPSILON,
    )


def test_save_as_csv_with_file_name(mock_save_as_csv):
    output_directory = "/path/to/directory"
    file_name = "data"
    data = FittingData(COLUMNS)
    data.save_csv(output_directory=output_directory, name=file_name)
    content = [data.all_columns, *data.all_records]
    assert_calls(
        mock_save_as_csv,
        [
            (
                [],
                dict(
                    content=content,
                    output_directory=output_directory,
                    file_name=file_name,
                ),
            )
        ],
        rel=EPSILON,
    )
