from pathlib import Path

import numpy as np
from pytest_cases import (
    THIS_MODULE,
    fixture,
    fixture_ref,
    parametrize,
    parametrize_with_cases,
)

from eddington import FitData, FitFunctionsRegistry, FitResult, linear
from eddington.cli import eddington_cli
from tests.util import dummy_function

FIT_FUNC = dummy_function("dummy", "Dummy Syntax")
A = np.array([1, 2])
FIT_DATA = FitData.random(linear, a=A)
FIT_RESULT = FitResult(
    a0=[0.9, 2.3],
    a=[1.1, 1.9],
    aerr=[0.1, 0.2],
    acov=[[0.01, 0.03], [0.03, 0.02]],
    degrees_of_freedom=6,
    chi2=0.345,
)
SHEET = "sheet"


@fixture
def mock_fit_to_data(mocker):
    fit_method = mocker.patch("eddington.cli.fit_to_data")
    fit_method.return_value = FIT_RESULT
    return fit_method


def case_csv_data(mock_read_from_csv):
    mock_read_from_csv.return_value = FIT_DATA
    data_file_name = "data.csv"
    return data_file_name, mock_read_from_csv, None


def case_json_data(mock_read_from_json):
    mock_read_from_json.return_value = FIT_DATA
    data_file_name = "data.json"
    return data_file_name, mock_read_from_json, None


def case_excel_data(mock_read_from_excel):
    mock_read_from_excel.return_value = FIT_DATA
    data_file_name = "data.xlsx"
    return data_file_name, mock_read_from_excel, SHEET


@parametrize_with_cases(
    argnames="data_file_name, read_method, sheet", cases=THIS_MODULE
)
def test_read_data_from_file(
    data_file_name,
    read_method,
    sheet,
    cli_runner,
    mock_fit_to_data,
    mock_load_function,
    tmpdir,
):
    mock_load_function.return_value = FIT_FUNC
    data_file = Path(tmpdir) / data_file_name
    data_file.touch()
    cli_args = ["fit", FIT_FUNC.name, "-d", str(data_file)]
    if sheet is not None:
        cli_args.append(f"--sheet={sheet}")
    result = cli_runner.invoke(eddington_cli, cli_args)
    assert result.exit_code == 0, "Result code should be successful"
    assert (
        result.output == f"{FIT_RESULT.pretty_string}\n"
    ), "Output is different than expected"
    read_kwargs = dict(filepath=data_file)
    if sheet is not None:
        read_kwargs["sheet"] = sheet
    read_method.assert_called_once_with(**read_kwargs)
    mock_load_function.assert_called_with(FIT_FUNC.name)
    mock_fit_to_data.assert_called_with(FIT_DATA, FIT_FUNC)


def test_read_data_from_excel_fails_for_no_sheet_name(cli_runner, tmpdir):
    data_file = Path(tmpdir) / "data.xlsx"
    data_file.touch()
    result = cli_runner.invoke(
        eddington_cli, ["fit", FIT_FUNC.name, "-d", str(data_file)]
    )
    assert result.exit_code == 1, "Command should fail"
    assert (
        result.output == "Sheet name has not been specified!\n"
    ), "Output is different than expected"


def test_read_data_from_unknown_type(cli_runner, tmpdir):
    data_file = Path(tmpdir) / "data.bla"
    data_file.touch()
    result = cli_runner.invoke(
        eddington_cli, ["fit", FIT_FUNC.name, "-d", str(data_file)]
    )
    assert result.exit_code == 1, "Command should fail"
    assert (
        result.output == 'Cannot read data with ".bla" suffix\n'
    ), "Output is different than expected"
