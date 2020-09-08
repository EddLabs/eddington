from pathlib import Path

import numpy as np
from click.testing import CliRunner
from pytest_cases import THIS_MODULE, fixture, parametrize_with_cases

from eddington import FitData, FitResult, linear
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
def mock_load_fit_func(mock_load_function):
    mock_load_function.return_value = FIT_FUNC
    return mock_load_function


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
def test_simple_fit(
    data_file_name,
    read_method,
    sheet,
    mock_fit_to_data,
    mock_load_fit_func,
    tmpdir,
):
    data_file = existing_data_file(data_file_name, tmpdir)
    read_kwargs = dict(
        filepath=data_file,
        x_column=None,
        xerr_column=None,
        y_column=None,
        yerr_column=None,
    )
    assert_successful_result(
        cli_args=["fit", FIT_FUNC.name, "-d", str(data_file)],
        base_read_kwargs=read_kwargs,
        read_method=read_method,
        mock_load_fit_func=mock_load_fit_func,
        mock_fit_to_data=mock_fit_to_data,
        sheet=sheet,
    )


@parametrize_with_cases(
    argnames="data_file_name, read_method, sheet", cases=THIS_MODULE
)
def test_fit_with_specified_x_column(
    data_file_name,
    read_method,
    sheet,
    mock_fit_to_data,
    mock_load_fit_func,
    tmpdir,
):
    x_column = "x_column"
    data_file = existing_data_file(data_file_name, tmpdir)
    read_kwargs = dict(
        filepath=data_file,
        x_column=x_column,
        xerr_column=None,
        y_column=None,
        yerr_column=None,
    )
    assert_successful_result(
        cli_args=["fit", FIT_FUNC.name, "-d", str(data_file), "--x-column", x_column],
        base_read_kwargs=read_kwargs,
        read_method=read_method,
        mock_load_fit_func=mock_load_fit_func,
        mock_fit_to_data=mock_fit_to_data,
        sheet=sheet,
    )


@parametrize_with_cases(
    argnames="data_file_name, read_method, sheet", cases=THIS_MODULE
)
def test_fit_with_specified_xerr_column(
    data_file_name,
    read_method,
    sheet,
    mock_fit_to_data,
    mock_load_fit_func,
    tmpdir,
):
    xerr_column = "xerr_column"
    data_file = existing_data_file(data_file_name, tmpdir)
    read_kwargs = dict(
        filepath=data_file,
        x_column=None,
        xerr_column=xerr_column,
        y_column=None,
        yerr_column=None,
    )
    assert_successful_result(
        cli_args=[
            "fit",
            FIT_FUNC.name,
            "-d",
            str(data_file),
            "--xerr-column",
            xerr_column,
        ],
        base_read_kwargs=read_kwargs,
        read_method=read_method,
        mock_load_fit_func=mock_load_fit_func,
        mock_fit_to_data=mock_fit_to_data,
        sheet=sheet,
    )


@parametrize_with_cases(
    argnames="data_file_name, read_method, sheet", cases=THIS_MODULE
)
def test_fit_with_specified_y_column(
    data_file_name,
    read_method,
    sheet,
    mock_fit_to_data,
    mock_load_fit_func,
    tmpdir,
):
    y_column = "y_column"
    data_file = existing_data_file(data_file_name, tmpdir)
    read_kwargs = dict(
        filepath=data_file,
        x_column=None,
        xerr_column=None,
        y_column=y_column,
        yerr_column=None,
    )
    assert_successful_result(
        cli_args=["fit", FIT_FUNC.name, "-d", str(data_file), "--y-column", y_column],
        base_read_kwargs=read_kwargs,
        read_method=read_method,
        mock_load_fit_func=mock_load_fit_func,
        mock_fit_to_data=mock_fit_to_data,
        sheet=sheet,
    )


@parametrize_with_cases(
    argnames="data_file_name, read_method, sheet", cases=THIS_MODULE
)
def test_fit_with_specified_yerr_column(
    data_file_name,
    read_method,
    sheet,
    mock_fit_to_data,
    mock_load_fit_func,
    tmpdir,
):
    yerr_column = "yerr_column"
    data_file = existing_data_file(data_file_name, tmpdir)
    read_kwargs = dict(
        filepath=data_file,
        x_column=None,
        xerr_column=None,
        y_column=None,
        yerr_column=yerr_column,
    )
    assert_successful_result(
        cli_args=[
            "fit",
            FIT_FUNC.name,
            "-d",
            str(data_file),
            "--yerr-column",
            yerr_column,
        ],
        base_read_kwargs=read_kwargs,
        read_method=read_method,
        mock_load_fit_func=mock_load_fit_func,
        mock_fit_to_data=mock_fit_to_data,
        sheet=sheet,
    )


def test_read_data_from_excel_fails_for_no_sheet_name(tmpdir):
    cli_runner = CliRunner()
    data_file = Path(tmpdir) / "data.xlsx"
    data_file.touch()
    result = cli_runner.invoke(
        eddington_cli, ["fit", FIT_FUNC.name, "-d", str(data_file)]
    )
    assert result.exit_code == 1, "Command should fail"
    assert (
        result.output == "Sheet name has not been specified!\n"
    ), "Output is different than expected"


def test_read_data_from_unknown_type(tmpdir):
    cli_runner = CliRunner()
    data_file = Path(tmpdir) / "data.bla"
    data_file.touch()
    result = cli_runner.invoke(
        eddington_cli, ["fit", FIT_FUNC.name, "-d", str(data_file)]
    )
    assert result.exit_code == 1, "Command should fail"
    assert (
        result.output == 'Cannot read data with ".bla" suffix\n'
    ), "Output is different than expected"


def existing_data_file(data_file_name, tmpdir):
    data_file = Path(tmpdir) / data_file_name
    data_file.touch()
    return data_file


def build_cli_args(args, sheet):
    cli_args = args
    if sheet is not None:
        cli_args.append(f"--sheet={sheet}")
    return cli_args


def assert_successful_result(
    cli_args, base_read_kwargs, read_method, mock_load_fit_func, mock_fit_to_data, sheet
):
    cli_runner = CliRunner()
    cli_args = build_cli_args(cli_args, sheet)
    result = cli_runner.invoke(eddington_cli, cli_args)
    assert result.exit_code == 0, "Result code should be successful"
    assert (
        result.output == f"{FIT_RESULT.pretty_string}\n"
    ), "Output is different than expected"

    if sheet is not None:
        base_read_kwargs["sheet"] = sheet
    read_method.assert_called_once_with(**base_read_kwargs)
    mock_load_fit_func.assert_called_with(FIT_FUNC.name)
    mock_fit_to_data.assert_called_with(FIT_DATA, FIT_FUNC)
