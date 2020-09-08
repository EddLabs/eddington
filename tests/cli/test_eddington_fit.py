from pathlib import Path

import numpy as np
from click.testing import CliRunner
from pytest_cases import (
    THIS_MODULE,
    fixture,
    fixture_ref,
    parametrize,
    parametrize_with_cases,
    case,
)

from eddington import FitData, FitResult, linear
from eddington.cli import eddington_cli
from tests.conftest import mock_read_from_csv, mock_read_from_excel, mock_read_from_json
from tests.util import assert_dict_equal, dummy_function

COLUMNS_SET_TAG = "column_set"

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


@fixture
def mock_plot_fitting(mocker):
    return mocker.patch("eddington.cli.plot_fitting")


@fixture
def mock_show_or_export(mocker):
    return mocker.patch("eddington.cli.show_or_export")


@case(tags=[COLUMNS_SET_TAG])
def case_no_args():
    return [], dict(
        x_column=None,
        xerr_column=None,
        y_column=None,
        yerr_column=None,
    )


@case(tags=[COLUMNS_SET_TAG])
def case_x_column():
    x_column = "x_column"
    return ["--x-column", x_column], dict(
        x_column=x_column,
        xerr_column=None,
        y_column=None,
        yerr_column=None,
    )


@case(tags=[COLUMNS_SET_TAG])
def case_xerr_column():
    xerr_column = "xerr_column"
    return ["--xerr-column", xerr_column], dict(
        x_column=None,
        xerr_column=xerr_column,
        y_column=None,
        yerr_column=None,
    )


@case(tags=[COLUMNS_SET_TAG])
def case_y_column():
    y_column = "y_column"
    return ["--y-column", y_column], dict(
        x_column=None,
        xerr_column=None,
        y_column=y_column,
        yerr_column=None,
    )


@case(tags=[COLUMNS_SET_TAG])
def case_yerr_column():
    yerr_column = "yerr_column"
    return ["--yerr-column", yerr_column], dict(
        x_column=None,
        xerr_column=None,
        y_column=None,
        yerr_column=yerr_column,
    )


read_methods = parametrize(
    argnames="read_method, sheet, data_file_name",
    argvalues=[
        (fixture_ref(mock_read_from_csv), None, "data.csv"),
        (fixture_ref(mock_read_from_json), None, "data.json"),
        (fixture_ref(mock_read_from_excel), SHEET, "data.xlsx"),
    ],
    idgen="{data_file_name}",
)


@read_methods
@parametrize_with_cases(
    argnames="cli_args, read_kwargs", cases=THIS_MODULE, has_tag=COLUMNS_SET_TAG
)
def test_fit_with_columns_set(
    cli_args,
    read_kwargs,
    read_method,
    sheet,
    data_file_name,
    cli_runner,
    tmpdir,
    mock_load_fit_func,
    mock_fit_to_data,
    mock_plot_fitting,
    mock_show_or_export,
):
    read_method.return_value = FIT_DATA
    data_file = make_existing(data_file_name, tmpdir)
    extra_cli_args = ["--sheet", sheet] if sheet is not None else []
    result = cli_runner.invoke(
        eddington_cli,
        ["fit", FIT_FUNC.name, "-d", str(data_file), *cli_args, *extra_cli_args],
    )
    assert_code_and_output(result)
    extra_read_kwargs = dict(sheet=sheet) if sheet is not None else dict()
    read_method.assert_called_once_with(
        filepath=data_file, **read_kwargs, **extra_read_kwargs
    )
    mock_load_fit_func.assert_called_with(FIT_FUNC.name)
    mock_fit_to_data.assert_called_with(FIT_DATA, FIT_FUNC)
    assert mock_plot_fitting.call_count == 1
    assert_dict_equal(
        mock_plot_fitting.call_args_list[0][1],
        dict(
            a=FIT_RESULT.a, data=FIT_DATA, func=FIT_FUNC, title_name=FIT_FUNC.title_name
        ),
        rel=1e-5,
    )
    mock_show_or_export.assert_called_once_with(mock_plot_fitting.return_value)


@read_methods
def test_fit_without_plot_fitting(
    read_method,
    sheet,
    data_file_name,
    cli_runner,
    tmpdir,
    mock_read_from_csv,
    mock_load_fit_func,
    mock_fit_to_data,
    mock_plot_fitting,
    mock_show_or_export,
):
    read_method.return_value = FIT_DATA
    data_file = make_existing(data_file_name, tmpdir)
    extra_cli_args = ["--sheet", sheet] if sheet is not None else []
    result = cli_runner.invoke(
        eddington_cli,
        [
            "fit",
            FIT_FUNC.name,
            "-d",
            str(data_file),
            "--no-plot-fitting",
            *extra_cli_args,
        ],
    )
    assert_code_and_output(result)
    extra_read_kwargs = dict(sheet=sheet) if sheet is not None else dict()
    read_method.assert_called_once_with(
        filepath=data_file,
        x_column=None,
        xerr_column=None,
        y_column=None,
        yerr_column=None,
        **extra_read_kwargs,
    )
    mock_load_fit_func.assert_called_with(FIT_FUNC.name)
    mock_fit_to_data.assert_called_with(FIT_DATA, FIT_FUNC)
    mock_plot_fitting.assert_not_called()
    mock_show_or_export.assert_not_called()


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


# Utility functions


def make_existing(data_file_name, tmpdir):
    data_file = Path(tmpdir) / data_file_name
    data_file.touch()
    return data_file


def assert_code_and_output(result):
    assert result.exit_code == 0, "Result code should be successful"
    assert (
        result.output == f"{FIT_RESULT.pretty_string}\n"
    ), "Output is different than expected"
