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
from tests.util import dummy_function, assert_calls

EPSILON = 1e-5

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
def mock_plot_residuals(mocker):
    return mocker.patch("eddington.cli.plot_residuals")


@fixture
def mock_plot_data(mocker):
    return mocker.patch("eddington.cli.plot_data")


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


@parametrize(
    argnames="should_plot_fitting",
    argvalues=[True, False],
    idgen="plot_fitting={should_plot_fitting}",
)
@parametrize(
    argnames="should_plot_residuals",
    argvalues=[True, False],
    idgen="plot_residuals={should_plot_residuals}",
)
@parametrize(
    argnames="should_plot_data",
    argvalues=[True, False],
    idgen="plot_data={should_plot_data}",
)
@parametrize(
    argnames="should_output",
    argvalues=[True, False],
    idgen="output_directory={should_output}",
)
@parametrize(
    argnames="read_method, sheet, data_file_name",
    argvalues=[
        (fixture_ref(mock_read_from_csv), None, "data.csv"),
        (fixture_ref(mock_read_from_json), None, "data.json"),
        (fixture_ref(mock_read_from_excel), SHEET, "data.xlsx"),
    ],
    idgen="{data_file_name}",
)
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
    should_plot_fitting,
    should_plot_residuals,
    should_plot_data,
    should_output,
    tmpdir,
    mock_load_fit_func,
    mock_fit_to_data,
    mock_plot_fitting,
    mock_plot_residuals,
    mock_plot_data,
    mock_show_or_export,
):
    read_method.return_value = FIT_DATA
    data_file = make_existing(data_file_name, tmpdir)
    extra_cli_args = []
    if sheet is not None:
        extra_cli_args.extend(["--sheet", sheet])
    extend_args_by_flag(
        extra_cli_args, should_plot_fitting, "--plot-fitting", "--no-plot-fitting"
    )
    extend_args_by_flag(
        extra_cli_args, should_plot_residuals, "--plot-residuals", "--no-plot-residuals"
    )
    extend_args_by_flag(
        extra_cli_args, should_plot_data, "--plot-data", "--no-plot-data"
    )
    output_directory = None
    if should_output:
        output_directory = tmpdir / "output"
        output_directory.mkdir()
        extra_cli_args.extend(["--output-dir", str(output_directory)])
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
    show_calls = []
    if should_plot_data:
        assert_calls(
            mock_plot_data,
            [
                (
                    [],
                    dict(
                        data=FIT_DATA,
                        title_name=f"{FIT_FUNC.title_name} - Data",
                    ),
                ),
            ],
            rel=EPSILON,
        )
        output_data_path = None
        if should_output:
            output_data_path = output_directory / f"{FIT_FUNC.name}_data.png"
        show_calls.append(
            ([mock_plot_data.return_value], dict(output_path=output_data_path))
        )
    else:
        mock_plot_data.assert_not_called()
    if should_plot_fitting:
        assert_calls(
            mock_plot_fitting,
            [
                (
                    [],
                    dict(
                        a=FIT_RESULT.a,
                        data=FIT_DATA,
                        func=FIT_FUNC,
                        title_name=FIT_FUNC.title_name,
                    ),
                ),
            ],
            rel=EPSILON,
        )
        output_fitting_path = None
        if should_output:
            output_fitting_path = output_directory / f"{FIT_FUNC.name}.png"
        show_calls.append(
            ([mock_plot_fitting.return_value], dict(output_path=output_fitting_path))
        )
    else:
        mock_plot_fitting.assert_not_called()
    if should_plot_residuals:
        assert_calls(
            mock_plot_residuals,
            [
                (
                    [],
                    dict(
                        a=FIT_RESULT.a,
                        data=FIT_DATA,
                        func=FIT_FUNC,
                        title_name=f"{FIT_FUNC.title_name} - Residuals",
                    ),
                ),
            ],
            rel=EPSILON,
        )
        output_residuals_path = None
        if should_output:
            output_residuals_path = output_directory / f"{FIT_FUNC.name}_residuals.png"
        show_calls.append(
            (
                [mock_plot_residuals.return_value],
                dict(output_path=output_residuals_path),
            )
        )
    else:
        mock_plot_residuals.assert_not_called()
    assert_calls(mock_show_or_export, show_calls, rel=EPSILON)


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


def extend_args_by_flag(args, bool_flag, true_value, false_value):
    if bool_flag:
        args.append(true_value)
    else:
        args.append(false_value)


def make_existing(data_file_name, tmpdir):
    data_file = Path(tmpdir) / data_file_name
    data_file.touch()
    return data_file


def assert_code_and_output(result):
    assert result.exit_code == 0, "Result code should be successful"
    assert (
        result.output == f"{FIT_RESULT.pretty_string}\n"
    ), "Output is different than expected"
