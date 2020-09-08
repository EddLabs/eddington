"""CLI for Eddington."""
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import click
import numpy as np
from prettytable import PrettyTable

from eddington import (
    FitData,
    FitFunctionsRegistry,
    __version__,
    fit_to_data,
    plot_data,
    plot_fitting,
    plot_residuals,
    show_or_export,
)

# pylint: disable=too-many-arguments,invalid-name,too-many-locals


@click.group("eddington")
@click.version_option(version=__version__)
def eddington_cli():
    """Command line for Eddington."""


@eddington_cli.command("list")
@click.option(
    "-r",
    "--regex",
    type=str,
    default=None,
    help="Filter functions by a regular expression",
)
def eddington_list(regex: Optional[str]):
    """Prints all fit functions in a pretty table."""
    table = PrettyTable(field_names=["Function", "Syntax"])
    for func in FitFunctionsRegistry.all():
        if regex is None or re.search(regex, func.name):
            table.add_row([func.signature, func.syntax])
    click.echo(table)


@eddington_cli.command("fit")
@click.pass_context
@click.argument("fit_func", type=str, default="linear")
@click.option(
    "-d",
    "--data-file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    help="Data file to read from.",
)
@click.option("-s", "--sheet", type=str, help="Sheet name for excel files.")
@click.option(
    "--a0",
    type=str,
    help=(
        "Initial guess for the fitting algorithm. "
        "Should be given as floating point numbers separated by commas"
    ),
)
@click.option("--x-column", type=str, help="Column to read x values from.")
@click.option("--xerr-column", type=str, help="Column to read x error values from.")
@click.option("--y-column", type=str, help="Column to read y values from.")
@click.option("--yerr-column", type=str, help="Column to read y error values from.")
@click.option("--x-label", type=str, help="Label for the x axis.")
@click.option("--y-label", type=str, help="Label for the y axis.")
@click.option("--grid/--no-grid", default=False, help="Add grid lines to plots.")
@click.option(
    "--plot-fitting/--no-plot-fitting",
    "should_plot_fitting",
    default=True,
    help="Should plot fitting.",
)
@click.option(
    "--plot-residuals/--no-plot-residuals",
    "should_plot_residuals",
    default=True,
    help="Should plot residuals.",
)
@click.option(
    "--plot-data/--no-plot-data",
    "should_plot_data",
    default=False,
    help="Should plot data.",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(dir_okay=True, file_okay=False),
    help="Output directory to save plots in.",
)
@click.option(
    "--json",
    is_flag=True,
    default=False,
    help="Save result as json instead of text.",
)
def eddington_fit(
    ctx: click.Context,
    fit_func: Optional[str],
    data_file: str,
    sheet: Optional[str],
    a0: Optional[str],
    x_column: Optional[str],
    xerr_column: Optional[str],
    y_column: Optional[str],
    yerr_column: Optional[str],
    x_label: Optional[str],
    y_label: Optional[str],
    grid,
    should_plot_fitting: bool,
    should_plot_residuals: bool,
    should_plot_data: bool,
    output_dir: Union[Path, str],
    json: bool,
):
    """Fit data file according to a fitting function."""
    # fmt: off
    data = __load_data_file(
        ctx, Path(data_file), sheet,
        x_column=x_column, xerr_column=xerr_column,
        y_column=y_column, yerr_column=yerr_column,
    )
    # fmt: on
    func = FitFunctionsRegistry.load(fit_func)
    result = fit_to_data(data, func, a0=__calc_a0(a0))
    click.echo(result.pretty_string)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if json:
            result.save_json(output_dir / f"{func.name}_result.json")
        else:
            result.save_txt(output_dir / f"{func.name}_result.txt")
    if x_label is None:
        x_label = data.x_column
    if y_label is None:
        y_label = data.y_column
    plot_kwargs: Dict[str, Any] = dict(xlabel=x_label, ylabel=y_label, grid=grid)
    if should_plot_data:
        show_or_export(
            plot_data(data=data, title_name=f"{func.title_name} - Data", **plot_kwargs),
            output_path=__optional_path(output_dir, f"{func.name}_data.png"),
        )
    if should_plot_fitting:
        show_or_export(
            plot_fitting(
                func=func,
                data=data,
                a=result.a,
                title_name=f"{func.title_name}",
                **plot_kwargs,
            ),
            output_path=__optional_path(output_dir, f"{func.name}.png"),
        )
    if should_plot_residuals:
        show_or_export(
            plot_residuals(
                func=func,
                data=data,
                a=result.a,
                title_name=f"{func.title_name} - Residuals",
                **plot_kwargs,
            ),
            output_path=__optional_path(output_dir, f"{func.name}_residuals.png"),
        )


def __calc_a0(a0):
    if a0 is None:
        return None
    return np.array(list(map(float, re.split(",[ \t]*", a0))))


def __load_data_file(
    ctx: click.Context, data_file: Path, sheet: Optional[str], **kwargs
):
    suffix = data_file.suffix
    if suffix == ".csv":
        return FitData.read_from_csv(filepath=data_file, **kwargs)
    if suffix == ".json":
        return FitData.read_from_json(filepath=data_file, **kwargs)
    if suffix != ".xlsx":
        click.echo(f'Cannot read data with "{suffix}" suffix')
        ctx.exit(1)
    if sheet is None:
        click.echo("Sheet name has not been specified!")
        ctx.exit(1)
    return FitData.read_from_excel(filepath=data_file, sheet=sheet, **kwargs)


def __optional_path(directory: Optional[Path], file_name: str):
    if directory is None:
        return None
    return directory / file_name
