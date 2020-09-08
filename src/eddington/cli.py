"""CLI for Eddington."""
import re
from pathlib import Path
from typing import Optional

import click
from prettytable import PrettyTable

from eddington import (
    FitData,
    FitFunctionsRegistry,
    __version__,
    fit_to_data,
    plot_fitting,
    show_or_export,
)

# pylint: disable=too-many-arguments


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
@click.option("--x-column", type=str, help="Column to read x values from.")
@click.option("--xerr-column", type=str, help="Column to read x error values from.")
@click.option("--y-column", type=str, help="Column to read y values from.")
@click.option("--yerr-column", type=str, help="Column to read y error values from.")
@click.option(
    "--plot-fitting/--no-plot-fitting",
    "should_plot_fitting",
    default=True,
    help="Should plot fitting",
)
def eddington_fit(
    ctx: click.Context,
    fit_func: Optional[str],
    data_file: str,
    sheet: Optional[str],
    x_column: Optional[str],
    xerr_column: Optional[str],
    y_column: Optional[str],
    yerr_column: Optional[str],
    should_plot_fitting: bool,
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
    result = fit_to_data(data, func)
    click.echo(result.pretty_string)
    if should_plot_fitting:
        fig = plot_fitting(
            func=func, data=data, a=result.a, title_name=f"{func.title_name}"
        )
        show_or_export(fig)


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
