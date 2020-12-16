"""Utility methods for the Eddington CLI."""
import re
from pathlib import Path
from typing import Optional

import click
import numpy as np

from eddington.fitting import fit
from eddington.fitting_data import FittingData
from eddington.fitting_functions_list import linear, polynomial
from eddington.fitting_functions_registry import FittingFunctionsRegistry
from eddington.fitting_result import FittingResult
from eddington.plot import plot_data, plot_fitting, plot_residuals, show_or_export


def load_data_file(ctx: click.Context, data_file: Path, sheet: Optional[str], **kwargs):
    """Load data file with any suffix."""
    suffix = data_file.suffix
    if suffix == ".csv":
        return FittingData.read_from_csv(filepath=data_file, **kwargs)
    if suffix == ".json":
        return FittingData.read_from_json(filepath=data_file, **kwargs)
    if suffix != ".xlsx":
        click.echo(f'Cannot read data with "{suffix}" suffix')
        ctx.exit(1)
    if sheet is None:
        click.echo("Sheet name has not been specified!")
        ctx.exit(1)
    return FittingData.read_from_excel(filepath=data_file, sheet=sheet, **kwargs)


def load_fitting_function(
    ctx: click.Context, func_name: Optional[str], polynomial_degree: Optional[int]
):
    """Load appropriate fitting function."""
    if func_name == "":
        if polynomial_degree is not None:
            return polynomial(polynomial_degree)
        return linear
    if polynomial_degree is not None:
        click.echo("Cannot accept both polynomial and fitting function")
        ctx.exit(1)
    return FittingFunctionsRegistry.load(func_name)


def fit_and_plot(  # pylint: disable=too-many-arguments,invalid-name
    data,
    func,
    a0,
    legend,
    output_dir,
    is_json,
    title,
    x_label,
    y_label,
    should_plot_data,
    should_plot_fitting,
    should_plot_residuals,
    **plot_kwargs,
):
    """Plot everything needed."""
    output_dir = None if output_dir is None else Path(output_dir)
    result = fit(data, func, a0=a0)
    write_and_export_result(
        result, func_name=func.name, output_dir=output_dir, is_json=is_json
    )
    title = func.name.title() if title is None else title
    x_label = data.x_column if x_label is None else x_label
    y_label = data.y_column if y_label is None else y_label
    if should_plot_data:
        with plot_data(
            data=data,
            title_name=f"{title} - Data",
            xlabel=x_label,
            ylabel=y_label,
            **plot_kwargs,
        ) as fig:
            show_or_export(
                fig, output_path=__optional_path(output_dir, f"{func.name}_data.png")
            )
    if should_plot_fitting:
        with plot_fitting(
            func=func,
            data=data,
            a=result.a,
            title_name=f"{title}",
            legend=legend,
            xlabel=x_label,
            ylabel=y_label,
            **plot_kwargs,
        ) as fig:
            show_or_export(
                fig, output_path=__optional_path(output_dir, f"{func.name}.png")
            )
    if should_plot_residuals:
        with plot_residuals(
            func=func,
            data=data,
            a=result.a,
            title_name=f"{title} - Residuals",
            xlabel=x_label,
            ylabel=y_label,
            **plot_kwargs,
        ) as fig:
            show_or_export(
                fig,
                output_path=__optional_path(output_dir, f"{func.name}_residuals.png"),
            )


def extract_array_from_string(a0: Optional[str]):  # pylint: disable=invalid-name
    """Split a0 string to an array."""
    if a0 is None:
        return None
    return np.array(list(map(float, re.split(",[ \t]*", a0))))


def write_and_export_result(
    result: FittingResult,
    func_name: str,
    output_dir: Optional[Path] = None,
    is_json: bool = False,
):
    """Write result to console and to file if specified so."""
    click.echo(result.pretty_string)
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    if is_json:
        result.save_json(output_dir / f"{func_name}_result.json")
    else:
        result.save_txt(output_dir / f"{func_name}_result.txt")


def __optional_path(directory: Optional[Path], file_name: str):
    if directory is None:
        return None
    return directory / file_name
