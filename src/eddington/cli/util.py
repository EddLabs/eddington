"""Utility methods for the Eddington CLI."""
import re
from pathlib import Path
from typing import Optional

import click
import numpy as np

from eddington import FittingDataInvalidFile, FittingFunction
from eddington.exceptions import EddingtonCLIError
from eddington.fitting import fit
from eddington.fitting_data import FittingData
from eddington.fitting_functions_list import linear, polynomial
from eddington.fitting_functions_registry import FittingFunctionsRegistry
from eddington.fitting_result import FittingResult
from eddington.plot.plot_legacy import (
    LineStyle,
    plot_data,
    plot_fitting,
    plot_residuals,
    show_or_export,
)


def load_data_file(data_file: Path, **kwargs) -> FittingData:
    """
    Load data file with any suffix.

    :param data_file: The type of the file to be loaded.
    :type data_file: Path
    :param kwargs: Keyword arguments for the actual reading method
    :type kwargs: dict
    :return: FittingData
    :raises FittingDataInvalidFile: Given an unknown file suffix, raise exception.
    """
    suffix = data_file.suffix
    if suffix == ".xlsx":
        return FittingData.read_from_excel(filepath=data_file, **kwargs)
    if "sheet" in kwargs:
        del kwargs["sheet"]
    if suffix == ".csv":
        return FittingData.read_from_csv(filepath=data_file, **kwargs)
    if suffix == ".json":
        return FittingData.read_from_json(filepath=data_file, **kwargs)
    raise FittingDataInvalidFile(f"Cannot read fitting data from a {suffix} file.")


def load_fitting_function(
    func_name: Optional[str], polynomial_degree: Optional[int]
) -> FittingFunction:
    """
    Load fitting function.

    If function name is given, get from registry.
    If polynomial degree is given, get polynomial with given degree.
    If none are given, returns linear.

    :param func_name: Function name to get from registry
    :type func_name: Optional[str]
    :param polynomial_degree: Degree of the polynomial.
    :type polynomial_degree: Optional[int]
    :return: Matching fitting function.
    :rtype: FittingFunction
    :raises EddingtonCLIError: If both function name and polynomial degree are given
        raise an exception.

    """
    if func_name is None or func_name.strip() == "":
        if polynomial_degree is not None:
            return polynomial(polynomial_degree)
        return linear
    if polynomial_degree is not None:
        raise EddingtonCLIError("Cannot accept both polynomial and fitting function")
    return FittingFunctionsRegistry.load(func_name)


def fit_and_plot(  # pylint: disable=too-many-arguments,invalid-name,too-many-locals
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
    color=None,
    linestyle=LineStyle.SOLID,
    data_color=None,
    **plot_kwargs,
):
    """
    Plot everything needed.

    # noqa: DAR101
    """
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
            color=data_color,
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
            color=color,
            linestyle=linestyle,
            data_color=data_color,
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
            color=color,
            **plot_kwargs,
        ) as fig:
            show_or_export(
                fig,
                output_path=__optional_path(output_dir, f"{func.name}_residuals.png"),
            )


def extract_array_from_string(  # pylint: disable=invalid-name
    a0: Optional[str],
) -> Optional[np.ndarray]:
    """
    Split a0 string to an array.

    :param a0: Initial guess values separated by commas.
    :type a0: Optional[str]
    :return: Parsed a0 or None.
    :rtype: Optional[np.ndarray]
    """
    if a0 is None:
        return None
    return np.array(list(map(float, re.split(",[ \t]*", a0))))


def write_and_export_result(
    result: FittingResult,
    func_name: str,
    output_dir: Optional[Path] = None,
    is_json: bool = False,
) -> None:
    """
    Write result to console and to file if specified so.

    :param result: Fitting result to be written to console or saved
    :type result: FittingResult
    :param func_name: Name of the fitting function.
    :type func_name: str
    :param output_dir: Optional output directory to save result in.
    :type output_dir: Optional[Path]
    :param is_json: Save in json format or txt format
    :type is_json: bool
    """
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
