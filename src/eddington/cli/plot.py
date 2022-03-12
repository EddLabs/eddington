"""Plot CLI method."""
from pathlib import Path
from typing import List, Optional, Tuple, Union

import click

from eddington import show_or_export
from eddington.cli.common_flags import (
    data_color_option,
    data_file_option,
    fitting_function_argument,
    is_grid_option,
    is_legend_option,
    is_x_log_scale_option,
    is_y_log_scale_option,
    polynomial_option,
    search_option,
    sheet_option,
    title_option,
    x_column_option,
    x_label_option,
    xerr_column_option,
    y_column_option,
    y_label_option,
    yerr_column_option,
)
from eddington.cli.main_cli import eddington_cli
from eddington.cli.util import (
    extract_array_from_string,
    load_data_file,
    load_fitting_function,
)
from eddington.consts import PLOT_DOMAIN_MULTIPLIER
from eddington.plot.figure_builder import FigureBuilder
from eddington.plot.line_style import LineStyle

# pylint: disable=invalid-name,too-many-arguments,too-many-locals
from eddington.plot.plot_util import build_repr_string


@eddington_cli.command("plot")
@fitting_function_argument
@polynomial_option
@click.option(
    "-p",
    "--parameters",
    type=str,
    help=(
        "Parameters values for the fitting function. "
        "Should be given as floating point numbers separated by commas"
    ),
    multiple=True,
)
@data_file_option
@sheet_option
@x_column_option
@xerr_column_option
@y_column_option
@yerr_column_option
@search_option
@title_option
@x_label_option
@y_label_option
@is_grid_option
@click.option(
    "-c", "--color", "colors", type=str, help="Color of the fitting plot", multiple=True
)
@click.option(
    "--linestyle",
    "linestyles",
    type=click.Choice(LineStyle.all()),
    help="Line style of the fitting plot",
    multiple=True,
    callback=lambda ctx, param, values: [LineStyle(value) for value in values],
)
@data_color_option
@is_legend_option
@is_x_log_scale_option
@is_y_log_scale_option
@click.option(
    "-o",
    "--output-path",
    type=click.Path(dir_okay=False, file_okay=True),
    help="Output path to save plots in.",
)
@click.option(
    "-r",
    "--residuals",
    is_flag=True,
    help="Show residuals error bars instead of plots.",
)
def plot_cli(
    fitting_function_name: Optional[str],
    polynomial_degree: Optional[int],
    parameters: List[str],
    data_file: str,
    sheet: Optional[str],
    x_column: Optional[str],
    xerr_column: Optional[str],
    y_column: Optional[str],
    yerr_column: Optional[str],
    search: bool,
    title: Optional[str],
    x_label: Optional[str],
    y_label: Optional[str],
    grid: bool,
    legend: bool,
    colors: Union[List[Optional[str]], Tuple[Optional[str], ...]],
    linestyles: Union[List[LineStyle], Tuple[LineStyle, ...]],
    data_color: Optional[str],
    x_log_scale: bool,
    y_log_scale: bool,
    output_path: Union[Path, str],
    residuals: bool,
):
    """Plot fitting functions according to data file and a parameters set."""
    data = load_data_file(
        Path(data_file),
        sheet=sheet,
        x_column=x_column,
        xerr_column=xerr_column,
        y_column=y_column,
        yerr_column=yerr_column,
        search=search,
    )
    func = load_fitting_function(
        func_name=fitting_function_name, polynomial_degree=polynomial_degree
    )
    parameters_sets = [extract_array_from_string(a0) for a0 in parameters]
    colors, linestyles = list(colors), list(linestyles)
    if len(colors) < len(parameters_sets):
        colors += [None for _ in range(len(parameters_sets) - len(colors))]
    if len(linestyles) < len(parameters_sets):
        linestyles += [
            LineStyle.SOLID for _ in range(len(parameters_sets) - len(linestyles))
        ]
    figure_builder = FigureBuilder(
        title=title, xlabel=x_label, ylabel=y_label, legend=legend, grid=grid
    )
    if x_log_scale:
        figure_builder.add_x_log_scale()
    if y_log_scale:
        figure_builder.add_y_log_scale()
    if not residuals:
        figure_builder.add_data(data=data, color=data_color)
    x_domain = data.x_domain * PLOT_DOMAIN_MULTIPLIER
    for a0, color, linestyle in zip(parameters_sets, colors, linestyles):
        if a0 is None:
            continue
        label = build_repr_string(a0)
        if residuals:
            residuals_data = data.residuals(fit_func=func, a=a0)
            figure_builder.add_data(data=residuals_data, label=label, color=color)
        else:
            figure_builder.add_plot(
                interval=x_domain,
                func=func,
                a=a0,
                label=label,
                color=color,
                linestyle=linestyle,
            )

    show_or_export(fig=figure_builder.build(), output_path=output_path)
