"""Plot CLI method."""
from pathlib import Path
from typing import List, Optional, Union

import click

from eddington.cli.common_flags import (
    data_file_option,
    fitting_function_argument,
    is_grid_option,
    is_legend_option,
    is_x_log_scale_option,
    is_y_log_scale_option,
    polynomial_option,
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
from eddington.plot import (
    add_errorbar,
    add_legend,
    add_plot,
    build_repr_string,
    get_checkers_list,
    get_figure,
    get_plot_borders,
    get_x_plot_values,
    show_or_export,
)

# pylint: disable=invalid-name,too-many-arguments,too-many-locals,duplicate-code


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
@title_option
@x_label_option
@y_label_option
@is_grid_option
@is_legend_option
@is_x_log_scale_option
@is_y_log_scale_option
@click.option(
    "-o",
    "--output-path",
    type=click.Path(dir_okay=False, file_okay=True),
    help="Output path to save plots in.",
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
    title: Optional[str],
    x_label: Optional[str],
    y_label: Optional[str],
    grid,
    legend: Optional[bool],
    x_log_scale: bool,
    y_log_scale: bool,
    output_path: Union[Path, str],
):
    """Plot fitting functions according to data file and a parameters set."""
    data = load_data_file(
        Path(data_file),
        sheet=sheet,
        x_column=x_column,
        xerr_column=xerr_column,
        y_column=y_column,
        yerr_column=yerr_column,
    )
    func = load_fitting_function(
        func_name=fitting_function_name, polynomial_degree=polynomial_degree
    )
    parameters_sets = [extract_array_from_string(a0) for a0 in parameters]
    ax, figure = get_figure(
        title_name=title,
        xlabel=x_label,
        ylabel=y_label,
        grid=grid,
        x_log_scale=x_log_scale,
        y_log_scale=y_log_scale,
    )
    xmin, xmax = get_plot_borders(x=data.x)
    checkers_list = get_checkers_list(values=data.x, min_val=xmin, max_val=xmax)
    add_errorbar(
        ax=ax,
        x=data.x[checkers_list],
        xerr=data.xerr[checkers_list],
        y=data.y[checkers_list],
        yerr=data.yerr[checkers_list],
    )
    x_values = get_x_plot_values(xmin, xmax)
    for a0 in parameters_sets:
        if a0 is None:
            continue
        add_plot(ax=ax, x=x_values, y=func(a0, x_values), label=build_repr_string(a0))
    if len(parameters_sets) != 0 and legend:
        add_legend(ax=ax, is_legend=True)

    show_or_export(fig=figure, output_path=output_path)
