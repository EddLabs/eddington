"""Fit CLI method."""
from pathlib import Path
from typing import Optional, Union

import click

from eddington.cli.common_flags import (
    a0_option,
    data_file_option,
    fitting_function_argument,
    is_grid_option,
    is_json_option,
    is_legend_option,
    is_x_log_scale_option,
    is_y_log_scale_option,
    output_dir_option,
    polynomial_option,
    sheet_option,
    should_plot_data_option,
    should_plot_fitting_option,
    should_plot_residuls_option,
    title_option,
    x_label_option,
    y_label_option,
)
from eddington.cli.main_cli import eddington_cli
from eddington.cli.util import (
    extract_array_from_string,
    fit_and_plot,
    load_data_file,
    load_fitting_function,
)

# pylint: disable=invalid-name,too-many-arguments,too-many-locals,duplicate-code


@eddington_cli.command("fit")
@click.pass_context
@fitting_function_argument
@polynomial_option
@a0_option
@data_file_option
@sheet_option
@click.option("--x-column", type=str, help="Column to read x values from.")
@click.option("--xerr-column", type=str, help="Column to read x error values from.")
@click.option("--y-column", type=str, help="Column to read y values from.")
@click.option("--yerr-column", type=str, help="Column to read y error values from.")
@title_option
@x_label_option
@y_label_option
@is_grid_option
@should_plot_fitting_option
@should_plot_residuls_option
@should_plot_data_option
@is_legend_option
@is_x_log_scale_option
@is_y_log_scale_option
@output_dir_option
@is_json_option
def eddington_fit(
    ctx: click.Context,
    fitting_function_name: Optional[str],
    polynomial_degree: Optional[int],
    a0: Optional[str],
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
    should_plot_fitting: bool,
    should_plot_residuals: bool,
    should_plot_data: bool,
    legend: Optional[bool],
    x_log_scale: bool,
    y_log_scale: bool,
    output_dir: Union[Path, str],
    json: bool,
):
    """Fitting data file according to a fitting function."""
    data = load_data_file(
        ctx,
        Path(data_file),
        sheet,
        x_column=x_column,
        xerr_column=xerr_column,
        y_column=y_column,
        yerr_column=yerr_column,
    )
    func = load_fitting_function(
        ctx=ctx, func_name=fitting_function_name, polynomial_degree=polynomial_degree
    )
    fit_and_plot(
        data=data,
        func=func,
        a0=extract_array_from_string(a0),
        legend=legend,
        output_dir=output_dir,
        is_json=json,
        title=title,
        x_label=x_label,
        y_label=y_label,
        should_plot_data=should_plot_data,
        should_plot_fitting=should_plot_fitting,
        should_plot_residuals=should_plot_residuals,
        x_log_scale=x_log_scale,
        y_log_scale=y_log_scale,
        grid=grid,
    )
