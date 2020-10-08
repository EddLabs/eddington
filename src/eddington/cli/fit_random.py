"""Fit-random CLI method."""
from pathlib import Path
from typing import Optional, Union

import click

from eddington.cli.common_flags import (
    a0_option,
    fitting_function_argument,
    is_grid_option,
    is_json_option,
    is_legend_option,
    is_x_log_scale_option,
    is_y_log_scale_option,
    output_dir_option,
    polynomial_option,
    should_plot_data_option,
    should_plot_fitting_option,
    should_plot_residuls_option,
    x_label_option,
    y_label_option,
)
from eddington.cli.main_cli import eddington_cli
from eddington.cli.util import calculate_a0, fit_and_plot, load_fitting_function
from eddington.fitting_data import FittingData

# pylint: disable=invalid-name,too-many-arguments,too-many-locals,duplicate-code


@eddington_cli.command("fit-random")
@click.pass_context
@fitting_function_argument
@polynomial_option
@a0_option
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
def eddington_fit_random(
    ctx: click.Context,
    fitting_function_name: Optional[str],
    polynomial_degree: Optional[int],
    a0: Optional[str],
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
    """
    Fitting random data using the Eddington fitting algorithm.

    This is best used for testing fitting functions.
    """
    func = load_fitting_function(
        ctx=ctx, func_name=fitting_function_name, polynomial_degree=polynomial_degree
    )
    data = FittingData.random(func)
    fit_and_plot(
        data=data,
        func=func,
        a0=calculate_a0(a0),
        legend=legend,
        output_dir=output_dir,
        is_json=json,
        x_label=x_label,
        y_label=y_label,
        x_log_scale=x_log_scale,
        y_log_scale=y_log_scale,
        grid=grid,
        should_plot_data=should_plot_data,
        should_plot_fitting=should_plot_fitting,
        should_plot_residuals=should_plot_residuals,
    )
