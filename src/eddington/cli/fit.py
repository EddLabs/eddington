"""Fit CLI method."""
from pathlib import Path
from typing import Optional, Union

import click

from eddington.cli.common_flags import (
    data_color_option,
    data_file_option,
    fitting_function_argument,
    is_grid_option,
    is_legend_option,
    is_x_log_scale_option,
    is_y_log_scale_option,
    output_dir_option,
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
from eddington.fitting import fit
from eddington.fitting_result import FittingResult
from eddington.plot.figure_builder import FigureBuilder
from eddington.plot.line_style import LineStyle
from eddington.plot.plot_util import build_repr_string, show_or_export

# pylint: disable=invalid-name,too-many-arguments,too-many-locals


@eddington_cli.command("fit")
@fitting_function_argument
@polynomial_option
@click.option(
    "--a0",
    type=str,
    help=(
        "Initial guess for the fitting algorithm. "
        "Should be given as floating point numbers separated by commas"
    ),
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
@is_legend_option
@click.option("-c", "--color", type=str, help="Color of the fitting plot")
@click.option(
    "--linestyle",
    type=click.Choice(LineStyle.all()),
    help="Line style of the fitting plot",
    callback=lambda ctx, param, value: LineStyle(value),
    default=LineStyle.SOLID.value,
)
@data_color_option
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
@is_x_log_scale_option
@is_y_log_scale_option
@output_dir_option(required=False)
@click.option(
    "--json",
    is_flag=True,
    default=False,
    help="Save result as json instead of text.",
)
def fit_cli(
    fitting_function_name: Optional[str],
    polynomial_degree: Optional[int],
    a0: Optional[str],
    data_file: Union[str, Path],
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
    color: Optional[str],
    linestyle: LineStyle,
    data_color: Optional[str],
    should_plot_fitting: bool,
    should_plot_residuals: bool,
    should_plot_data: bool,
    x_log_scale: bool,
    y_log_scale: bool,
    output_dir: Union[Path, str],
    json: bool,
):
    """Fitting data file according to a fitting function."""
    data_file = Path(data_file)
    data = load_data_file(
        data_file,
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
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    result = fit(data, func, a0=extract_array_from_string(a0))
    write_and_export_result(
        result, func_name=func.name, output_dir=output_dir, is_json=json
    )
    title = func.name.title() if title is None else title
    x_label = data.x_column if x_label is None else x_label
    y_label = data.y_column if y_label is None else y_label
    if should_plot_data:
        data_figure_builder = FigureBuilder(
            title=f"{title} - Data",
            xlabel=x_label,
            ylabel=y_label,
            grid=grid,
            legend=legend,
        )
        if x_log_scale:
            data_figure_builder.add_x_log_scale()
        if y_log_scale:
            data_figure_builder.add_y_log_scale()
        data_figure_builder.add_data(data=data, label=data_file.stem, color=data_color)
        with data_figure_builder.build() as data_fig:
            show_or_export(
                data_fig,
                output_path=__optional_path(output_dir, f"{func.name}_data.png"),
            )
    x_domain = data.x_domain * PLOT_DOMAIN_MULTIPLIER
    a_repr_string = build_repr_string(result.a)
    if should_plot_fitting:
        fitting_figure_builder = FigureBuilder(
            title=title,
            xlabel=x_label,
            ylabel=y_label,
            grid=grid,
            legend=legend,
        )
        if x_log_scale:
            fitting_figure_builder.add_x_log_scale()
        if y_log_scale:
            fitting_figure_builder.add_y_log_scale()
        fitting_figure_builder.add_data(
            data=data, label=data_file.stem, color=data_color
        )
        fitting_figure_builder.add_plot(
            interval=x_domain,
            a=result.a,
            func=func,
            label=a_repr_string,
            color=color,
            linestyle=linestyle,
        )
        with fitting_figure_builder.build() as fit_fig:
            show_or_export(
                fit_fig,
                output_path=__optional_path(output_dir, f"{func.name}.png"),
            )
    if should_plot_residuals:
        residuals_figure_builder = FigureBuilder(
            title=title,
            xlabel=x_label,
            ylabel=y_label,
            grid=grid,
            legend=legend,
        )
        if x_log_scale:
            residuals_figure_builder.add_x_log_scale()
        if y_log_scale:
            residuals_figure_builder.add_y_log_scale()
        residuals_data = data.residuals(fit_func=func, a=result.a)
        residuals_figure_builder.add_data(
            data=residuals_data, label=a_repr_string, color=color
        )
        residuals_figure_builder.add_horizontal_line(
            interval=x_domain, y_value=0.0, linestyle=LineStyle.DASHED, color="k"
        )
        with residuals_figure_builder.build() as residuals_fig:
            show_or_export(
                residuals_fig,
                output_path=__optional_path(output_dir, f"{func.name}_residuals.png"),
            )


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
    if is_json:
        result.save_json(output_dir / f"{func_name}_result.json")
    else:
        result.save_txt(output_dir / f"{func_name}_result.txt")


def __optional_path(directory: Optional[Path], file_name: str):
    if directory is None:
        return None
    return directory / file_name
