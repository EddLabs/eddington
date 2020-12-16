"""List of common flags in use of the Eddington CLI."""
import click

# Fitting


data_file_option = click.option(
    "-d",
    "--data-file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    help="Data file to read from.",
)
sheet_option = click.option(
    "-s", "--sheet", type=str, help="Sheet name for excel files."
)
fitting_function_argument = click.argument(
    "fitting_function_name", type=str, default=""
)
polynomial_option = click.option(
    "-p",
    "--polynomial",
    "polynomial_degree",
    type=int,
    help="Fitting data to polynomial of nth degree.",
)
a0_option = click.option(
    "--a0",
    type=str,
    help=(
        "Initial guess for the fitting algorithm. "
        "Should be given as floating point numbers separated by commas"
    ),
)


# Plot

title_option = click.option("--title", type=str, help="Title for the fitting plot.")

x_label_option = click.option("--x-label", type=str, help="Label for the x axis.")

y_label_option = click.option("--y-label", type=str, help="Label for the y axis.")

is_grid_option = click.option(
    "--grid/--no-grid", default=False, help="Add grid lines to plots."
)
should_plot_fitting_option = click.option(
    "--plot-fitting/--no-plot-fitting",
    "should_plot_fitting",
    default=True,
    help="Should plot fitting.",
)

should_plot_residuls_option = click.option(
    "--plot-residuals/--no-plot-residuals",
    "should_plot_residuals",
    default=True,
    help="Should plot residuals.",
)

should_plot_data_option = click.option(
    "--plot-data/--no-plot-data",
    "should_plot_data",
    default=False,
    help="Should plot data.",
)

is_legend_option = click.option(
    "--legend/--no-legend",
    default=None,
    help="Should add legend to fitting plot.",
)

is_x_log_scale_option = click.option(
    "--x-log-scale/--no-x-log-scale",
    default=False,
    help="Change x axis scale to logarithmic.",
)

is_y_log_scale_option = click.option(
    "--y-log-scale/--no-y-log-scale",
    default=False,
    help="Change y axis scale to logarithmic.",
)


# Output
output_dir_option = click.option(
    "-o",
    "--output-dir",
    type=click.Path(dir_okay=True, file_okay=False),
    help="Output directory to save plots in.",
)

is_json_option = click.option(
    "--json",
    is_flag=True,
    default=False,
    help="Save result as json instead of text.",
)
