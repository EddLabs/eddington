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
x_column_option = click.option(
    "--x-column", type=str, help="Column to read x values from."
)

xerr_column_option = click.option(
    "--xerr-column", type=str, help="Column to read x error values from."
)

y_column_option = click.option(
    "--y-column", type=str, help="Column to read y values from."
)

yerr_column_option = click.option(
    "--yerr-column", type=str, help="Column to read y error values from."
)
search_option = click.option(
    "--search/--no-search",
    is_flag=True,
    default=True,
    help="Search missing columns if weren't specified.",
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


# Plot

title_option = click.option("--title", type=str, help="Title for the fitting plot.")

x_label_option = click.option("--x-label", type=str, help="Label for the x axis.")

y_label_option = click.option("--y-label", type=str, help="Label for the y axis.")

is_grid_option = click.option(
    "--grid/--no-grid", default=False, help="Add grid lines to plots."
)

is_legend_option = click.option(
    "--legend/--no-legend",
    default=False,
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

data_color_option = click.option(
    "--dcolor",
    "--data-color",
    "data_color",
    type=str,
    help="Color of the data error bar.",
)


# Output
def output_dir_option(required: bool):
    """
    Add click option of output directory.

    :param required: is this option required
    :type required: bool
    :return: click option
    """
    return click.option(
        "-o",
        "--output-dir",
        type=click.Path(dir_okay=True, file_okay=False),
        required=required,
        help="Output directory to save in.",
    )
