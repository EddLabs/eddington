"""Fit-random CLI method."""
from pathlib import Path
from typing import Optional, Union

import click

from eddington.cli.common_flags import (
    fitting_function_argument,
    output_dir_option,
    polynomial_option,
)
from eddington.cli.main_cli import eddington_cli
from eddington.cli.util import extract_array_from_string, load_fitting_function
from eddington.consts import (
    DEFAULT_MAX_COEFF,
    DEFAULT_MIN_COEFF,
    DEFAULT_XMAX,
    DEFAULT_XMIN,
    DEFAULT_XSIGMA,
    DEFAULT_YSIGMA,
)
from eddington.random_util import random_data

# pylint: disable=too-many-arguments


@eddington_cli.command("generate-data")
@fitting_function_argument
@polynomial_option
@click.option(
    "--a",
    type=str,
    help="Actual parameters of the randomized data.",
)
@click.option(
    "--random-x-min",
    type=float,
    default=DEFAULT_XMIN,
    help="Minimum x value for the random data",
)
@click.option(
    "--random-x-max",
    type=float,
    default=DEFAULT_XMAX,
    help="Maximum x value for the random data",
)
@click.option(
    "--x-sigma",
    type=float,
    default=DEFAULT_XSIGMA,
    help="Standard deviation of the random x values",
)
@click.option(
    "--y-sigma",
    type=float,
    default=DEFAULT_YSIGMA,
    help="Standard deviation of the random y values",
)
@click.option(
    "--min-coeff",
    type=float,
    default=DEFAULT_MIN_COEFF,
    help="Minimum value for the parameters",
)
@click.option(
    "--max-coeff",
    type=float,
    default=DEFAULT_MAX_COEFF,
    help="Maximum value for the parameters",
)
@output_dir_option(required=True)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["csv", "excel"]),
    default="csv",
    help="Format to save the data",
)
def generate_data_cli(
    fitting_function_name: Optional[str],
    polynomial_degree: Optional[int],
    a: Optional[str],
    random_x_min: float,
    random_x_max: float,
    min_coeff: float,
    max_coeff: float,
    x_sigma: float,
    y_sigma: float,
    output_dir: Union[str, Path],
    output_format: str,
):
    """
    Fitting random data using the Eddington fitting algorithm.

    This is best used for testing fitting functions.
    """
    func = load_fitting_function(
        func_name=fitting_function_name, polynomial_degree=polynomial_degree
    )
    data = random_data(
        func,
        xmin=random_x_min,
        xmax=random_x_max,
        min_coeff=min_coeff,
        max_coeff=max_coeff,
        xsigma=x_sigma,
        ysigma=y_sigma,
        a=extract_array_from_string(a),
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_format == "csv":
        data.save_csv(output_directory=output_dir, name="random_data")
    if output_format == "excel":
        data.save_excel(output_directory=output_dir, name="random_data")
