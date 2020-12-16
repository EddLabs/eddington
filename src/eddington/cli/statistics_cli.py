"""Statistics CLI method."""
from pathlib import Path
from typing import Optional

import click

from eddington.cli.common_flags import data_file_option, sheet_option
from eddington.cli.main_cli import eddington_cli
from eddington.cli.util import load_data_file


@eddington_cli.command("statistics")
@click.pass_context
@data_file_option
@sheet_option
def eddington_statistics(ctx: click.Context, data_file: str, sheet: Optional[str]):
    """Print statistics of given data file."""
    data = load_data_file(
        ctx,
        Path(data_file),
        sheet,
    )
    click.echo(f'Data statistics of "{data_file}"')
    for column in data.all_columns:
        click.echo(f"{column}:")
        column_statistics = data.statistics(column)
        click.echo(f"\tMean: {column_statistics.mean}")
        click.echo(f"\tMedian: {column_statistics.median}")
        click.echo(f"\tVariance: {column_statistics.variance}")
        click.echo(f"\tStandard Deviation: {column_statistics.standard_deviation}")
        click.echo(f"\tMinimum: {column_statistics.minimum_value}")
        click.echo(f"\tMaximum: {column_statistics.maximum_value}")
