"""Statistics CLI method."""
from pathlib import Path
from typing import Optional, Union

import click

from eddington.cli.common_flags import data_file_option, output_dir_option, sheet_option
from eddington.cli.main_cli import eddington_cli
from eddington.cli.util import load_data_file

# pylint: disable=too-many-arguments


@eddington_cli.command("statistics")
@click.pass_context
@data_file_option
@sheet_option
@output_dir_option
@click.option("-n", "--file-name", type=str, help="Output file name.")
@click.option(
    "-f",
    "--file-format",
    type=click.Choice(["csv", "xlsx"], case_sensitive=False),
    default="xlsx",
    help="Output file name.",
)
def eddington_statistics(
    ctx: click.Context,
    data_file: str,
    sheet: Optional[str],
    output_dir: Optional[Union[Path, str]],
    file_name: Optional[str],
    file_format: str,
):
    """Print statistics of given data file."""
    data = load_data_file(
        ctx,
        Path(data_file),
        sheet,
    )
    if output_dir is None:
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
        return
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    if file_format == "csv":
        data.save_statistics_csv(output_directory=output_dir, name=file_name)
    if file_format == "xlsx":
        data.save_statistics_excel(output_directory=output_dir, name=file_name)
