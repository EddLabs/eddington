"""list CLI method."""
import re
from typing import Optional

import click
from prettytable import PrettyTable

from eddington import FittingFunctionsRegistry
from eddington.cli.main_cli import eddington_cli


@eddington_cli.command("list")
@click.option(
    "-r",
    "--regex",
    type=str,
    default=None,
    help="Filter functions by a regular expression",
)
def eddington_list(regex: Optional[str]):
    """Prints all fitting functions in a pretty table."""
    table = PrettyTable(field_names=["Function", "Syntax"])
    for func in FittingFunctionsRegistry.all():
        if regex is None or re.search(regex, func.name):
            table.add_row([func.name, func.syntax])
    click.echo(table)
