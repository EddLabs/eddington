import re
from typing import Optional

from prettytable import PrettyTable

from eddington import __version__, FitFunctionsRegistry

import click


@click.group("eddington")
@click.version_option(version=__version__)
def eddington_cli():
    """Command line for Eddington."""


@eddington_cli.command("list")
@click.option(
    "-r",
    "--regex",
    type=str,
    default=None,
    help="Filter functions by a regular expression",
)
def eddington_list(regex: Optional[str]):
    """Prints all fit functions in a pretty table."""
    table = PrettyTable(field_names=["Function", "Syntax"])
    for func in FitFunctionsRegistry.all():
        if regex is None or re.search(regex, func.name):
            table.add_row([func.signature, func.syntax])
    click.echo(table)
