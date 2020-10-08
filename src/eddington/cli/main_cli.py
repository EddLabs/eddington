"""Main CLI method."""

import click

from eddington import __version__


@click.group("eddington")
@click.version_option(version=__version__)
def eddington_cli():
    """Command line for Eddington."""
