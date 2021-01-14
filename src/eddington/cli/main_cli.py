"""Main CLI method."""
import sys

import click

from eddington import __version__, EddingtonException


class CatchEddingtonExceptions(click.Group):

    def __call__(self, *args, **kwargs):
        try:
            return self.main(*args, **kwargs)
        except EddingtonException as exc:
            click.echo(
                f'{click.style(exc.__class__.__name__, fg="yellow")}: '
                f'{click.style(exc, fg="red")}'
            )
            sys.exit(1)


@click.group("eddington", cls=CatchEddingtonExceptions)
@click.version_option(version=__version__)
def eddington_cli():
    """Command line for Eddington."""
