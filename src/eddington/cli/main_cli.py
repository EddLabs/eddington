"""Main CLI method."""
import sys

import click

from eddington import EddingtonException, __version__


class CatchEddingtonExceptions(click.Group):
    """Implementation for a commands group that catches Eddington exceptions."""

    def __call__(self, *args, **kwargs):
        """
        Override of the call method that catches Eddington exceptions.

        :param args: Arguments for the subcommand
        :type args: list
        :param kwargs: Keyword argument for the subcommand.
        :type kwargs: dict
        :return: Returns the output of the subcommand
        """
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
