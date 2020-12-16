"""CLI for Eddington."""

from eddington.cli.fit import eddington_fit
from eddington.cli.fit_random import eddington_fit_random
from eddington.cli.list import eddington_list
from eddington.cli.main_cli import eddington_cli
from eddington.cli.statistics_cli import eddington_statistics

__all__ = [
    "eddington_cli",
    "eddington_statistics",
    "eddington_list",
    "eddington_fit",
    "eddington_fit_random",
]
