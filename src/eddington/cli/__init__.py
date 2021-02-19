"""CLI for Eddington."""

from eddington.cli.fit import fit_cli
from eddington.cli.generate_data import generate_data_cli
from eddington.cli.list import list_cli
from eddington.cli.main_cli import eddington_cli
from eddington.cli.plot import plot_cli
from eddington.cli.statistics_cli import statistics_cli

__all__ = [
    "eddington_cli",
    "statistics_cli",
    "list_cli",
    "fit_cli",
    "generate_data_cli",
    "plot_cli",
]
