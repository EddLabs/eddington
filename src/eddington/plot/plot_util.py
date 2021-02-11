"""Module for general plot utility functions."""
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np

from eddington.plot.figure import Figure
from eddington.print_util import to_relevant_precision_string


def show_or_export(fig: Figure, output_path=None):
    """
    Show plot or export it to a file.

    :param fig: a plot figure
    :param output_path: Path or None. if None, show plot. otherwise, save to path.
    """
    if output_path is None:
        plt.show()
        return
    fig.savefig(output_path)


def build_repr_string(parameters: Union[List[float], np.ndarray]) -> str:
    """
    Format parameters array into representation string.

    :param parameters: Array of parameters to be formatted
    :type parameters: List of floats or numpy.ndarray
    :return: Formatted string
    :rtype: str
    """
    arguments_values = [
        f"a[{i}]={to_relevant_precision_string(val)}"
        for i, val in enumerate(parameters)
    ]
    return f"[{', '.join(arguments_values)}]"
