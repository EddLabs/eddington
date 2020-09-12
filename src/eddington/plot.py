"""Plotting methods."""
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from eddington.exceptions import PlottingError
from eddington.fitting_data import FittingData
from eddington.print_util import to_precise_string


def plot_residuals(  # pylint: disable=invalid-name,too-many-arguments
    func,
    data: FittingData,
    a: np.ndarray,
    title_name,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    grid: bool = False,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
):
    """
    Plot residuals plot.

    :param func: Fitting function.
    :type func: :class:`FittingFunction`
    :param data: Fitting data
    :type data: :class:`FittingData`
    :param a: The parameters result
    :type a: ``numpy.ndarray`` or ``list``
    :param title_name: Optional. Title for the figure.
    :type title_name: str
    :param xlabel: Optional. Label of the x axis
    :type xlabel: str
    :param ylabel: Optional. Label of the x axis
    :type ylabel: str
    :param grid: Add grid lines or not
    :type grid: bool
    :param xmin: Optional. minimum value for x in plot
    :type xmin: float
    :param xmax: Optional. maximum value for x in plot
    :type xmax: float
    :returns: ``matplotlib.pyplot.Figure``
    """
    fig = plot_data(
        data=data.residuals(func, a),
        title_name=title_name,
        xlabel=xlabel,
        ylabel=ylabel,
        grid=grid,
    )
    xmin, xmax = get_plot_borders(x=data.x, xmin=xmin, xmax=xmax)
    horizontal_line(fig=fig, xmin=xmin, xmax=xmax)
    return fig


def plot_fitting(  # pylint: disable=C0103,R0913,R0914
    func,
    data: FittingData,
    a: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],
    title_name,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    grid: bool = False,
    legend: Optional[bool] = None,
    step: Optional[float] = None,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
):
    """
    Plot fitting plot.

    :param func: Fitting function.
    :type func: :class:`FittingFunction`
    :param data: Fitting data
    :type data: :class:`FittingData`
    :param a: The parameters result
    :type a: ``numpy.ndarray``, a list of ``numpy.ndarray`` items or a dictionary from
     strings to ``numpy.ndarray``
    :param title_name: Optional. Title for the figure.
    :type title_name: str
    :param xlabel: Optional. Label of the x axis
    :type xlabel: str
    :param ylabel: Optional. Label of the x axis
    :type ylabel: str
    :param grid: Add grid lines or not
    :type grid: bool
    :param legend: Add legend or not
    :type legend: bool
    :param step: Optional. Steps between samples for the fitting graph
    :type step: float
    :param xmin: Optional. minimum value for x in plot
    :type xmin: float
    :param xmax: Optional. maximum value for x in plot
    :type xmax: float
    :returns: ``matplotlib.pyplot.Figure``
    """
    fig = plot_data(
        data=data, title_name=title_name, xlabel=xlabel, ylabel=ylabel, grid=grid
    )
    xmin, xmax = get_plot_borders(x=data.x, xmin=xmin, xmax=xmax)
    if step is None:
        step = (xmax - xmin) / 1000.0
    x = np.arange(xmin, xmax, step=step)  # pylint: disable=invalid-name
    a_dict = __get_a_dict(a)
    for label, a_value in a_dict.items():
        plot(fig=fig, x=x, y=func(a_value, x), label=label)
    if __get_legend(legend, a_dict):
        plt.legend()
    return fig


def plot_data(
    data: FittingData,
    title_name,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    grid: bool = False,
):
    """
    Plot fitting data.

    :param data: Fitting data
    :type data: :class:`FittingData`
    :param title_name: Optional. Title for the figure.
    :type title_name: str
    :param xlabel: Optional. Label of the x axis
    :type xlabel: str
    :param ylabel: Optional. Label of the x axis
    :type ylabel: str
    :param grid: Add grid lines or not
    :type grid: bool
    :returns: ``matplotlib.pyplot.Figure``
    """
    fig = get_figure(title_name=title_name, xlabel=xlabel, ylabel=ylabel, grid=grid)
    errorbar(fig=fig, x=data.x, y=data.y, xerr=data.xerr, yerr=data.yerr)
    return fig


def get_figure(
    title_name,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    grid: bool = False,
):
    """
    Gets a figure from matplotlib.

    :param title_name: Optional. Title for the figure.
    :param xlabel: Optional. Label of the x axis
    :param ylabel: Optional. Label of the x axis
    :param grid: Add grid lines or not
    :return: Figure instance
    """
    fig = plt.figure()
    title(fig=fig, title_name=title_name)
    label_axes(fig=fig, xlabel=xlabel, ylabel=ylabel)
    add_grid(fig=fig, is_grid=grid)
    return fig


def title(fig, title_name):
    """
    Add/remove title to figure.

    :param title_name: Optional. If None, don't add title. otherwise, add given title
    :param fig: Plot figure.
    """
    if title_name is not None:
        plt.title(title_name, figure=fig)


def label_axes(fig, xlabel, ylabel):
    """
    Add/remove labels to figure.

    :param fig: Plot figure.
    :param xlabel: Optional. If None, don't add label. otherwise, add given label
    :param ylabel: Optional. If None, don't add label. otherwise, add given label
    """
    if xlabel is not None:
        plt.xlabel(xlabel, figure=fig)
    if ylabel is not None:
        plt.ylabel(ylabel, figure=fig)


def add_grid(fig, is_grid):
    """
    Add/remove grid to figure.

    :param fig: Plot figure
    :param is_grid: Boolean. add or remote grid to plot
    """
    if is_grid:
        plt.grid(True, figure=fig)


def plot(x, y, fig, label=None):  # pylint: disable=C0103
    """
    Plot y as a function of x.

    :param x: X values
    :param y: Y values
    :param fig: Plot figure
    :param label: Optional. Label for the plot that would be added to the legend
    """
    plt.plot(x, y, figure=fig, label=label)


def horizontal_line(  # pylint: disable=C0103
    fig: plt.Figure, xmin: float, xmax: float, y=0
):
    """
    Add horizontal line to figure.

    :param xmin: Minimum x value of line
    :param xmax: Maximum x value of line
    :param y: The y value of the line
    :param fig: Plot figure
    """
    plt.hlines(y, xmin=xmin, xmax=xmax, linestyles="dashed", figure=fig)


def errorbar(fig, x, y, xerr, yerr):  # pylint: disable=C0103
    """
    Plot error bar to figure.

    :param x: X values
    :param y: Y values
    :param xerr: Errors of x
    :param yerr: Errors of y
    :param fig: Plot figure
    """
    plt.errorbar(
        x=x,
        y=y,
        xerr=xerr,
        yerr=yerr,
        markersize=1,
        marker="o",
        linestyle="None",
        figure=fig,
    )


def get_plot_borders(  # pylint: disable=invalid-name
    x: np.ndarray, xmin: Optional[float] = None, xmax: Optional[float] = None
) -> Tuple[float, float]:
    """
    Get borders for a plot based on its x values.

    :param x: x values list or array.
    :type x: ``numpy.ndarray`` or ``list``
    :param xmin: Minimum x value of line
    :type xmin: float
    :param xmax: Maximum x value of line
    :type xmax: float
    :return: tuple. minimum and maximum values for the plot.
    """
    data_xmin = np.min(x)
    data_xmax = np.max(x)
    gap = (data_xmax - data_xmin) * 0.1
    if xmin is None:
        xmin = data_xmin - gap
    if xmax is None:
        xmax = data_xmax + gap
    return xmin, xmax


def show_or_export(fig: plt.Figure, output_path=None):
    """
    Show plot or export it to a file.

    :param fig: a plot figure
    :param output_path: Path or None. if None, show plot. otherwise, save to path.
    """
    if output_path is None:
        plt.show()
        return
    fig.savefig(output_path)


def __get_a_dict(a):  # pylint: disable=invalid-name
    if isinstance(a, dict):
        return a
    if isinstance(a, list):
        return {__build_repr_string(a_value): a_value for a_value in a}
    if isinstance(a, np.ndarray):
        return {__build_repr_string(a): a}
    raise PlottingError(
        f"{a} has unmatching type. Can except only numpy arrays, "
        "lists of numpy arrays and dictionaries."
    )


def __build_repr_string(a):  # pylint: disable=invalid-name
    arguments_values = [f"a[{i}]={to_precise_string(val)}" for i, val in enumerate(a)]
    return f"[{', '.join(arguments_values)}]"


def __get_legend(legend, a):  # pylint: disable=invalid-name
    if legend is not None:
        return legend
    if len(a) >= 2:
        return True
    return False
