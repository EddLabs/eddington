"""Plotting methods."""
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from eddington.exceptions import PlottingError
from eddington.fitting_data import FittingData
from eddington.print_util import to_relevant_precision_string


class Figure:
    """
    Wraps matplotlib Figure class.

    It releases the memory when the figure is not longer in use.
    """

    def __init__(self, fig):
        """Figure constructor."""
        self._actual_fig = fig

    def __enter__(self):
        """Return self when entering as context."""
        return self

    def __getattr__(self, item):
        """Get attributes from wrapped figure."""
        return getattr(self._actual_fig, item)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clear memory on exit."""
        plt.clf()
        plt.close("all")


def plot_residuals(  # pylint: disable=invalid-name,too-many-arguments
    func,
    data: FittingData,
    a: np.ndarray,
    title_name,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    grid: bool = False,
    x_log_scale: bool = False,
    y_log_scale: bool = False,
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
    :param x_log_scale: Set the scale of the  x axis to be logarithmic.
    :type x_log_scale: bool
    :param y_log_scale: Set the scale of the y axis to be logarithmic.
    :type y_log_scale: bool
    :param xmin: Optional. minimum value for x in plot
    :type xmin: float
    :param xmax: Optional. maximum value for x in plot
    :type xmax: float
    :returns: ``matplotlib.pyplot.Figure``
    """
    ax, fig = get_figure(
        title_name=title_name,
        xlabel=xlabel,
        ylabel=ylabel,
        grid=grid,
        x_log_scale=x_log_scale,
        y_log_scale=y_log_scale,
    )
    errorbar(ax=ax, data=data.residuals(func, a))
    xmin, xmax = get_plot_borders(x=data.x, xmin=xmin, xmax=xmax)
    horizontal_line(ax=ax, xmin=xmin, xmax=xmax)
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
    x_log_scale: bool = False,
    y_log_scale: bool = False,
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
    :param legend: Optional. Add legend or not. If None, add legend when more than
     one parameters values has been presented.
    :type legend: bool
    :param x_log_scale: Set the scale of the  x axis to be logarithmic.
    :type x_log_scale: bool
    :param y_log_scale: Set the scale of the y axis to be logarithmic.
    :type y_log_scale: bool
    :param step: Optional. Steps between samples for the fitting graph
    :type step: float
    :param xmin: Optional. minimum value for x in plot
    :type xmin: float
    :param xmax: Optional. maximum value for x in plot
    :type xmax: float
    :returns: ``matplotlib.pyplot.Figure``
    """
    ax, fig = get_figure(
        title_name=title_name,
        xlabel=xlabel,
        ylabel=ylabel,
        grid=grid,
        x_log_scale=x_log_scale,
        y_log_scale=y_log_scale,
    )
    errorbar(ax=ax, data=data)
    xmin, xmax = get_plot_borders(x=data.x, xmin=xmin, xmax=xmax)
    if step is None:
        step = (xmax - xmin) / 1000.0
    x = np.arange(xmin, xmax, step=step)  # pylint: disable=invalid-name
    a_dict = __get_a_dict(a)
    for label, a_value in a_dict.items():
        add_plot(ax=ax, x=x, y=func(a_value, x), label=label)
    add_legend(ax, __should_add_legend(legend, a_dict))
    return fig


def plot_data(  # pylint: disable=too-many-arguments
    data: FittingData,
    title_name,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    grid: bool = False,
    x_log_scale: bool = False,
    y_log_scale: bool = False,
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
    :param xmin: Optional. Minimum value for x. if None, calculated from given data
    :type xmin: float
    :param xmax: Optional. Maximum value for x. if None, calculated from given data
    :type xmax: float
    :param grid: Add grid lines or not
    :type grid: bool
    :param x_log_scale: Set the scale of the  x axis to be logarithmic.
    :type x_log_scale: bool
    :param y_log_scale: Set the scale of the y axis to be logarithmic.
    :type y_log_scale: bool
    :returns: ``matplotlib.pyplot.Figure``
    """
    ax, fig = get_figure(  # pylint: disable=invalid-name
        title_name=title_name,
        xlabel=xlabel,
        ylabel=ylabel,
        grid=grid,
        x_log_scale=x_log_scale,
        y_log_scale=y_log_scale,
    )
    errorbar(ax=ax, data=data)
    xmin, xmax = get_plot_borders(x=data.x, xmin=xmin, xmax=xmax)
    limit_axes(ax=ax, xmin=xmin, xmax=xmax)
    return fig


def get_figure(  # pylint: disable=too-many-arguments
    title_name,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    grid: bool = False,
    x_log_scale: bool = False,
    y_log_scale: bool = False,
):
    """
    Gets a figure from matplotlib.

    :param title_name: Optional. Title for the figure.
    :type title_name: str
    :param xlabel: Optional. Label of the x axis
    :type xlabel: str
    :param ylabel: Optional. Label of the x axis
    :type ylabel: str
    :param grid: Add grid lines or not
    :type grid: bool
    :param x_log_scale: Set the scale of the  x axis to be logarithmic.
    :type x_log_scale: bool
    :param y_log_scale: Set the scale of the y axis to be logarithmic.
    :type y_log_scale: bool
    :return: Figure instance
    """
    fig = plt.figure()
    ax = fig.add_subplot()  # pylint: disable=invalid-name
    title(ax=ax, title_name=title_name)
    label_axes(ax=ax, xlabel=xlabel, ylabel=ylabel)
    add_grid(ax=ax, is_grid=grid)
    set_scales(ax=ax, is_x_log_scale=x_log_scale, is_y_log_scale=y_log_scale)

    return ax, Figure(fig)


def title(ax: plt.Axes, title_name: Optional[str]):  # pylint: disable=invalid-name
    """
    Add/remove title to figure.

    :param ax: Figure axes.
    :type ax: matplotlib.pyplot.Axes
    :param title_name: Optional. If None, don't add title. otherwise, add given title
    :type title_name: str
    """
    if title_name is not None:
        ax.set_title(title_name)


def label_axes(  # pylint: disable=invalid-name
    ax: plt.Axes, xlabel: Optional[str], ylabel: Optional[str]
):
    """
    Add/remove labels to figure.

    :param ax: Figure axes.
    :type ax: matplotlib.pyplot.Axes
    :param xlabel: Optional. If None, don't add label. otherwise, add given label
    :type xlabel: str
    :param ylabel: Optional. If None, don't add label. otherwise, add given label
    :type ylabel: str
    """
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)


def limit_axes(  # pylint: disable=invalid-name
    ax: plt.Axes, xmin: Optional[float] = None, xmax: Optional[float] = None
):
    """
    Set limits on axes.

    :param ax: Figure axes.
    :type ax: matplotlib.pyplot.Axes
    :param xmin: Optional. Minimum value for x. if None, calculated from given data
    :type xmin: float
    :param xmax: Optional. Maximum value for x. if None, calculated from given data
    :type xmax: float
    """
    if xmin is not None:
        ax.set_xlim(left=xmin)
    if xmax is not None:
        ax.set_xlim(right=xmax)


def add_grid(ax: plt.Axes, is_grid: bool):  # pylint: disable=invalid-name
    """
    Add/remove grid to figure.

    :param ax: Figure axes.
    :type ax: matplotlib.pyplot.Axes
    :param is_grid: Add or remote grid to plot
    :type is_grid: bool
    """
    ax.grid(is_grid)


def add_legend(ax: plt.Axes, is_legend: bool):  # pylint: disable=invalid-name
    """
    Add/remove legend to figure.

    :param ax: Figure axes.
    :type ax: matplotlib.pyplot.Axes
    :param is_legend: Add or remote legend to plot
    :type is_legend: bool
    """
    if is_legend:
        ax.legend()


def set_scales(  # pylint: disable=invalid-name
    ax: plt.Axes, is_x_log_scale: bool, is_y_log_scale: bool
):
    """
    Change x axis scale to logarithmic.

    :param ax: Figure axes
    :type ax: matplotlib.pyplot.Axes
    :param is_x_log_scale: Change x axis scale to logarithmic or not.
    :type is_x_log_scale: bool
    :param is_y_log_scale: Change y axis scale to logarithmic or not.
    :type is_y_log_scale: bool
    """
    if is_x_log_scale:
        ax.set_xscale("log")
    if is_y_log_scale:
        ax.set_yscale("log")


def add_plot(
    ax: plt.Axes,
    x: Union[np.ndarray, List[float]],
    y: Union[np.ndarray, List[float]],
    label: Optional[str] = None,
):  # pylint: disable=C0103
    """
    Plot y as a function of x.

    :param ax: Figure axes.
    :type ax: matplotlib.pyplot.Axes
    :param x: X values
    :type x: list of floats or ``numpy.ndarray``
    :param y: Y values
    :type y: list of floats or ``numpy.ndarray``
    :param label: Optional. Label for the plot that would be added to the legend
    :type label: str
    """
    ax.plot(x, y, label=label)


def horizontal_line(  # pylint: disable=C0103
    ax: plt.Axes, xmin: float, xmax: float, y: float = 0
):
    """
    Add horizontal line to figure.

    :param ax: Figure axes.
    :type ax: matplotlib.pyplot.Axes
    :param xmin: Minimum x value of line
    :type xmin: float
    :param xmax: Maximum x value of line
    :type xmax: float
    :param y: The y value of the line
    :type y: float
    """
    ax.hlines(y, xmin=xmin, xmax=xmax, linestyles="dashed")


def errorbar(ax: plt.Axes, data: FittingData):  # pylint: disable=C0103
    """
    Plot error bar to figure.

    :param ax: Figure axes.
    :type ax: matplotlib.pyplot.Axes
    :param data: Data to visualize
    :type data: eddington.fitting_data.FittingData
    """
    ax.errorbar(
        x=data.x,
        y=data.y,
        xerr=data.xerr,
        yerr=data.yerr,
        markersize=1,
        marker="o",
        linestyle="None",
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
    arguments_values = [
        f"a[{i}]={to_relevant_precision_string(val)}" for i, val in enumerate(a)
    ]
    return f"[{', '.join(arguments_values)}]"


def __should_add_legend(legend, a_dict):  # pylint: disable=invalid-name
    if legend is not None:
        return legend
    if len(a_dict) >= 2:
        return True
    return False
