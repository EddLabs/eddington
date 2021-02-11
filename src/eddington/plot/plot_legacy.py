"""
Legacy plotting methods.

Those methods will be removed in version 0.1.0
"""
from collections import OrderedDict
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from eddington.exceptions import PlottingError
from eddington.fitting_data import FittingData
from eddington.print_util import to_relevant_precision_string


class LineStyle(Enum):
    """Enum class for line style options."""

    SOLID = "solid"
    DASHED = "dashed"
    DASHDOT = "dashdot"
    DOTTED = "dotted"
    NONE = "none"

    @classmethod
    def all(cls) -> List[str]:
        """
        Get all line style values.

        :return: Possible values of line styles
        :rtype: List[str]
        """
        return [linestyle.value for linestyle in cls]


class Figure:
    """
    Wraps matplotlib Figure class.

    It releases the memory when the figure is no longer in use.
    """

    def __init__(self, fig: plt.Figure):
        """
        Figure constructor.

        :param fig: Actual matplotlib figure
        :type fig: plt.Figure
        """
        self._actual_fig = fig

    def __enter__(self):
        """
        Return self when entering as context.

        :return: self
        :rtype: Figure
        """
        return self

    def __getattr__(self, item: str):
        """
        Get attributes from wrapped figure.

        :param item: Item name to be returned
        :type item: str
        :return: Required item
        """
        return getattr(self._actual_fig, item)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Clear memory on exit.

        # noqa: DAR101
        """
        plt.clf()
        plt.close("all")


def plot_residuals(  # pylint: disable=invalid-name,too-many-arguments,too-many-locals
    func,
    data: FittingData,
    a: np.ndarray,
    title_name,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    grid: bool = False,
    color: Optional[str] = None,
    x_log_scale: bool = False,
    y_log_scale: bool = False,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
):
    """
    Plot residuals plot.

    :param func: Fitting function.
    :type func: FittingFunction
    :param data: Fitting data to be plotted.
    :type data: FittingData
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
    :param color: Optional. Color to use for in the plot.
    :type color: str
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
    __validate_all_columns_exist(data)
    ax, fig = get_figure(
        title_name=title_name,
        xlabel=xlabel,
        ylabel=ylabel,
        grid=grid,
        x_log_scale=x_log_scale,
        y_log_scale=y_log_scale,
    )
    xmin, xmax = get_plot_borders(x=data.x, xmin=xmin, xmax=xmax)  # type: ignore
    checkers_list = get_checkers_list(
        values=data.x, min_val=xmin, max_val=xmax  # type: ignore
    )
    residuals = data.residuals(fit_func=func, a=a)
    add_errorbar(
        ax=ax,
        x=residuals.x[checkers_list],  # type: ignore
        xerr=residuals.xerr[checkers_list],  # type: ignore
        y=residuals.y[checkers_list],  # type: ignore
        yerr=residuals.yerr[checkers_list],  # type: ignore
        color=color,
    )
    horizontal_line(ax=ax, xmin=xmin, xmax=xmax)
    return fig


def plot_fitting(  # pylint: disable=C0103,R0913,R0914
    func,
    data: FittingData,
    a: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],
    title_name,
    color: Optional[Union[str, List[str], Dict[str, str]]] = None,
    linestyle: Union[
        LineStyle, List[LineStyle], Dict[str, LineStyle]
    ] = LineStyle.SOLID,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    grid: bool = False,
    legend: Optional[bool] = None,
    data_color: Optional[str] = None,
    x_log_scale: bool = False,
    y_log_scale: bool = False,
    step: Optional[float] = None,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
):
    """
    Plot fitting plot.

    :param func: Fitting function.
    :type func: FittingFunction
    :param data: Fitting data to be plotted.
    :type data: FittingData
    :param a: The parameters result
    :type a: ``numpy.ndarray``, a list of ``numpy.ndarray`` items or a dictionary from
        strings to ``numpy.ndarray``
    :param color: Colors to use for each fit. Can be single value, a list of values
        or a dictionary between a color to a value
    :type color: str, List of str or a dictionary between str to str
    :param linestyle: Line styles to use for each fit. Can be single value, a list of
        values or a dictionary between a line style to a value
    :type color: str, List of str or a dictionary between str to LineStyle
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
    :param data_color: Optional. Color of the data error bar.
    :type data_color: str
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
    __validate_all_columns_exist(data)
    ax, fig = get_figure(
        title_name=title_name,
        xlabel=xlabel,
        ylabel=ylabel,
        grid=grid,
        x_log_scale=x_log_scale,
        y_log_scale=y_log_scale,
    )
    xmin, xmax = get_plot_borders(x=data.x, xmin=xmin, xmax=xmax)  # type: ignore
    checkers_list = get_checkers_list(
        values=data.x, min_val=xmin, max_val=xmax  # type: ignore
    )
    add_errorbar(
        ax=ax,
        x=data.x[checkers_list],  # type: ignore
        xerr=data.xerr[checkers_list],  # type: ignore
        y=data.y[checkers_list],  # type: ignore
        yerr=data.yerr[checkers_list],  # type: ignore
        color=data_color,
    )
    x = get_x_plot_values(xmin=xmin, xmax=xmax, step=step)
    a_dict = __get_a_dict(a)
    labels = list(a_dict.keys())
    colors_dict = __build_values_dict(value=color, labels=labels)
    linestyle_dict = __build_values_dict(
        value=linestyle, labels=labels, default_value=LineStyle.SOLID
    )
    for label, a_value in a_dict.items():
        add_plot(
            ax=ax,
            x=x,
            y=func(a_value, x),
            label=label,
            color=colors_dict.get(label, None),
            linestyle=linestyle_dict.get(label, LineStyle.SOLID),
        )
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
    color: Optional[str] = None,
    x_log_scale: bool = False,
    y_log_scale: bool = False,
):
    """
    Plot fitting data.

    :param data: Fitting data to be plotted.
    :type data: FittingData
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
    :param color: Optional. Color to use for in the plot.
    :type color: str
    :param x_log_scale: Set the scale of the  x axis to be logarithmic.
    :type x_log_scale: bool
    :param y_log_scale: Set the scale of the y axis to be logarithmic.
    :type y_log_scale: bool
    :returns: ``matplotlib.pyplot.Figure``
    """
    __validate_all_columns_exist(data)
    ax, fig = get_figure(  # pylint: disable=invalid-name
        title_name=title_name,
        xlabel=xlabel,
        ylabel=ylabel,
        grid=grid,
        x_log_scale=x_log_scale,
        y_log_scale=y_log_scale,
    )
    xmin, xmax = get_plot_borders(x=data.x, xmin=xmin, xmax=xmax)  # type: ignore
    checkers_list = get_checkers_list(
        values=data.x, min_val=xmin, max_val=xmax  # type: ignore
    )
    add_errorbar(
        ax=ax,
        x=data.x[checkers_list],  # type: ignore
        xerr=data.xerr[checkers_list],  # type: ignore
        y=data.y[checkers_list],  # type: ignore
        yerr=data.yerr[checkers_list],  # type: ignore
        color=color,
    )
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


def add_plot(  # pylint: disable=invalid-name,too-many-arguments
    ax: plt.Axes,
    x: Union[np.ndarray, List[float]],
    y: Union[np.ndarray, List[float]],
    label: Optional[str] = None,
    color: Optional[str] = None,
    linestyle: LineStyle = LineStyle.SOLID,
):
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
    :param color: Color of the plot
    :type color: str
    :param linestyle: The style to use for the plot. Solid by default
    :type linestyle: LineStyle
    """
    ax.plot(x, y, label=label, color=color, linestyle=linestyle.value)


def add_errorbar(  # pylint: disable=invalid-name,too-many-arguments
    ax: plt.Axes,
    x: Union[np.ndarray, List[float]],
    xerr: Union[np.ndarray, List[float]],
    y: Union[np.ndarray, List[float]],
    yerr: Union[np.ndarray, List[float]],
    label: Optional[str] = None,
    color: Optional[str] = None,
):
    """
    Plot error bar to figure.

    :param ax: Figure axes.
    :type ax: matplotlib.pyplot.Axes
    :param x: X values
    :type x: list of floats or ``numpy.ndarray``
    :param xerr: X error values
    :type xerr: list of floats or ``numpy.ndarray``
    :param y: Y values
    :type y: list of floats or ``numpy.ndarray``
    :param yerr: Y error values
    :type yerr: list of floats or ``numpy.ndarray``
    :param label: Optional. Label for the error bar that would be added to the legend
    :type label: str
    :param color: Color of the plot
    :type color: str
    """
    ax.errorbar(
        x=x,
        y=y,
        xerr=xerr,
        yerr=yerr,
        markersize=1,
        marker="o",
        linestyle="None",
        label=label,
        ecolor=color,
        mec=color,
    )


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


def get_x_plot_values(
    xmin: float, xmax: float, step: Optional[float] = None
) -> np.ndarray:
    """
    Get x values to use in plot methods.

    :param xmin: Minimum x value
    :type xmin: float
    :param xmax: Maximum x value
    :type xmax: float
    :param step: Optional. gap between each x values
    :type step: None or float
    :return: array of x values
    :rtype: numpy.ndarray
    """
    if step is None:
        step = (xmax - xmin) / 1000.0
    return np.arange(xmin, xmax, step=step)


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


def get_checkers_list(
    values: Union[List[float], np.ndarray],
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> List[bool]:
    """
    Get a boolean map of valid indices in an array.

    :param values: Values to be checked
    :type values: List of floats or numpy.ndarray
    :param min_val: Optional. Minimum allowed value
    :type min_val: float
    :param max_val: Optional. Maximum allowed value
    :type max_val: float
    :return: List of booleans indicating valid indices
    :rtype: List[bool]
    """
    return [__in_bounds(val=val, min_val=min_val, max_val=max_val) for val in values]


def __get_a_dict(a):  # pylint: disable=invalid-name
    if isinstance(a, (dict, OrderedDict)):
        return a
    if isinstance(a, list):
        return OrderedDict([(build_repr_string(a_value), a_value) for a_value in a])
    if isinstance(a, np.ndarray):
        return OrderedDict([(build_repr_string(a), a)])
    raise PlottingError(
        f"{a} has unmatching type. Can except only numpy arrays, "
        "lists of numpy arrays and dictionaries."
    )


def __build_values_dict(value, labels, default_value=None):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if not isinstance(value, list):
        value = [value]
    if len(value) < len(labels):
        value += [default_value for _ in range(len(labels) - len(value))]
    return dict(zip(labels, value))


def __should_add_legend(legend, a_dict):  # pylint: disable=invalid-name
    if legend is not None:
        return legend
    if len(a_dict) >= 2:
        return True
    return False


def __in_bounds(val, min_val=None, max_val=None):  # pylint: disable=invalid-name
    if min_val is not None and val < min_val:
        return False
    if max_val is not None and val > max_val:
        return False
    return True


def __validate_all_columns_exist(data):
    if not any(data.records_indices):
        raise PlottingError("Cannot plot without any chosen record.")
    for column_type, column_name in data.used_columns.items():
        if column_name is None:
            raise PlottingError(f"Cannot plot without {column_type} values.")
