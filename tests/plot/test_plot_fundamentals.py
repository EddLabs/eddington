from unittest.mock import Mock

import numpy as np
from pytest_cases import THIS_MODULE, parametrize_with_cases

from eddington import (
    FittingData,
    linear,
    plot_data,
    plot_fitting,
    plot_residuals,
    show_or_export,
)
from tests.util import assert_dict_equal, assert_list_equal

EPSILON = 1e-5

FUNC = linear
X = np.arange(1, 11)
A = np.array([1, 2])
FIT_DATA = FittingData.random(FUNC, x=X, a=A, measurements=X.shape[0])
TITLE_NAME = "Title"


def case_plot_data():
    return dict(data=FIT_DATA, title_name=TITLE_NAME), plot_data


def case_plot_fitting():
    return dict(func=FUNC, data=FIT_DATA, a=A, title_name=TITLE_NAME), plot_fitting


def case_plot_residuals():
    return dict(func=FUNC, data=FIT_DATA, a=A, title_name=TITLE_NAME), plot_residuals


@parametrize_with_cases(argnames="base_dict, plot_method", cases=THIS_MODULE)
def test_simple_plot(base_dict, plot_method, mock_figure):
    fig = plot_method(**base_dict)
    assert fig == mock_figure, "Figure is different than expected"
    ax = mock_figure.add_subplot.return_value
    ax.set_title.assert_called_once_with(TITLE_NAME)
    ax.set_xlabel.assert_not_called()
    ax.set_ylabel.assert_not_called()
    ax.grid.assert_called_with(False)


@parametrize_with_cases(argnames="base_dict, plot_method", cases=THIS_MODULE)
def test_plot_with_xlabel(base_dict, plot_method, mock_figure):
    xlabel = "X Label"
    fig = plot_method(**base_dict, xlabel=xlabel)
    assert fig == mock_figure, "Figure is different than expected"
    ax = mock_figure.add_subplot.return_value
    ax.set_xlabel.assert_called_once_with(xlabel)


@parametrize_with_cases(argnames="base_dict, plot_method", cases=THIS_MODULE)
def test_plot_with_ylabel(base_dict, plot_method, mock_figure):
    ylabel = "Y Label"
    fig = plot_method(**base_dict, ylabel=ylabel)
    assert fig == mock_figure, "Figure is different than expected"
    ax = mock_figure.add_subplot.return_value
    ax.set_ylabel.assert_called_once_with(ylabel)


@parametrize_with_cases(argnames="base_dict, plot_method", cases=THIS_MODULE)
def test_plot_with_grid(base_dict, plot_method, mock_figure):
    fig = plot_method(**base_dict, grid=True)
    assert fig == mock_figure, "Figure is different than expected"
    ax = mock_figure.add_subplot.return_value
    ax.grid.assert_called_once_with(True)


@parametrize_with_cases(
    argnames="base_dict, plot_method", cases=[case_plot_data, case_plot_fitting]
)
def test_error_bar(base_dict, plot_method, mock_figure):
    fig = plot_method(**base_dict)
    assert fig == mock_figure, "Figure is different than expected"
    ax = mock_figure.add_subplot.return_value
    assert ax.errorbar.call_count == 1
    assert_dict_equal(
        ax.errorbar.call_args_list[0][1],
        dict(
            x=FIT_DATA.x,
            y=FIT_DATA.y,
            xerr=FIT_DATA.xerr,
            yerr=FIT_DATA.yerr,
            markersize=1,
            marker="o",
            linestyle="None",
        ),
        rel=1e-5,
    )


@parametrize_with_cases(argnames="base_dict, plot_method", cases=[case_plot_residuals])
def test_residuals_error_bar(base_dict, plot_method, mock_figure):
    fig = plot_method(**base_dict)
    assert fig == mock_figure, "Figure is different than expected"
    y_residuals = FIT_DATA.y - FUNC(A, FIT_DATA.x)
    ax = mock_figure.add_subplot.return_value
    assert ax.errorbar.call_count == 1
    assert_dict_equal(
        ax.errorbar.call_args_list[0][1],
        dict(
            x=FIT_DATA.x,
            y=y_residuals,
            xerr=FIT_DATA.xerr,
            yerr=FIT_DATA.yerr,
            markersize=1,
            marker="o",
            linestyle="None",
        ),
        rel=1e-5,
    )


@parametrize_with_cases(argnames="base_dict, plot_method", cases=[case_plot_residuals])
def test_plot_residuals_without_boundaries(base_dict, plot_method, mock_figure):
    fig = plot_method(**base_dict)
    assert fig == mock_figure, "Figure is different than expected"
    ax = mock_figure.add_subplot.return_value
    assert ax.hlines.call_count == 1
    assert_list_equal(ax.hlines.call_args_list[0][0], [0], rel=EPSILON)
    assert_dict_equal(
        ax.hlines.call_args_list[0][1],
        dict(xmin=0.1, xmax=10.9, linestyles="dashed"),
        rel=EPSILON,
    )


@parametrize_with_cases(argnames="base_dict, plot_method", cases=[case_plot_residuals])
def test_plot_residuals_with_xmin(base_dict, plot_method, mock_figure):
    fig = plot_method(**base_dict, xmin=-10)
    assert fig == mock_figure, "Figure is different than expected"
    ax = mock_figure.add_subplot.return_value
    assert ax.hlines.call_count == 1
    assert_list_equal(ax.hlines.call_args_list[0][0], [0], rel=EPSILON)
    assert_dict_equal(
        ax.hlines.call_args_list[0][1],
        dict(xmin=-10, xmax=10.9, linestyles="dashed"),
        rel=EPSILON,
    )


@parametrize_with_cases(argnames="base_dict, plot_method", cases=[case_plot_residuals])
def test_plot_residuals_with_xmax(base_dict, plot_method, mock_figure):
    fig = plot_method(**base_dict, xmax=20)
    assert fig == mock_figure, "Figure is different than expected"
    ax = mock_figure.add_subplot.return_value
    assert ax.hlines.call_count == 1
    assert_list_equal(ax.hlines.call_args_list[0][0], [0], rel=EPSILON)
    assert_dict_equal(
        ax.hlines.call_args_list[0][1],
        dict(xmin=0.1, xmax=20, linestyles="dashed"),
        rel=EPSILON,
    )


def test_show_or_export_without_output(mock_figure, mock_plt_show):
    fig = Mock()
    show_or_export(fig, None)
    mock_plt_show.assert_called_once_with()


def test_show_or_export_with_output(mock_figure):
    output = "/path/to/output"
    fig = Mock()
    show_or_export(fig, output)
    fig.savefig.assert_called_once_with(output)
