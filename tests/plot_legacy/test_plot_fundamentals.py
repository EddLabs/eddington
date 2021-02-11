from unittest.mock import Mock

import pytest
from pytest_cases import parametrize_with_cases

from eddington import show_or_export
from eddington.exceptions import PlottingError
from eddington.plot.line_style import LineStyle
from tests.plot_legacy import cases
from tests.util import assert_calls


@parametrize_with_cases(argnames="base_dict, plot_method", cases=cases)
def test_simple_plot(base_dict, plot_method, mock_figure):
    fig = plot_method(**base_dict)
    assert (
        fig._raw_figure == mock_figure  # pylint: disable=protected-access
    ), "Figure is different than expected"
    ax = mock_figure.add_subplot.return_value
    ax.set_title.assert_called_once_with(cases.TITLE_NAME)
    ax.set_xlabel.assert_not_called()
    ax.set_ylabel.assert_not_called()
    ax.grid.assert_called_with(False)
    ax.set_xscale.assert_not_called()
    ax.set_yscale.assert_not_called()


@parametrize_with_cases(argnames="base_dict, plot_method", cases=cases)
def test_plot_with_xlabel(base_dict, plot_method, mock_figure):
    xlabel = "X Label"
    fig = plot_method(**base_dict, xlabel=xlabel)
    assert (
        fig._raw_figure == mock_figure  # pylint: disable=protected-access
    ), "Figure is different than expected"
    ax = mock_figure.add_subplot.return_value
    ax.set_xlabel.assert_called_once_with(xlabel)


@parametrize_with_cases(argnames="base_dict, plot_method", cases=cases)
def test_plot_with_ylabel(base_dict, plot_method, mock_figure):
    ylabel = "Y Label"
    fig = plot_method(**base_dict, ylabel=ylabel)
    assert (
        fig._raw_figure == mock_figure  # pylint: disable=protected-access
    ), "Figure is different than expected"
    ax = mock_figure.add_subplot.return_value
    ax.set_ylabel.assert_called_once_with(ylabel)


@parametrize_with_cases(argnames="base_dict, plot_method", cases=cases)
def test_plot_with_grid(base_dict, plot_method, mock_figure):
    fig = plot_method(**base_dict, grid=True)
    assert (
        fig._raw_figure == mock_figure  # pylint: disable=protected-access
    ), "Figure is different than expected"
    ax = mock_figure.add_subplot.return_value
    ax.grid.assert_called_once_with(True)


@parametrize_with_cases(argnames="base_dict, plot_method", cases=cases)
def test_plot_with_x_log_scale(base_dict, plot_method, mock_figure):
    fig = plot_method(**base_dict, x_log_scale=True)
    assert (
        fig._raw_figure == mock_figure  # pylint: disable=protected-access
    ), "Figure is different than expected"
    ax = mock_figure.add_subplot.return_value
    ax.set_xscale.assert_called_once_with("log")
    ax.set_yscale.assert_not_called()


@parametrize_with_cases(argnames="base_dict, plot_method", cases=cases)
def test_plot_with_y_log_scale(base_dict, plot_method, mock_figure):
    fig = plot_method(**base_dict, y_log_scale=True)
    assert (
        fig._raw_figure == mock_figure  # pylint: disable=protected-access
    ), "Figure is different than expected"
    ax = mock_figure.add_subplot.return_value
    ax.set_xscale.assert_not_called()
    ax.set_yscale.assert_called_once_with("log")


@parametrize_with_cases(
    argnames="base_dict, plot_method",
    cases=[cases.case_plot_data, cases.case_plot_fitting],
)
def test_error_bar_without_boundaries(base_dict, plot_method, mock_figure):
    fig = plot_method(**base_dict)
    assert (
        fig._raw_figure == mock_figure  # pylint: disable=protected-access
    ), "Figure is different than expected"
    data = base_dict["data"]
    assert_calls(
        mock_figure.add_subplot.return_value.errorbar,
        calls=[
            (
                [],
                dict(
                    x=data.x,
                    y=data.y,
                    xerr=data.xerr,
                    yerr=data.yerr,
                    markersize=1,
                    marker="o",
                    linestyle="None",
                    label=None,
                    ecolor=None,
                    mec=None,
                ),
            ),
        ],
        rel=1e-5,
    )


@parametrize_with_cases(
    argnames="base_dict, plot_method",
    cases=[cases.case_plot_data, cases.case_plot_fitting],
)
def test_error_bar_with_xmin(base_dict, plot_method, mock_figure):
    xmin = 4
    fig = plot_method(**base_dict, xmin=xmin)
    assert (
        fig._raw_figure == mock_figure  # pylint: disable=protected-access
    ), "Figure is different than expected"
    data = base_dict["data"]
    data_filter = [val >= xmin for val in data.x]
    assert_calls(
        mock_figure.add_subplot.return_value.errorbar,
        calls=[
            (
                [],
                dict(
                    x=data.x[data_filter],
                    y=data.y[data_filter],
                    xerr=data.xerr[data_filter],
                    yerr=data.yerr[data_filter],
                    markersize=1,
                    marker="o",
                    linestyle="None",
                    label=None,
                    ecolor=None,
                    mec=None,
                ),
            ),
        ],
        rel=1e-5,
    )


@parametrize_with_cases(
    argnames="base_dict, plot_method",
    cases=[cases.case_plot_data, cases.case_plot_fitting],
)
def test_error_bar_with_xmax(base_dict, plot_method, mock_figure):
    xmax = 4
    fig = plot_method(**base_dict, xmax=xmax)
    assert (
        fig._raw_figure == mock_figure  # pylint: disable=protected-access
    ), "Figure is different than expected"
    data = base_dict["data"]
    data_filter = [val <= xmax for val in data.x]
    assert_calls(
        mock_figure.add_subplot.return_value.errorbar,
        calls=[
            (
                [],
                dict(
                    x=data.x[data_filter],
                    y=data.y[data_filter],
                    xerr=data.xerr[data_filter],
                    yerr=data.yerr[data_filter],
                    markersize=1,
                    marker="o",
                    linestyle="None",
                    label=None,
                    ecolor=None,
                    mec=None,
                ),
            ),
        ],
        rel=1e-5,
    )


@parametrize_with_cases(argnames="base_dict, plot_method", cases=cases)
def test_plot_as_context(
    base_dict, plot_method, mock_figure, mock_plt_clf, mock_plt_close
):
    output = "/path/to/output"
    with plot_method(**base_dict) as fig:
        fig.savefig(output)
    mock_figure.savefig.assert_called_once_with(output)
    mock_plt_clf.assert_called_once_with()
    mock_plt_close.assert_called_once_with("all")


@parametrize_with_cases(argnames="base_dict, plot_method", cases=cases)
def test_plot_with_no_x_raises_exception(
    base_dict, plot_method, mock_figure, mock_plt_clf, mock_plt_close
):
    base_dict["data"].x_column = None

    with pytest.raises(PlottingError, match="^Cannot plot without x values.$"):
        plot_method(**base_dict)


@parametrize_with_cases(argnames="base_dict, plot_method", cases=cases)
def test_plot_with_no_y_raises_exception(
    base_dict, plot_method, mock_figure, mock_plt_clf, mock_plt_close
):
    base_dict["data"].y_column = None

    with pytest.raises(PlottingError, match="^Cannot plot without y values.$"):
        plot_method(**base_dict)


@parametrize_with_cases(argnames="base_dict, plot_method", cases=cases)
def test_plot_with_no_data_raises_exception(
    base_dict, plot_method, mock_figure, mock_plt_clf, mock_plt_close
):
    base_dict["data"].unselect_all_records()

    with pytest.raises(PlottingError, match="^Cannot plot without any chosen record.$"):
        plot_method(**base_dict)


def test_show_or_export_without_output(mock_plt_show):
    fig = Mock()
    show_or_export(fig, None)
    mock_plt_show.assert_called_once_with()


def test_show_or_export_with_output():
    output = "/path/to/output"
    fig = Mock()
    show_or_export(fig, output)
    fig.savefig.assert_called_once_with(output)


def test_plot_linestyles():
    assert set(LineStyle.all()) == {"solid", "none", "dashed", "dashdot", "dotted"}
