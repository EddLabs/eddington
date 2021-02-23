import numpy as np
import pytest
from pytest_cases import fixture, parametrize_with_cases

from eddington.exceptions import PlottingError
from eddington.interval import Interval
from eddington.plot.figure_builder import FigureBuilder, FigureInstruction
from eddington.plot.line_style import LineStyle
from tests.dummy_functions import dummy_func1, dummy_func2
from tests.util import assert_calls

EPSILON = 1e-5


@fixture
def mock_figure(mocker):
    figure_class = mocker.patch("eddington.plot.figure_builder.Figure")
    return figure_class.return_value


# Successful build recipe


def case_title_in_constructor(mock_figure):
    title = "I am a title"
    figure_builder = FigureBuilder(title=title)
    yield figure_builder
    mock_figure.ax.set_title.assert_called_once_with(title)


def case_add_title(mock_figure):
    title = "I am a title"
    figure_builder = FigureBuilder()
    figure_builder.add_title(title)
    yield figure_builder
    mock_figure.ax.set_title.assert_called_once_with(title)


def case_xlabel_in_constructor(mock_figure):
    xlabel = "I am xlabel"
    figure_builder = FigureBuilder(xlabel=xlabel)
    yield figure_builder
    mock_figure.ax.set_xlabel.assert_called_once_with(xlabel)


def case_add_xlabel(mock_figure):
    xlabel = "I am xlabel"
    figure_builder = FigureBuilder()
    figure_builder.add_xlabel(xlabel)
    yield figure_builder
    mock_figure.ax.set_xlabel.assert_called_once_with(xlabel)


def case_ylabel_in_constructor(mock_figure):
    ylabel = "I am ylabel"
    figure_builder = FigureBuilder(ylabel=ylabel)
    yield figure_builder
    mock_figure.ax.set_ylabel.assert_called_once_with(ylabel)


def case_add_ylabel(mock_figure):
    ylabel = "I am ylabel"
    figure_builder = FigureBuilder()
    figure_builder.add_ylabel(ylabel)
    yield figure_builder
    mock_figure.ax.set_ylabel.assert_called_once_with(ylabel)


def case_grid_in_constructor(mock_figure):
    figure_builder = FigureBuilder(grid=True)
    yield figure_builder
    mock_figure.ax.grid.assert_called_once_with()


def case_add_grid(mock_figure):
    figure_builder = FigureBuilder()
    figure_builder.add_grid()
    yield figure_builder
    mock_figure.ax.grid.assert_called_once_with()


def case_legend_in_constructor(mock_figure):
    figure_builder = FigureBuilder(legend=True)
    yield figure_builder
    mock_figure.ax.legend.assert_called_once_with()


def case_add_legend(mock_figure):
    figure_builder = FigureBuilder()
    figure_builder.add_legend()
    yield figure_builder
    mock_figure.ax.legend.assert_called_once_with()


def case_add_x_log_scale(mock_figure):
    figure_builder = FigureBuilder()
    figure_builder.add_x_log_scale()
    yield figure_builder
    mock_figure.ax.set_xscale.assert_called_once_with("log")


def case_add_y_log_scale(mock_figure):
    figure_builder = FigureBuilder()
    figure_builder.add_y_log_scale()
    yield figure_builder
    mock_figure.ax.set_yscale.assert_called_once_with("log")


def case_add_horizontal_line(mock_figure):
    y_value = 2
    domain = Interval(min_val=-20, max_val=30)
    figure_builder = FigureBuilder()
    figure_builder.add_horizontal_line(y_value=y_value, interval=domain)
    yield figure_builder
    mock_figure.ax.hlines.assert_called_once_with(
        colors=None,
        linestyle="solid",
        xmin=domain.min_val,
        xmax=domain.max_val,
        y=y_value,
    )


def case_add_horizontal_line_with_linestyle_and_color(mock_figure):
    y_value = 2
    linestyle = LineStyle.DASHED
    color = "blue"
    domain = Interval(min_val=-20, max_val=30)
    figure_builder = FigureBuilder()
    figure_builder.add_horizontal_line(
        y_value=y_value, interval=domain, linestyle=linestyle, color=color
    )
    yield figure_builder
    mock_figure.ax.hlines.assert_called_once_with(
        colors=color,
        linestyle="dashed",
        xmin=domain.min_val,
        xmax=domain.max_val,
        y=y_value,
    )


def case_two_horizontal_lines(mock_figure):
    y1, y2 = 2, 8
    domain1 = Interval(min_val=-20, max_val=30)
    domain2 = Interval(min_val=-10, max_val=40)
    figure_builder = FigureBuilder()
    figure_builder.add_horizontal_line(y_value=y1, interval=domain1)
    figure_builder.add_horizontal_line(y_value=y2, interval=domain2)
    yield figure_builder
    assert_calls(
        mock_figure.ax.hlines,
        [
            (
                [],
                dict(
                    colors=None,
                    linestyle="solid",
                    xmin=domain1.min_val,
                    xmax=domain1.max_val,
                    y=y1,
                ),
            ),
            (
                [],
                dict(
                    colors=None,
                    linestyle="solid",
                    xmin=domain2.min_val,
                    xmax=domain2.max_val,
                    y=y2,
                ),
            ),
        ],
        rel=EPSILON,
    )


def case_add_plot(mock_figure):
    a = [1, 2]
    domain = Interval(min_val=-20, max_val=30)
    figure_builder = FigureBuilder()
    figure_builder.add_plot(interval=domain, a=a, func=dummy_func1)
    yield figure_builder
    x = domain.ticks(1000)
    assert_calls(
        mock_figure.ax.plot,
        [([x, dummy_func1(a, x)], dict(label=None, color=None, linestyle="solid"))],
        rel=EPSILON,
    )


def case_add_plot_with_additional_args(mock_figure):
    a = [1, 2]
    label = "a label"
    color = "black"
    linestyle = LineStyle.DOTTED
    domain = Interval(min_val=-20, max_val=30)
    figure_builder = FigureBuilder()
    figure_builder.add_plot(
        interval=domain,
        a=a,
        func=dummy_func1,
        label=label,
        color=color,
        linestyle=linestyle,
    )
    yield figure_builder
    x = domain.ticks(1000)
    assert_calls(
        mock_figure.ax.plot,
        [([x, dummy_func1(a, x)], dict(label=label, color=color, linestyle="dotted"))],
        rel=EPSILON,
    )


def case_add_two_plots(mock_figure):
    a1 = [1, 2]
    a2 = [3, 4, 5, 6]
    domain1 = Interval(min_val=-20, max_val=30)
    domain2 = Interval(min_val=-10, max_val=40)
    figure_builder = FigureBuilder()
    figure_builder.add_plot(interval=domain1, a=a1, func=dummy_func1)
    figure_builder.add_plot(interval=domain2, a=a2, func=dummy_func2)
    yield figure_builder
    x1, x2 = domain1.ticks(1000), domain2.ticks(1000)
    assert_calls(
        mock_figure.ax.plot,
        [
            (
                [x1, dummy_func1(a1, x1)],
                dict(label=None, color=None, linestyle="solid"),
            ),
            (
                [x2, dummy_func2(a2, x2)],
                dict(label=None, color=None, linestyle="solid"),
            ),
        ],
        rel=EPSILON,
    )


def case_add_error_bar(mock_figure):
    size = 20
    x, xerr, y, yerr = (
        np.random.uniform(0, 1, size=size),
        np.random.uniform(0, 1, size=size),
        np.random.uniform(0, 1, size=size),
        np.random.uniform(0, 1, size=size),
    )
    figure_builder = FigureBuilder()
    figure_builder.add_error_bar(x=x, xerr=xerr, y=y, yerr=yerr)
    yield figure_builder
    assert_calls(
        mock_figure.ax.errorbar,
        [
            (
                [],
                dict(
                    x=x,
                    xerr=xerr,
                    y=y,
                    yerr=yerr,
                    label=None,
                    ecolor=None,
                    mec=None,
                    markersize=1,
                    marker="o",
                    linestyle="None",
                ),
            )
        ],
        rel=EPSILON,
    )


def case_add_error_bar_with_additional_args(mock_figure):
    size = 20
    x, xerr, y, yerr = (
        np.random.uniform(0, 1, size=size),
        np.random.uniform(0, 1, size=size),
        np.random.uniform(0, 1, size=size),
        np.random.uniform(0, 1, size=size),
    )
    label = "a label"
    color = "brown"
    figure_builder = FigureBuilder()
    figure_builder.add_error_bar(
        x=x, xerr=xerr, y=y, yerr=yerr, label=label, color=color
    )
    yield figure_builder
    assert_calls(
        mock_figure.ax.errorbar,
        [
            (
                [],
                dict(
                    x=x,
                    xerr=xerr,
                    y=y,
                    yerr=yerr,
                    label=label,
                    ecolor=color,
                    mec=color,
                    markersize=1,
                    marker="o",
                    linestyle="None",
                ),
            )
        ],
        rel=EPSILON,
    )


def case_add_two_error_bars(mock_figure):
    size = 20
    x1, xerr1, y1, yerr1 = (
        np.random.uniform(0, 1, size=size),
        np.random.uniform(0, 1, size=size),
        np.random.uniform(0, 1, size=size),
        np.random.uniform(0, 1, size=size),
    )
    x2, xerr2, y2, yerr2 = (
        np.random.uniform(0, 1, size=size),
        np.random.uniform(0, 1, size=size),
        np.random.uniform(0, 1, size=size),
        np.random.uniform(0, 1, size=size),
    )
    figure_builder = FigureBuilder()
    figure_builder.add_error_bar(x=x1, xerr=xerr1, y=y1, yerr=yerr1)
    figure_builder.add_error_bar(x=x2, xerr=xerr2, y=y2, yerr=yerr2)
    yield figure_builder
    assert_calls(
        mock_figure.ax.errorbar,
        [
            (
                [],
                dict(
                    x=x1,
                    xerr=xerr1,
                    y=y1,
                    yerr=yerr1,
                    label=None,
                    ecolor=None,
                    mec=None,
                    markersize=1,
                    marker="o",
                    linestyle="None",
                ),
            ),
            (
                [],
                dict(
                    x=x2,
                    xerr=xerr2,
                    y=y2,
                    yerr=yerr2,
                    label=None,
                    ecolor=None,
                    mec=None,
                    markersize=1,
                    marker="o",
                    linestyle="None",
                ),
            ),
        ],
        rel=EPSILON,
    )


@parametrize_with_cases(argnames="figure_builder", cases=".")
def test_figure_builder_build_recipe(mock_figure, figure_builder):
    actual_figure = figure_builder.build()
    assert actual_figure == mock_figure


# Failed


def test_figure_builder_fail_to_add_empty_instruction(mock_figure):
    figure_builder = FigureBuilder()
    figure_builder.add_instruction(FigureInstruction(name="bla"))

    with pytest.raises(
        NotImplementedError,
        match="^PlotInstructions.add_plot should be override.$",
    ):
        figure_builder.build()


def test_figure_builder_fail_to_add_title_twice(mock_figure):
    title1 = "I am a title"
    title2 = "I am also a title"
    figure_builder = FigureBuilder()
    figure_builder.add_title(title1)

    with pytest.raises(PlottingError, match="^Cannot set title twice.$"):
        figure_builder.add_title(title2)


def test_figure_builder_fail_to_add_xlabel_twice(mock_figure):
    xlabel1 = "I am xlabel"
    xlabel2 = "I am also xlabel"
    figure_builder = FigureBuilder()
    figure_builder.add_xlabel(xlabel1)

    with pytest.raises(PlottingError, match="^Cannot set xlabel twice.$"):
        figure_builder.add_xlabel(xlabel2)


def test_figure_builder_fail_to_add_ylabel_twice(mock_figure):
    ylabel1 = "I am ylabel"
    ylabel2 = "I am also ylabel"
    figure_builder = FigureBuilder()
    figure_builder.add_ylabel(ylabel1)

    with pytest.raises(PlottingError, match="^Cannot set ylabel twice.$"):
        figure_builder.add_ylabel(ylabel2)


def test_figure_builder_fail_to_add_grid_twice(mock_figure):
    figure_builder = FigureBuilder()
    figure_builder.add_grid()

    with pytest.raises(PlottingError, match="^Cannot set grid twice.$"):
        figure_builder.add_grid()


def test_figure_builder_fail_to_add_legend_twice(mock_figure):
    figure_builder = FigureBuilder()
    figure_builder.add_legend()

    with pytest.raises(PlottingError, match="^Cannot set legend twice.$"):
        figure_builder.add_legend()


def test_figure_builder_fail_to_add_x_log_scale_twice(mock_figure):
    figure_builder = FigureBuilder()
    figure_builder.add_x_log_scale()

    with pytest.raises(PlottingError, match="^Cannot set x_scale twice.$"):
        figure_builder.add_x_log_scale()


def test_figure_builder_fail_to_add_y_log_scale_twice(mock_figure):
    figure_builder = FigureBuilder()
    figure_builder.add_y_log_scale()

    with pytest.raises(PlottingError, match="^Cannot set y_scale twice.$"):
        figure_builder.add_y_log_scale()
