from unittest.mock import Mock

from pytest_cases import parametrize_with_cases

from eddington import show_or_export
from tests.plot import cases
from tests.util import assert_dict_equal


@parametrize_with_cases(argnames="base_dict, plot_method", cases=cases)
def test_simple_plot(base_dict, plot_method, mock_figure):
    fig = plot_method(**base_dict)
    assert fig == mock_figure, "Figure is different than expected"
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
    assert fig == mock_figure, "Figure is different than expected"
    ax = mock_figure.add_subplot.return_value
    ax.set_xlabel.assert_called_once_with(xlabel)


@parametrize_with_cases(argnames="base_dict, plot_method", cases=cases)
def test_plot_with_ylabel(base_dict, plot_method, mock_figure):
    ylabel = "Y Label"
    fig = plot_method(**base_dict, ylabel=ylabel)
    assert fig == mock_figure, "Figure is different than expected"
    ax = mock_figure.add_subplot.return_value
    ax.set_ylabel.assert_called_once_with(ylabel)


@parametrize_with_cases(argnames="base_dict, plot_method", cases=cases)
def test_plot_with_grid(base_dict, plot_method, mock_figure):
    fig = plot_method(**base_dict, grid=True)
    assert fig == mock_figure, "Figure is different than expected"
    ax = mock_figure.add_subplot.return_value
    ax.grid.assert_called_once_with(True)


@parametrize_with_cases(argnames="base_dict, plot_method", cases=cases)
def test_plot_with_x_log_scale(base_dict, plot_method, mock_figure):
    fig = plot_method(**base_dict, x_log_scale=True)
    assert fig == mock_figure, "Figure is different than expected"
    ax = mock_figure.add_subplot.return_value
    ax.set_xscale.assert_called_once_with("log")
    ax.set_yscale.assert_not_called()


@parametrize_with_cases(argnames="base_dict, plot_method", cases=cases)
def test_plot_with_y_log_scale(base_dict, plot_method, mock_figure):
    fig = plot_method(**base_dict, y_log_scale=True)
    assert fig == mock_figure, "Figure is different than expected"
    ax = mock_figure.add_subplot.return_value
    ax.set_xscale.assert_not_called()
    ax.set_yscale.assert_called_once_with("log")


@parametrize_with_cases(
    argnames="base_dict, plot_method",
    cases=[cases.case_plot_data, cases.case_plot_fitting],
)
def test_error_bar(base_dict, plot_method, mock_figure):
    fig = plot_method(**base_dict)
    assert fig == mock_figure, "Figure is different than expected"
    ax = mock_figure.add_subplot.return_value
    assert ax.errorbar.call_count == 1
    assert_dict_equal(
        ax.errorbar.call_args_list[0][1],
        dict(
            x=cases.FIT_DATA.x,
            y=cases.FIT_DATA.y,
            xerr=cases.FIT_DATA.xerr,
            yerr=cases.FIT_DATA.yerr,
            markersize=1,
            marker="o",
            linestyle="None",
        ),
        rel=1e-5,
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
