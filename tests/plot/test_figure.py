import mock
import pytest

from eddington.plot.figure import Figure


@pytest.fixture
def mock_plt(mocker):
    return mocker.patch("eddington.plot.figure.plt")


def test_figure_default_constructor(mock_plt):
    figure = Figure()
    assert figure._raw_figure == mock_plt.figure.return_value
    assert figure.ax == mock_plt.figure.return_value.add_subplot.return_value
    mock_plt.figure.assert_called_once_with()
    mock_plt.figure.return_value.add_subplot.assert_called_once_with()


def test_figure_constructor_with_existing_figure(mock_plt):
    raw_figure = mock.Mock()
    figure = Figure(raw_figure=raw_figure)
    assert figure._raw_figure == raw_figure
    assert figure.ax == raw_figure.add_subplot.return_value
    mock_plt.figure.assert_not_called()
    raw_figure.add_subplot.assert_called_once_with()


def test_figure_get_attr(mock_plt):
    raw_figure = mock.Mock()
    figure = Figure(raw_figure=raw_figure)
    assert figure.bla == raw_figure.bla


def test_figure_as_context(mock_plt):
    raw_figure = mock.Mock()
    with Figure(raw_figure=raw_figure):
        mock_plt.clf.assert_not_called()
        mock_plt.close.assert_not_called()

    mock_plt.clf.assert_called_once_with()
    mock_plt.close.assert_called_once_with("all")
