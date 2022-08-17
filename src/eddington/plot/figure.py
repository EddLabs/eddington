"""Module containing figure wrapper class."""
from typing import Optional

import matplotlib.pyplot as plt


class Figure:
    """
    Wraps matplotlib Figure class.

    It releases the memory when the figure is no longer in use.
    """

    def __init__(self, raw_figure: Optional[plt.Figure] = None):
        """
        Figure constructor.

        :param raw_figure: Raw figure that this class wraps
        :type raw_figure: plt.Figure
        """
        if raw_figure is None:
            raw_figure = plt.figure()
        self._raw_figure = raw_figure
        self.ax = self._raw_figure.add_subplot()

    @property
    def ax(self) -> plt.Axes:
        """
        Axes object getter.

        :return: figure axes
        :rtype: matplotlib.pyplot.Axes
        """
        return self._ax

    @ax.setter
    def ax(self, ax: plt.Axes):
        """
        Axes object setter.

        :param ax: figure axes
        :type ax: matplotlib.pyplot.Axes
        """
        self._ax = ax

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
        return getattr(self._raw_figure, item)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Clear memory on exit.

        # noqa: DAR101
        """
        plt.clf()
        plt.close("all")
