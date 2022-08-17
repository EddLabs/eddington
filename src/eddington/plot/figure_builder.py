"""Module for building plotting figure using the builder design pattern."""
from dataclasses import InitVar, dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np

from eddington.consts import DEFAULT_TICKS
from eddington.exceptions import PlottingError
from eddington.fitting_data import FittingData
from eddington.fitting_function_class import FittingFunction
from eddington.interval import Interval
from eddington.plot.figure import Figure
from eddington.plot.line_style import LineStyle

# pylint: disable=too-few-public-methods


@dataclass
class FigureInstruction:
    """
    Instructions interface for specifying what to add to the figure.

    :param name: Name of the instruction, used for identification
    :type name: str
    :param unique: Whether this instruction is unique per figure builder. False by
        default
    :type name: bool
    """

    name: str
    unique: bool = field(default=False)

    def add_to_figure(self, fig: Figure):
        """
        Add this instruction to figure.

        :param fig: Figure to add element to
        :type fig: Figure
        :raises NotImplementedError: If not overridden, raise not implemented error.
        """
        raise NotImplementedError("PlotInstructions.add_plot should be override.")


class FigureTitleInstruction(FigureInstruction):
    """Add title to figure."""

    def __init__(self, title: str):
        """
        Instruction constructor.

        :param title: Title to add to the figure
        :type title: str
        """
        super().__init__(name="title", unique=True)
        self.title = title

    def add_to_figure(self, fig: Figure):
        """
        Add this instruction to figure.

        :param fig: Figure to add element to
        :type fig: Figure
        """
        fig.ax.set_title(self.title)


class FigureXLabelInstruction(FigureInstruction):
    """Add x axis label to figure."""

    def __init__(self, xlabel: str):
        """
        Instruction constructor.

        :param xlabel: X axis label to add to the figure
        :type xlabel: str
        """
        super().__init__(name="xlabel", unique=True)
        self.xlabel = xlabel

    def add_to_figure(self, fig: Figure):
        """
        Add this instruction to figure.

        :param fig: Figure to add element to
        :type fig: Figure
        """
        fig.ax.set_xlabel(self.xlabel)


class FigureYLabelInstruction(FigureInstruction):
    """Add y axis label to figure."""

    def __init__(self, ylabel: str):
        """
        Instruction constructor.

        :param ylabel: Y axis label to add to the figure
        :type ylabel: str
        """
        super().__init__(name="ylabel", unique=True)
        self.ylabel = ylabel

    def add_to_figure(self, fig: Figure):
        """
        Add this instruction to figure.

        :param fig: Figure to add element to
        :type fig: Figure
        """
        fig.ax.set_ylabel(self.ylabel)


class FigureGridInstruction(FigureInstruction):
    """Add grid lines to figure."""

    def __init__(self):
        """Instruction constructor."""
        super().__init__(name="grid", unique=True)

    def add_to_figure(self, fig: Figure):
        """
        Add this instruction to figure.

        :param fig: Figure to add element to
        :type fig: Figure
        """
        fig.ax.grid()


class FigureXScaleInstruction(FigureInstruction):
    """Set x axis scale of figure."""

    def __init__(self, scale: str):
        """
        Instruction constructor.

        :param scale: Scale to set in the x axis
        :type scale: str
        """
        super().__init__(name="x_scale", unique=True)
        self.scale = scale

    def add_to_figure(self, fig: Figure):
        """
        Add this instruction to figure.

        :param fig: Figure to add element to
        :type fig: Figure
        """
        fig.ax.set_xscale(self.scale)


class FigureYScaleInstruction(FigureInstruction):
    """Set y axis scale of figure."""

    def __init__(self, scale: str):
        """
        Instruction constructor.

        :param scale: Scale to set in the y axis
        :type scale: str
        """
        super().__init__(name="y_scale", unique=True)
        self.scale = scale

    def add_to_figure(self, fig: Figure):
        """
        Add this instruction to figure.

        :param fig: Figure to add element to
        :type fig: Figure
        """
        fig.ax.set_yscale(self.scale)


class FigureLegendInstruction(FigureInstruction):
    """Add legend to figure."""

    def __init__(self):
        """Instruction constructor."""
        super().__init__(name="legend", unique=True)

    def add_to_figure(self, fig: Figure):
        """
        Add this instruction to figure.

        :param fig: Figure to add element to
        :type fig: Figure
        """
        fig.ax.legend()


class HorizontalLineInstruction(FigureInstruction):
    """Add horizontal line in a given height to figure."""

    def __init__(
        self,
        interval: Interval,
        y_value: float,
        linestyle: LineStyle = LineStyle.SOLID,
        color: Optional[str] = None,
    ):
        """
        Instruction constructor.

        :param interval: Interval representing x values
        :type interval: Interval
        :param y_value: y value of the horizontal line
        :type y_value: float
        :param linestyle: Line style of the horizontal line
        :type linestyle: LineStyle
        :param color: Optional. Color of the line style
        :type color: str
        """
        super().__init__(name="horizontal_line")
        self.interval = interval
        self.y_value = y_value
        self.linestyle = linestyle
        self.color = color

    def add_to_figure(self, fig: Figure):
        """
        Add this instruction to figure.

        :param fig: Figure to add element to
        :type fig: Figure
        """
        fig.ax.hlines(
            y=self.y_value,
            xmin=self.interval.min_val,
            xmax=self.interval.max_val,
            linestyle=self.linestyle.value,
            colors=self.color,
        )


class FigurePlotInstruction(FigureInstruction):
    """Add function plot to figure."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        interval: Interval,
        func: FittingFunction,
        a: Union[np.ndarray, List[float]],
        label: Optional[str] = None,
        color: Optional[str] = None,
        linestyle: LineStyle = LineStyle.SOLID,
        ticks: int = DEFAULT_TICKS,
    ):
        """
        Instruction constructor.

        :param interval: Interval representing x values
        :type interval: Interval
        :param func: Function to show its plot
        :type func: FittingFunction
        :param a: parameters to use as input to fitting function
        :type a: floats list or numpy.ndarray
        :param label: Label of the plot to add to the legend.
        :type label: str
        :param color: Optional. Color of the line style
        :type color: str
        :param linestyle: Line style of the horizontal line
        :type linestyle: LineStyle
        :param ticks: Number of ticks to use for plotting calculation
        :type ticks: int
        """
        super().__init__(name="plot")
        self.interval = interval
        self.a = a
        self.func = func
        self.label = label
        self.color = color
        self.linestyle = linestyle
        self.ticks = ticks

    def add_to_figure(self, fig: Figure):
        """
        Add this instruction to figure.

        :param fig: Figure to add element to
        :type fig: Figure
        """
        x = self.interval.ticks(self.ticks)
        y = self.func(self.a, x)
        fig.ax.plot(
            x, y, label=self.label, color=self.color, linestyle=self.linestyle.value
        )


class FigureErrorBarInstruction(FigureInstruction):
    """Add error bar to figure."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        x: Union[np.ndarray, List[float]],
        xerr: Union[np.ndarray, List[float], None],
        y: Union[np.ndarray, List[float]],
        yerr: Union[np.ndarray, List[float], None],
        label: Optional[str] = None,
        color: Optional[str] = None,
    ):
        """
        Instruction constructor.

        :param x: X values of the error bar
        :type x: floats list or numpy.ndarray
        :param xerr: X error values of the error bar
        :type xerr: floats list or numpy.ndarray
        :param y: Y values of the error bar
        :type y: floats list or numpy.ndarray
        :param yerr: Y error values of the error bar
        :type yerr: floats list or numpy.ndarray
        :param label: Label of the error bar to add to the legend.
        :type label: str
        :param color: Color of the error bar.
        :type color: str
        """
        super().__init__(name="errorbar")
        self.x = x
        self.xerr = xerr
        self.y = y
        self.yerr = yerr
        self.label = label
        self.color = color

    def add_to_figure(self, fig: Figure):
        """
        Add this instruction to figure.

        :param fig: Figure to add element to
        :type fig: Figure
        """
        fig.ax.errorbar(
            x=self.x,
            y=self.y,
            xerr=self.xerr,
            yerr=self.yerr,
            markersize=1,
            marker="o",
            linestyle="None",
            label=self.label,
            ecolor=self.color,
            mec=self.color,
        )


class FigureScatterInstruction(FigureInstruction):
    """Add error bar to figure."""

    def __init__(
        self,
        x: Union[np.ndarray, List[float]],
        y: Union[np.ndarray, List[float]],
        label: Optional[str] = None,
        color: Optional[str] = None,
    ):
        """
        Instruction constructor.

        :param x: X values of the error bar
        :type x: floats list or numpy.ndarray
        :param y: Y values of the error bar
        :type y: floats list or numpy.ndarray
        :param label: Label of the error bar to add to the legend.
        :type label: str
        :param color: Color of the error bar.
        :type color: str
        """
        super().__init__(name="scatter")
        self.x = x
        self.y = y
        self.label = label
        self.color = color

    def add_to_figure(self, fig: Figure):
        """
        Add this instruction to figure.

        :param fig: Figure to add element to
        :type fig: Figure
        """
        fig.ax.scatter(
            x=self.x,
            y=self.y,
            marker="o",
            s=2,
            linestyle="None",
            label=self.label,
            color=self.color,
        )


@dataclass
class FigureBuilder:
    """Builder class for creating a figure."""

    instructions: List[FigureInstruction] = field(default_factory=list, init=False)
    unique_instructions: Dict[str, FigureInstruction] = field(
        default_factory=dict, init=False
    )
    title: InitVar[Optional[str]] = None
    xlabel: InitVar[Optional[str]] = None
    ylabel: InitVar[Optional[str]] = None
    grid: InitVar[bool] = False
    legend: InitVar[bool] = False

    def __post_init__(  # pylint: disable=too-many-arguments
        self,
        title: Optional[str],
        xlabel: Optional[str],
        ylabel: Optional[str],
        grid: bool,
        legend: bool,
    ):
        """
        Initialize unique instructions if were given.

        :param title: Optional title of the figure:
        :type title: str
        :param xlabel: Optional title of the x axis of the figure:
        :type xlabel: str
        :param ylabel: Optional title of the x axis of the figure:
        :type ylabel: str
        :param grid: If true, add grid lines to the figure
        :type grid: bool
        :param legend: If true, add legend to the figure
        :type legend: bool
        """
        if title is not None:
            self.add_title(title)
        if xlabel is not None:
            self.add_xlabel(xlabel)
        if ylabel is not None:
            self.add_ylabel(ylabel)
        if grid:
            self.add_grid()
        if legend:
            self.add_legend()

    def add_title(self, title: str):
        """
        Add title instruction to figure.

        :param title: Title to add to the figure
        :type title: str
        :return: self
        :rtype: FigureBuilder
        """
        return self.add_instruction(FigureTitleInstruction(title=title))

    def add_xlabel(self, xlabel: str):
        """
        Add x label instruction to figure.

        :param xlabel: X axis label to add to the figure
        :type xlabel: str
        :return: self
        :rtype: FigureBuilder
        """
        return self.add_instruction(FigureXLabelInstruction(xlabel=xlabel))

    def add_ylabel(self, ylabel: str):
        """
        Add y label instruction to figure.

        :param ylabel: Y axis label to add to the figure
        :type ylabel: str
        :return: self
        :rtype: FigureBuilder
        """
        return self.add_instruction(FigureYLabelInstruction(ylabel=ylabel))

    def add_x_log_scale(self):
        """
        Set x axis scale to logarithmic scale.

        :return: self
        :rtype: FigureBuilder
        """
        return self.add_instruction(FigureXScaleInstruction("log"))

    def add_y_log_scale(self):
        """
        Set y axis scale to logarithmic scale.

        :return: self
        :rtype: FigureBuilder
        """
        return self.add_instruction(FigureYScaleInstruction("log"))

    def add_grid(self):
        """
        Set grid lines to figure.

        :return: self
        :rtype: FigureBuilder
        """
        return self.add_instruction(FigureGridInstruction())

    def add_legend(self):
        """
        Set legend lines to figure.

        :return: self
        :rtype: FigureBuilder
        """
        return self.add_instruction(FigureLegendInstruction())

    def add_horizontal_line(
        self,
        interval: Interval,
        y_value: float,
        linestyle: LineStyle = LineStyle.SOLID,
        color: Optional[str] = None,
    ):
        """
        Add horizontal line to figure.

        :param interval: Interval representing x values
        :type interval: Interval
        :param y_value: y value of the horizontal line
        :type y_value: float
        :param linestyle: Line style of the horizontal line
        :type linestyle: LineStyle
        :param color: Optional. Color of the line style
        :type color: str
        :return: self
        :rtype: FigureBuilder
        """
        return self.add_instruction(
            HorizontalLineInstruction(
                interval=interval, y_value=y_value, linestyle=linestyle, color=color
            )
        )

    def add_error_bar(  # pylint: disable=too-many-arguments
        self,
        x: Union[np.ndarray, List[float]],
        xerr: Union[np.ndarray, List[float], None],
        y: Union[np.ndarray, List[float]],
        yerr: Union[np.ndarray, List[float], None],
        label: Optional[str] = None,
        color: Optional[str] = None,
    ):
        """
        Add error bar to figure.

        :param x: X values of the error bar
        :type x: floats list or numpy.ndarray
        :param xerr: X error values of the error bar
        :type xerr: floats list or numpy.ndarray
        :param y: Y values of the error bar
        :type y: floats list or numpy.ndarray
        :param yerr: Y error values of the error bar
        :type yerr: floats list or numpy.ndarray
        :param label: Label of the error bar to add to the legend.
        :type label: str
        :param color: Color of the error bar.
        :type color: str
        :return: self
        :rtype: FigureBuilder
        """
        return self.add_instruction(
            FigureErrorBarInstruction(
                x=x,
                xerr=xerr,
                y=y,
                yerr=yerr,
                label=label,
                color=color,
            )
        )

    def add_scatter(
        self,
        x: Union[np.ndarray, List[float]],
        y: Union[np.ndarray, List[float]],
        label: Optional[str] = None,
        color: Optional[str] = None,
    ):
        """
        Add scatter to figure.

        :param x: X values of the error bar
        :type x: floats list or numpy.ndarray
        :param y: Y values of the error bar
        :type y: floats list or numpy.ndarray
        :param label: Label of the error bar to add to the legend.
        :type label: str
        :param color: Color of the error bar.
        :type color: str
        :return: self
        :rtype: FigureBuilder
        """
        return self.add_instruction(
            FigureScatterInstruction(x=x, y=y, label=label, color=color)
        )

    def add_data(
        self,
        data: FittingData,
        label: Optional[str] = None,
        color: Optional[str] = None,
    ):
        """
        Add scatter to figure.

        :param data: Data to plot
        :param label: Label of the error bar to add to the legend.
        :type label: str
        :param color: Color of the error bar.
        :type color: str
        :return: self
        :rtype: FigureBuilder
        :raises PlottingError: Raised when data doesn't have x or y values
        """
        x, y = data.x, data.y
        if x is None or y is None:
            raise PlottingError("Can't plot data without x or y values")
        xerr, yerr = data.xerr, data.yerr
        if xerr is None and yerr is None:
            return self.add_scatter(x=x, y=y, label=label, color=color)
        return self.add_error_bar(
            x=x, xerr=xerr, y=y, yerr=yerr, label=label, color=color
        )

    def add_plot(  # pylint: disable=too-many-arguments
        self,
        interval: Interval,
        a: Union[np.ndarray, List[float]],
        func: FittingFunction,
        label: Optional[str] = None,
        color: Optional[str] = None,
        linestyle: LineStyle = LineStyle.SOLID,
    ):
        """
        Add plot to figure.

        :param interval: Interval representing x values
        :type interval: Interval
        :param func: Function to show its plot
        :type func: FittingFunction
        :param a: parameters to use as input to fitting function
        :type a: floats list or numpy.ndarray
        :param label: Label of the plot to add to the legend.
        :type label: str
        :param color: Optional. Color of the line style
        :type color: str
        :param linestyle: Line style of the horizontal line
        :type linestyle: LineStyle
        :return: self
        :rtype: FigureBuilder
        """
        return self.add_instruction(
            FigurePlotInstruction(
                interval=interval,
                a=a,
                func=func,
                label=label,
                color=color,
                linestyle=linestyle,
            )
        )

    def add_instruction(self, instruction: FigureInstruction) -> "FigureBuilder":
        """
        Add general instruction to plot building.

        :param instruction: Instruction instance for what to add to the figure
        :type instruction: FigureInstruction
        :return: self
        :rtype: FigureBuilder
        :raises PlottingError: Raised when trying to add a unique instruction for the
            second time.
        """
        if instruction.unique:
            if instruction.name in self.unique_instructions:
                raise PlottingError(f"Cannot set {instruction.name} twice.")
            self.unique_instructions[instruction.name] = instruction
        else:
            self.instructions.append(instruction)
        return self

    def build(self, figure: Optional[Figure] = None) -> Figure:
        """
        Build figure.

        :param figure: Optional. If given, add the instruction to that figure.
            If not, creates a new figure.
        :type figure: Optional[Figure]
        :return: Built figure item
        :rtype: Figure
        """
        if figure is None:
            figure = Figure()
        for instruction in self.instructions:
            instruction.add_to_figure(figure)
        for instruction in self.unique_instructions.values():
            instruction.add_to_figure(figure)
        return figure
