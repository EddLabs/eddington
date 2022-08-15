"""Fitting data class insert the fitting algorithm."""
# pylint: disable=too-many-lines
import csv
import json
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from numbers import Number
from pathlib import Path
from typing import Any, Dict, ItemsView, Iterator, List, Optional, Union

import numpy as np
import openpyxl

from eddington import io_util
from eddington.exceptions import (
    FittingDataColumnExistenceError,
    FittingDataColumnIndexError,
    FittingDataColumnsLengthError,
    FittingDataError,
    FittingDataRecordIndexError,
    FittingDataRecordsSelectionError,
    FittingDataSetError,
)
from eddington.interval import Interval
from eddington.raw_data_builder import RawDataBuilder
from eddington.statistics import Statistics


@dataclass
class Columns:
    """Dataclass for chosen column names."""

    x: Optional[str] = field(default=None)
    xerr: Optional[str] = field(default=None)
    y: Optional[str] = field(default=None)
    yerr: Optional[str] = field(default=None)

    def __iter__(self) -> Iterator[Optional[str]]:
        """
        Iterate over the given columns.

        :return: Columns iterator
        :rtype: Iterator[Optional[str]]
        """
        return iter([self.x, self.xerr, self.y, self.yerr])

    def items(self) -> ItemsView[str, Optional[str]]:
        """
        Get columns as items.

        :return: column type to column name tuples list
        :rtype: ItemsView[str, Optional[str]]
        """
        return asdict(self).items()


class FittingData:  # pylint: disable=R0902,R0904
    """Fitting data class."""

    _x_index: Optional[int]
    _xerr_index: Optional[int]
    _yerr_index: Optional[int]
    _y_index: Optional[int]
    _x_column: Optional[str]
    _xerr_column: Optional[str]
    _yerr_column: Optional[str]
    _y_column: Optional[str]

    def __init__(  # pylint: disable=too-many-arguments
        self,
        data: Union[OrderedDict, Dict[str, np.ndarray]],
        x_column: Optional[Union[str, int]] = None,
        xerr_column: Optional[Union[str, int]] = None,
        y_column: Optional[Union[str, int]] = None,
        yerr_column: Optional[Union[str, int]] = None,
        search: bool = True,
    ):
        """
        Constructor.

        :param data: Dictionary from a column name to its values
        :type data: ``dict`` or ``OrderedDict`` from ``str`` to ``numpy.ndarray``
        :param x_column: Indicates which column should be used as the x parameter
        :type x_column: ``str`` or ``numpy.ndarray``
        :param xerr_column: Indicates which column should be used as the x error
            parameter
        :type xerr_column: ``str`` or ``numpy.ndarray``
        :param y_column: Indicates which column should be used as the y parameter
        :type y_column: ``str`` or ``numpy.ndarray``
        :param yerr_column: Indicates which column should be used as the y error
            parameter
        :type yerr_column: ``str`` or ``numpy.ndarray``
        :param search: Search for a column if it wasn't explicitly provided in the
            constructor.
        :type search: bool
        :raises FittingDataColumnsLengthError: Raised if not all columns have the same
            length
        """
        self._data = OrderedDict(
            [(key, np.array(value)) for key, value in data.items()]
        )
        self._x_column = self._xerr_column = self._y_column = self._yerr_column = None
        self._x_index = self._xerr_index = self._y_index = self._yerr_index = None
        self._statistics_map: Dict[str, Statistics] = OrderedDict()
        self._all_columns = list(self.data.keys())
        lengths = {value.size for value in self.data.values()}
        if len(lengths) != 1:
            raise FittingDataColumnsLengthError()
        self._number_of_records = next(iter(lengths))
        self.select_all_records()
        self.__update_statistics()
        if x_column is None and search:
            self.x_index = 1
        else:
            self.x_column = x_column
        if xerr_column is None and search and self.x_index is not None:
            self.xerr_index = self.x_index + 1
        else:
            self.xerr_column = xerr_column
        if y_column is None and search and self.xerr_index is not None:
            self.y_index = self.xerr_index + 1
        else:
            self.y_column = y_column
        if yerr_column is None and search and self.y_index is not None:
            self.yerr_index = self.y_index + 1
        else:
            self.yerr_column = yerr_column

    def copy(
        self,
        only_selected_columns: bool = False,
        only_selected_records: bool = False,
    ):
        """
        Make a copy of self.

        :param only_selected_columns: If true, copy only columns which are used as x,
            x error, y and y error columns. Otherwise, copy all columns
        :type only_selected_columns: bool
        :param only_selected_records: If true, copy only selected records. Otherwise,
            copy all records.
        :type only_selected_records: bool
        :return: a copy of this fitting data.
        :rtype: Fitting Data
        """
        if only_selected_columns:
            columns = [column for column in self.used_columns if column is not None]
        else:
            columns = self.all_columns
        raw_data = OrderedDict()
        for column in columns:
            raw_data[column] = self.column_data(
                column, only_selected=only_selected_records
            )
        new_fitting_data = FittingData(
            data=raw_data,
            x_column=self.x_column,
            xerr_column=self.xerr_column,
            y_column=self.y_column,
            yerr_column=self.yerr_column,
            search=False,
        )
        if not only_selected_records:
            new_fitting_data.records_indices = self.records_indices
        return new_fitting_data

    # Data properties are read-only

    @property
    def number_of_records(self) -> int:
        """
        Number of records.

        :return: Number of records
        :rtype: int
        """
        return self._number_of_records

    @property
    def number_of_columns(self) -> int:
        """
        Number of columns.

        :return: Number of columns
        :rtype: int
        """
        return len(self.all_columns)

    @property
    def data(self) -> OrderedDict:
        """
        Data matrix.

        :return: The actual raw data
        :rtype: OrderedDict
        """
        return self._data

    @property
    def statistics_map(self) -> Dict[str, Statistics]:
        """
        Return updated statistics map.

        :return: Statistics map of the data
        :rtype: Statistics
        """
        return self._statistics_map

    @property
    def all_records(self) -> List[List[Any]]:
        """
        Get all records in data as a list.

        :return: List of all records
        :rtype: List[List[Any]]
        """
        return [list(record) for record in zip(*self.data.values())]

    @property
    def records(self):
        """
        Get all selected records in data as a list.

        :return: records list
        """
        return list(
            zip(*[column[self.records_indices] for column in self.data.values()])
        )

    @property
    def all_columns(self) -> List[str]:
        """
        Property of columns list.

        :return: list of all columns
        :rtype: List[str]
        """
        return self._all_columns

    @property
    def used_columns(self) -> Columns:
        """
        Dictionary of columns in use.

        :return: columns used dictionary
        :rtype: Columns
        """
        return Columns(
            x=self.x_column,
            xerr=self.xerr_column,
            y=self.y_column,
            yerr=self.yerr_column,
        )

    @property
    def x(self) -> Optional[np.ndarray]:
        """
        Property of the x values.

        :return: values of the x column
        :rtype: np.ndarray
        """
        return self.column_data(self.x_column)

    @property
    def xerr(self) -> Optional[np.ndarray]:
        """
        Property of the x error values.

        :return: values of the x error column
        :rtype: np.ndarray
        """
        return self.column_data(self.xerr_column)

    @property
    def y(self) -> Optional[np.ndarray]:
        """
        Property of the y values.

        :return: values of the y column
        :rtype: np.ndarray
        """
        return self.column_data(self.y_column)

    @property
    def yerr(self) -> Optional[np.ndarray]:
        """
        Property of the y error values.

        :return: values of the y error column
        :rtype: np.ndarray
        """
        return self.column_data(self.yerr_column)

    @property
    def x_domain(self) -> Interval:
        """
        Minimal interval containing the values of the x column.

        :return: x values domain.
        :rtype: Interval
        """
        return self.column_domain(column_name=self.x_column)

    @property
    def y_domain(self) -> Interval:
        """
        Minimal interval containing the values of the y column.

        :return: y values domain.
        :rtype: Interval
        """
        return self.column_domain(column_name=self.y_column)

    # Records indices methods

    def select_record(self, index: int):
        """
        Select a record to be used in fitting.

        :param index: index of the desired record **starting from 1**.
        :type index: int
        """
        self.__validate_record_index(index)
        self.records_indices[index - 1] = True
        self.__update_statistics()

    def unselect_record(self, index: int):
        """
        Unselect a record to be used in fitting.

        :param index: index of the desired record **starting from 1**.
        :type index: int
        """
        self.__validate_record_index(index)
        self.records_indices[index - 1] = False
        self.__update_statistics()

    def select_all_records(self):
        """Select all records to be used in fitting."""
        self.records_indices = [True] * self.number_of_records

    def unselect_all_records(self):
        """Unselect all records from being used in fitting."""
        self.records_indices = [False] * self.number_of_records

    def select_by_x_domain(
        self,
        xmin: Optional[float] = None,
        xmax: Optional[float] = None,
        update_selected: bool = False,
    ) -> None:
        """
        Select records by limiting x values.

        :param xmin: Optional. Minimum value for x. If none, will not consider lower
            bound for x values
        :type xmin: float
        :param xmax: Optional. Maximum value for x. If none, will not consider upper
            bound for x values
        :type xmax: float
        :param update_selected: If true, combine with records which have already been
            selected. If false, select from all records
        :type update_selected: bool
        """
        selected_indices = self.__get_indices_in_interval(
            interval=Interval(min_val=xmin, max_val=xmax), column_name=self.x_column
        )
        if update_selected:
            self.records_indices = self.__combine_records_indices(
                self.records_indices, selected_indices
            )
        else:
            self.records_indices = selected_indices

    def select_by_y_domain(
        self,
        ymin: Optional[float] = None,
        ymax: Optional[float] = None,
        update_selected: bool = False,
    ) -> None:
        """
        Select records by limiting y values.

        :param ymin: Optional. Minimum value for y. If none, will not consider lower
            bound for y values
        :type ymin: float
        :param ymax: Optional. Maximum value for y. If none, will not consider upper
            bound for y values
        :type ymax: float
        :param update_selected: If true, combine with records which have already been
            selected. If false, select from all records
        :type update_selected: bool
        """
        selected_indices = self.__get_indices_in_interval(
            interval=Interval(min_val=ymin, max_val=ymax), column_name=self.y_column
        )
        if update_selected:
            self.records_indices = self.__combine_records_indices(
                self.records_indices, selected_indices
            )
        else:
            self.records_indices = selected_indices

    def select_by_domains(  # pylint: disable=too-many-arguments
        self,
        xmin: Optional[float] = None,
        xmax: Optional[float] = None,
        ymin: Optional[float] = None,
        ymax: Optional[float] = None,
        update_selected: bool = False,
    ) -> None:
        """
        Select records by limiting y values.

        :param xmin: Optional. Minimum value for x. If none, will not consider lower
            bound for x values
        :type xmin: float
        :param xmax: Optional. Maximum value for x. If none, will not consider upper
            bound for x values
        :type xmax: float
        :param ymin: Optional. Minimum value for y. If none, will not consider lower
            bound for y values
        :type ymin: float
        :param ymax: Optional. Maximum value for y. If none, will not consider upper
            bound for y values
        :type ymax: float
        :param update_selected: If true, combine with records which have already been
            selected. If false, select from all records
        :type update_selected: bool
        """
        x_selected_indices = self.__get_indices_in_interval(
            interval=Interval(min_val=xmin, max_val=xmax), column_name=self.x_column
        )
        y_selected_indices = self.__get_indices_in_interval(
            interval=Interval(min_val=ymin, max_val=ymax), column_name=self.y_column
        )
        if update_selected:
            self.records_indices = self.__combine_records_indices(
                self.records_indices, x_selected_indices, y_selected_indices
            )
        else:
            self.records_indices = self.__combine_records_indices(
                x_selected_indices, y_selected_indices
            )

    def is_selected(self, index):
        """
        Checks if a record is selected or not.

        :param index: index of the desired record **starting from 1**.
        :type index: int
        :returns: True if record is selected, otherwise False.
        :rtype: bool
        """
        return self.records_indices[index - 1]

    def all_selected(self) -> bool:
        """
        Checks whether all records are selected.

        :returns: True if all records are selected, False otherwise.
        :rtype: bool
        """
        return all(self.records_indices)

    def non_selected(self) -> bool:
        """
        Checks whether no record has been selected.

        :returns: True if no record has been selected, False otherwise.
        :rtype: bool
        """
        return not any(self.records_indices)

    @property
    def records_indices(self) -> List[bool]:
        """
        Property of selected indices.

        :return: List of booleans indicating which records are selected.
        :rtype: List[bool]
        """
        return self._records_indices

    @records_indices.setter
    def records_indices(self, records_indices: List[bool]):
        if len(records_indices) != self.number_of_records:
            raise FittingDataRecordsSelectionError(
                f"Should select {self.number_of_records} records,"
                f" only {len(records_indices)} selected."
            )
        if not all(isinstance(element, bool) for element in records_indices):
            raise FittingDataRecordsSelectionError(
                "When setting record indices, all values should be booleans."
            )
        self._records_indices = records_indices
        self.__update_statistics()

    @property
    def x_index(self) -> Optional[int]:
        """
        Index of the x column.

        :return: Index of the x column
        :rtype: int
        """
        return self._x_index

    @x_index.setter
    def x_index(self, x_index: int):
        self.__validate_column_index(index=x_index)
        self._x_index = x_index
        column_name = self.__get_column_name(x_index)
        if column_name != self.x_column:
            self.x_column = column_name

    @property
    def x_column(self):
        """
        Name of the x column.

        :return: The name of the x error column
        :rtype: str
        """
        return self._x_column

    @x_column.setter
    def x_column(self, x_column):
        self.__validate_column_name(x_column)
        self._x_column = x_column
        index = self.__get_column_index(x_column)
        if index != self.x_index:
            self.x_index = index

    @property
    def xerr_index(self) -> Optional[int]:
        """
        Index of the x error column.

        :return: Index of the x error column
        :rtype: int
        """
        return self._xerr_index

    @xerr_index.setter
    def xerr_index(self, xerr_index: int):
        self.__validate_column_index(index=xerr_index)
        self._xerr_index = xerr_index
        column_name = self.__get_column_name(xerr_index)
        if column_name != self.xerr_column:
            self.xerr_column = column_name

    @property
    def xerr_column(self):
        """
        Name of the x error column.

        :return: The name of the x error column
        :rtype: str
        """
        return self._xerr_column

    @xerr_column.setter
    def xerr_column(self, xerr_column):
        self.__validate_column_name(xerr_column)
        self._xerr_column = xerr_column
        index = self.__get_column_index(xerr_column)
        if index != self.xerr_index:
            self.xerr_index = index

    @property
    def y_index(self) -> Optional[int]:
        """
        Index of the y column.

        :return: Index of the y column
        :rtype: int
        """
        return self._y_index

    @y_index.setter
    def y_index(self, y_index: int):
        self.__validate_column_index(index=y_index)
        self._y_index = y_index
        column_name = self.__get_column_name(y_index)
        if column_name != self.y_column:
            self.y_column = column_name

    @property
    def y_column(self):
        """
        Name of the y column.

        :return: The name of the y column
        :rtype: str
        """
        return self._y_column

    @y_column.setter
    def y_column(self, y_column):
        self.__validate_column_name(y_column)
        self._y_column = y_column
        index = self.__get_column_index(y_column)
        if index != self.y_index:
            self.y_index = index

    @property
    def yerr_index(self) -> Optional[int]:
        """
        Index of the y error column.

        :return: Index of the y error column
        :rtype: int
        """
        return self._yerr_index

    @yerr_index.setter
    def yerr_index(self, yerr_index: int):
        self.__validate_column_index(index=yerr_index)
        self._yerr_index = yerr_index
        column_name = self.__get_column_name(yerr_index)
        if column_name != self.yerr_column:
            self.yerr_column = column_name

    @property
    def yerr_column(self):
        """
        Name of the y error column.

        :return: The name of the y error column
        :rtype: str
        """
        return self._yerr_column

    @yerr_column.setter
    def yerr_column(self, yerr_column):
        self.__validate_column_name(yerr_column)
        self._yerr_column = yerr_column
        index = self.__get_column_index(yerr_column)
        if index != self.yerr_index:
            self.yerr_index = index

    # More functionalities

    def residuals(self, fit_func, a: Union[List[float], np.ndarray]) -> "FittingData":
        """
        Creates residuals :class:`FittingData` objects.

        :param fit_func: :class:`FittingFunction` to evaluate with the fit data
        :type fit_func: ``FittingFunction``
        :param a: the parameters of the given fitting function
        :type a: ``numpy.ndarray``
        :returns: residuals :class:`FittingData`
        :raises FittingDataError: Raised when missing x or y column
        """
        if self.x_column is None:
            raise FittingDataError(
                "Could not calculate residuals data without x values."
            )
        if self.y_column is None:
            raise FittingDataError(
                "Could not calculate residuals data without y values."
            )
        y_residuals = self.y - fit_func(a, self.x)
        raw_data: Dict[str, np.ndarray] = {
            self.x_column: self.x,  # type: ignore
            self.y_column: y_residuals,
        }
        kwargs = dict(x_column=self.x_column, y_column=self.y_column, search=False)
        if self.xerr_column is not None:
            raw_data[self.xerr_column] = self.xerr  # type: ignore
            kwargs["xerr_column"] = self.xerr_column
        if self.yerr_column is not None:
            raw_data[self.yerr_column] = self.yerr  # type: ignore
            kwargs["yerr_column"] = self.yerr_column
        return FittingData(data=raw_data, **kwargs)

    # Read Methods

    @classmethod
    def read_from_excel(  # pylint: disable=too-many-arguments
        cls,
        filepath: Union[str, Path],
        sheet: str,
        x_column: Optional[Union[str, int]] = None,
        xerr_column: Optional[Union[str, int]] = None,
        y_column: Optional[Union[str, int]] = None,
        yerr_column: Optional[Union[str, int]] = None,
        search: bool = True,
    ):
        """
        Read :class:`FittingData` from excel file.

        :param filepath: str or Path. Path to location of excel file
        :param sheet: str. The name of the sheet to extract the data from.
        :param x_column: Indicates which column should be used as the
            x parameter
        :type x_column: ``str`` or ``numpy.ndarray``
        :param xerr_column: Indicates which column should be used as the x error
            parameter
        :type xerr_column: ``str`` or ``numpy.ndarray``
        :param y_column: Indicates which column should be used as the x parameter
        :type y_column: ``str`` or ``numpy.ndarray``
        :param yerr_column: Indicates which column should be used as the y error
            parameter
        :type yerr_column: ``str`` or ``numpy.ndarray``
        :param search: Search for a column if it wasn't explicitly provided.
        :type search: bool
        :returns: :class:`FittingData` read from the excel file.
        :raises FittingDataError: Raised when the given sheet do not exist in excel
            file.
        """
        if isinstance(filepath, str):
            filepath = Path(filepath)

        workbook = openpyxl.load_workbook(filepath, data_only=True)
        if sheet not in workbook.sheetnames:
            raise FittingDataError(
                f'Sheet named "{sheet}" does not exist in "{filepath.name}"'
            )
        rows = [list(row) for row in workbook[sheet].values]
        return cls.__build_from_rows(
            rows=rows,
            x_column=x_column,
            xerr_column=xerr_column,
            y_column=y_column,
            yerr_column=yerr_column,
            search=search,
        )

    @classmethod
    def read_from_csv(  # pylint: disable=too-many-arguments
        cls,
        filepath: Union[str, Path],
        x_column: Optional[Union[str, int]] = None,
        xerr_column: Optional[Union[str, int]] = None,
        y_column: Optional[Union[str, int]] = None,
        yerr_column: Optional[Union[str, int]] = None,
        search: bool = True,
    ):
        """
        Read :class:`FittingData` from csv file.

        :param filepath: str or Path. Path to location of csv file
        :param x_column: Indicates which column should be used as the x parameter
        :type x_column: ``str`` or ``numpy.ndarray``
        :param xerr_column: Indicates which column should be used as the x error
            parameter
        :type xerr_column: ``str`` or ``numpy.ndarray``
        :param y_column: Indicates which column should be used as the x parameter
        :type y_column: ``str`` or ``numpy.ndarray``
        :param yerr_column: Indicates which column should be used as the y error
            parameter
        :type yerr_column: ``str`` or ``numpy.ndarray``
        :param search: Search for a column if it wasn't explicitly provided.
        :type search: bool
        :returns: :class:`FittingData` read from the csv file.
        """
        if isinstance(filepath, str):
            filepath = Path(filepath)
        with open(filepath, mode="r", encoding="utf-8") as csv_file:
            csv_obj = csv.reader(csv_file)
            rows = list(csv_obj)
        return cls.__build_from_rows(
            rows=rows,
            x_column=x_column,
            xerr_column=xerr_column,
            y_column=y_column,
            yerr_column=yerr_column,
            search=search,
        )

    @classmethod
    def read_from_json(  # pylint: disable=too-many-arguments
        cls,
        filepath: Union[str, Path],
        x_column: Optional[Union[str, int]] = None,
        xerr_column: Optional[Union[str, int]] = None,
        y_column: Optional[Union[str, int]] = None,
        yerr_column: Optional[Union[str, int]] = None,
        search: bool = True,
    ):
        """
        Read :class:`FittingData` from json file.

        :param filepath: str or Path. Path to location of csv file
        :param x_column: Indicates which column should be used as the x parameter
        :type x_column: ``str`` or ``numpy.ndarray``
        :param xerr_column: Indicates which column should be used as the x error
            parameter
        :type xerr_column: ``str`` or ``numpy.ndarray``
        :param y_column: Indicates which column should be used as the x parameter
        :type y_column: ``str`` or ``numpy.ndarray``
        :param yerr_column: Indicates which column should be used as the y error
            parameter
        :type yerr_column: ``str`` or ``numpy.ndarray``
        :param search: Search for a column if it wasn't explicitly provided.
        :type search: bool
        :returns: :class:`FittingData` read from the json file.
        """
        if isinstance(filepath, str):
            filepath = Path(filepath)
        with open(filepath, mode="r", encoding="utf-8") as file:
            data = json.load(file, object_pairs_hook=OrderedDict)
        # fmt: off
        return FittingData(
            RawDataBuilder.fix_types_in_raw_dict(data),
            x_column=x_column, xerr_column=xerr_column,
            y_column=y_column, yerr_column=yerr_column,
            search=search,
        )
        # fmt: on

    # Getter methods

    def column_data(
        self, column_name: Optional[str], only_selected: bool = True
    ) -> Optional[np.ndarray]:
        """
        Get the data of a column.

        :param column_name: The header name of the desired column. if None, return
            None
        :type column_name: str
        :param only_selected: If true, return only values selected records. otherwise,
            Return values of all records.
        :type only_selected: bool
        :returns: The data of the given column
        :rtype: numpy.ndarray or None
        """
        if column_name is None:
            return None
        self.__validate_column_name(column_name)
        values = self.data[column_name]
        if only_selected:
            return values[self.records_indices]
        return values

    def cell_data(self, column_name: str, index: int) -> float:
        """
        Get the data of a column.

        :param column_name: The header name of the desired cell.
        :type column_name: str
        :param index: The index of the desired cell.
        :type index: int
        :returns: The value of the cell
        :rtype: float
        """
        self.__validate_column_name(column_name)
        self.__validate_record_index(index)
        return self.data[column_name][index - 1]

    def record_data(self, index: int) -> np.ndarray:
        """
        Get the data of a column.

        :param index: The index of the desired record.
        :type index: int
        :returns: The record
        :rtype: numpy.ndarray
        """
        self.__validate_record_index(index)
        return np.array(
            [self.data[column_name][index - 1] for column_name in self.all_columns]
        )

    def column_domain(
        self, column_name: Optional[str], only_selected: bool = True
    ) -> Interval:
        """
        Get the smallest interval containing the values in a column.

        :param column_name: The desired column name.
        :type column_name: str
        :param only_selected: If true, return only values selected records. otherwise,
            Return values of all records.
        :type only_selected: bool
        :returns: Minimal interval containing column values
        :rtype: Interval
        :raises ValueError: if column_name is None, raising Value error
        """
        values = self.column_data(column_name=column_name, only_selected=only_selected)
        if values is None:
            raise ValueError("Column must be specified correctly to get its domain.")
        return Interval(min_val=np.min(values), max_val=np.max(values))

    def statistics(self, column_name: str) -> Optional[Statistics]:
        """
        Get statistics of the values in a column.

        :param column_name: The column name to get statistics of
        :type column_name: str
        :returns: Statistics of the given column
        :rtype: Statistics
        :raises FittingDataColumnExistenceError: Raised when unknown column name is
            given.
        """
        if column_name not in self.all_columns:
            raise FittingDataColumnExistenceError(column_name)
        return self._statistics_map.get(column_name, None)

    # Setter methods

    def set_header(self, old_column, new_column):
        """
        Rename header.

        :param old_column: The old columns name
        :type old_column: str
        :param new_column: The new value to set for the header
        :type new_column: str
        :raises FittingDataSetError: Raised when trying to set a header which is empty
            or already been set.
        """
        if new_column == old_column:
            return
        if new_column == "":
            raise FittingDataSetError("Cannot set new header to be empty")
        if new_column in self.all_columns:
            raise FittingDataSetError(
                f'The column name "{new_column}" is already used.'
            )
        self.__validate_column_name(old_column)
        self._data[new_column] = self._data.pop(old_column)
        self._all_columns = list(self.data.keys())
        for column_type, column_name in self.used_columns.items():
            if column_name == old_column:
                setattr(self, f"{column_type}_column", new_column)
        self.__update_statistics()

    def set_cell(self, column_name: str, index: int, value: float):
        """
        Set new value to a cell.

        :param column_name: The column name
        :type column_name: str
        :param index: The number of the record to set, starting from 1
        :type index: int
        :param value: The new value to set for the cell
        :type value: float
        :raises FittingDataSetError: Raised when trying to set a cell with non number
            value
        """
        if not isinstance(value, Number):
            raise FittingDataSetError(
                f'The cell at record number "{index}", '
                f'column "{column_name}", has invalid syntax: {value}.'
            )
        self.__validate_column_name(column_name=column_name)
        self.__validate_record_index(index)
        self._data[column_name][index - 1] = value
        self.__update_statistics()

    # Save methods

    def save_excel(
        self,
        output_directory: Union[str, Path],
        name: str = "fitting_data",
        sheet: Optional[str] = None,
    ):
        """
        Save :class:`FittingData` to xlsx file.

        :param output_directory: Path to the directory for the new excel file to be
            saved.
        :type output_directory: ``Path`` or ``str``
        :param name: Optional. The name of the file, without the .xlsx suffix.
            "fitting_data" by default.
        :type name: str
        :param sheet: Optional. Name of the sheet that the data will be saved to.
        :type sheet: str
        """
        io_util.save_as_excel(
            content=[self.all_columns, *self.all_records],
            output_directory=output_directory,
            file_name=name,
            sheet=sheet,
        )

    def save_csv(self, output_directory: Union[str, Path], name: str = "fitting_data"):
        """
        Save :class:`FittingData` to csv file.

        :param output_directory:
            Path to the directory for the new excel file to be saved.
        :type output_directory: ``Path`` or ``str``
        :param name: Optional. The name of the file, without the .csv suffix.
            "fitting_data" by default.
        :type name: str
        """
        io_util.save_as_csv(
            content=[self.all_columns, *self.all_records],
            output_directory=output_directory,
            file_name=name,
        )

    # Private methods

    def __update_statistics(self):
        self._statistics_map.clear()
        if self.non_selected():
            return
        for column in self.all_columns:
            self._statistics_map[column] = Statistics.from_array(
                self.column_data(column)
            )

    def __get_column_index(self, column_name):
        if column_name is None:
            return None
        return self.all_columns.index(column_name) + 1

    def __get_column_name(self, index):
        if index is None:
            return None
        return self.all_columns[index - 1]

    def __get_indices_in_interval(self, interval: Interval, column_name: str):
        return [value in interval for value in self.data[column_name]]

    def __validate_column_name(self, column_name):
        if column_name is None:
            return
        if column_name not in self.all_columns:
            raise FittingDataColumnExistenceError(column_name)

    def __validate_column_index(self, index):
        if index is None:
            return
        if index < 1 or index > self.number_of_columns:
            raise FittingDataColumnIndexError(index, self.number_of_columns)

    def __validate_record_index(self, index):
        if index < 1 or index > self.number_of_records:
            raise FittingDataRecordIndexError(index, self.number_of_records)

    @classmethod
    def __combine_records_indices(cls, *indices_lists):
        return [all(selected_tuple) for selected_tuple in zip(*indices_lists)]

    @classmethod
    def __build_from_rows(  # pylint: disable=too-many-arguments
        cls,
        rows,
        x_column: Optional[Union[str, int]] = None,
        xerr_column: Optional[Union[str, int]] = None,
        y_column: Optional[Union[str, int]] = None,
        yerr_column: Optional[Union[str, int]] = None,
        search: bool = True,
    ):
        data = RawDataBuilder.build_raw_data(rows)
        return FittingData(
            data=data,
            x_column=x_column,
            xerr_column=xerr_column,
            y_column=y_column,
            yerr_column=yerr_column,
            search=search,
        )
