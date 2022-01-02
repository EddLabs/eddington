"""A helper class for building raw dat for fitting."""
import collections
from typing import List, Optional, Union

from eddington.exceptions import FittingDataInvalidFile


class RawDataBuilder:
    """Builder of raw data from file rows."""

    @classmethod
    def build_raw_data(cls, rows: List[List[Optional[str]]]) -> collections.OrderedDict:
        """
        Convert list of rows into a raw OrderedDict.

        That can be used as data for the FittingData class.

        :param rows: List of lists of strings. Values read from a data file (mostly
            excel file or csv file)
        :type rows: List[List[Optional[str]]]
        :return: Data as an ordered dictionary.
        :rtype: collections.OrderedDict
        :raises FittingDataInvalidFile: Raised when all rows are empty or no rows were
            given.
        """
        rows = cls.__trim_data(rows)
        if len(rows) == 0:
            raise FittingDataInvalidFile("All rows are empty.")
        headers, content = cls.__extract_headers(rows)
        cls.__validate_headers(headers)
        columns = list(zip(*content))
        raw_dict = collections.OrderedDict(zip(headers, columns))
        return cls.fix_types_in_raw_dict(raw_dict)

    @classmethod
    def fix_types_in_raw_dict(
        cls, raw_dict: collections.OrderedDict
    ) -> collections.OrderedDict:
        """
        Convert the types of a given raw dictionary into numpy array with floats.

        :param raw_dict: Raw data with values as a strings or floats.
        :type raw_dict: collections.OrderedDict
        :return: New raw data with values as floats
        :rtype: collections.OrderedDict
        """
        new_dict = collections.OrderedDict()
        for column, key in enumerate(raw_dict.keys()):
            new_dict[key] = cls.__convert_column(
                column_number=column, column=raw_dict[key]
            )
        return new_dict

    @classmethod
    def __trim_data(cls, rows):
        if len(rows) == 0:
            return rows
        first_row = rows[0]
        row_length = None
        for i, val in enumerate(first_row):
            if cls.__is_empty_value(val):
                row_length = i
                break
        if row_length is None:
            row_length = len(first_row)
        new_rows = []
        for i, row in enumerate(rows):
            row = list(row)
            if len(row) > row_length:
                if not cls.__is_empty_value(row[row_length]):
                    raise FittingDataInvalidFile(
                        f"Cell should be empty at row {i} column {row_length + 1}."
                    )
                row = row[:row_length]
            while len(row) != 0 and cls.__is_empty_value(row[-1]):
                row = row[:-1]
            if len(row) == 0:
                break
            if len(row) < row_length:
                row.extend([None for _ in range(row_length - len(row))])
            new_rows.append(row)
        return new_rows

    @classmethod
    def __extract_headers(cls, rows: List[List[Optional[str]]]):
        headers: List[Optional[str]] = rows[0]
        if cls.__are_headers(headers):
            content = rows[1:]
        else:
            headers = [str(i) for i in range(len(headers))]
            content = rows
        return headers, content

    @classmethod
    def __validate_headers(cls, headers):
        duplicate_headers = [
            item for item, count in collections.Counter(headers).items() if count > 1
        ]
        if len(duplicate_headers) != 0:
            raise FittingDataInvalidFile(
                f"The following headers appear more than once: "
                f'{", ".join(duplicate_headers)}'
            )

    @classmethod
    def __convert_column(cls, column_number, column):
        return [
            cls.__convert_cell(row_number, column_number, val)
            for row_number, val in enumerate(column, start=1)
        ]

    @classmethod
    def __convert_cell(cls, row_number, column_number, val):
        if isinstance(val, str):
            val = val.strip()
        if cls.__is_empty_value(val):
            raise FittingDataInvalidFile(
                f"Empty cell at row {row_number} column {column_number}."
            )
        try:
            return float(val)
        except ValueError as error:
            raise FittingDataInvalidFile(
                f"Cell should be a number at row {row_number} column {column_number}, "
                f'got "{val}".'
            ) from error

    @classmethod
    def __is_empty_value(cls, val: Optional[Union[str, float]]):
        if val is None:
            return True
        if isinstance(val, str) and val.strip() == "":
            return True
        return False

    @classmethod
    def __are_headers(cls, headers):
        return all(cls.__is_header(header) for header in headers)

    @classmethod
    def __is_header(cls, string):
        return isinstance(string, str) and not cls.__is_number(string)

    @classmethod
    def __is_number(cls, string):
        try:
            float(string)
            return True
        except ValueError:
            return False
