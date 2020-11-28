# pylint: disable=missing-class-docstring
"""Exception classes for Eddington."""
from pathlib import Path
from typing import Optional, Union


class EddingtonException(Exception):  # noqa: D101
    pass


# Fitting Function Errors


class FittingFunctionLoadError(EddingtonException):  # noqa: D101
    pass


class FittingFunctionSaveError(EddingtonException):  # noqa: D101
    pass


class FittingFunctionRuntimeError(EddingtonException):  # noqa: D101
    pass


# Fitting Data Errors


class FittingDataError(EddingtonException):  # noqa: D101
    pass


class FittingDataInvalidFile(FittingDataError):  # noqa: D101
    pass


class FittingDataHeaderDuplication(FittingDataError):  # noqa: D101
    def __init__(self, filepath, duplicate_headers):  # noqa: D107
        super().__init__(
            f'The following headers appear more than once in "{filepath}": '
            f'{", ".join(duplicate_headers)}'
        )


class FittingDataInvalidFileSyntax(FittingDataInvalidFile):  # noqa: D101
    def __init__(
        self, filepath: Union[str, Path], sheet: Optional[str] = None
    ) -> None:  # noqa: D107
        sheet_msg = "" if sheet is None else f' in sheet "{sheet}"'
        msg = f'"{filepath}" has invalid syntax{sheet_msg}.'
        super().__init__(msg)


class FittingDataColumnsLengthError(FittingDataInvalidFile):  # noqa: D101
    msg = "All columns in FittingData should have the same length"

    def __init__(self) -> None:  # noqa: D107
        super().__init__(self.msg)


class FittingDataColumnIndexError(FittingDataError):  # noqa: D101
    def __init__(self, index: int, max_index: int) -> None:  # noqa: D107
        super().__init__(
            f"No column number {index} in data. "
            f"index should be between 1 and {max_index}"
        )


class FittingDataColumnExistenceError(FittingDataError):  # noqa: D101
    def __init__(self, column: str) -> None:  # noqa: D107
        super().__init__(f'Could not find column "{column}" in data')


class FittingDataColumnsSelectionError(FittingDataError):  # noqa: D101
    pass


class FittingDataSetError(FittingDataError):  # noqa: D101
    pass


# Plot Errors


class PlottingError(EddingtonException):  # noqa: D101
    pass
