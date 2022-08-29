# pylint: disable=missing-class-docstring
"""Exception classes for Eddington."""


class EddingtonException(Exception):  # noqa: D101
    pass


# Interval Errors


class IntervalError(EddingtonException):  # noqa: D101
    pass


class IntervalIntersectionError(IntervalError):  # noqa: D101
    def __init__(self, *intervals):  # noqa: D107
        super().__init__(f"The intervals {intervals} do not intersect")


# Fitting Function Errors


class FittingFunctionParsingError(EddingtonException):  # noqa: D101
    pass


class FittingFunctionLoadError(EddingtonException):  # noqa: D101
    pass


class FittingFunctionSaveError(EddingtonException):  # noqa: D101
    pass


class FittingFunctionRuntimeError(EddingtonException):  # noqa: D101
    pass


# Fitting Data Errors


class FittingDataError(EddingtonException):  # noqa: D101
    pass


class FittingDataColumnsLengthError(FittingDataError):  # noqa: D101
    msg = "All columns in FittingData should have the same length"

    def __init__(self) -> None:  # noqa: D107
        super().__init__(self.msg)


class FittingDataInvalidFile(FittingDataError):  # noqa: D101
    pass


class FittingDataColumnIndexError(FittingDataError):  # noqa: D101
    def __init__(self, index: int, max_index: int) -> None:  # noqa: D107
        super().__init__(
            f"No column number {index} in data. "
            f"index should be between 1 and {max_index}"
        )


class FittingDataColumnExistenceError(FittingDataError):  # noqa: D101
    def __init__(self, column: str) -> None:  # noqa: D107
        super().__init__(f'Could not find column "{column}" in data')


class FittingDataRecordIndexError(FittingDataError):  # noqa: D101
    def __init__(self, index: int, number_of_records: int) -> None:  # noqa: D107
        super().__init__(
            f"Could not find record with index {index} in data. "
            f"Index should be between 1 and {number_of_records}."
        )


class FittingDataRecordsSelectionError(FittingDataError):  # noqa: D101
    pass


class FittingDataSetError(FittingDataError):  # noqa: D101
    pass


# Fitting Errors


class FittingError(EddingtonException):  # noqa: D101
    pass


# Plot Errors


class PlottingError(EddingtonException):  # noqa: D101
    pass


# CLI Errors


class EddingtonCLIError(EddingtonException):  # noqa: D101
    pass
