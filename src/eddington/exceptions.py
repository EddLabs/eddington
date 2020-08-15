# pylint: disable=missing-class-docstring
"""Exception classes for Eddington."""


class EddingtonException(Exception):  # noqa: D101
    pass


# Fit Function Errors


class FitFunctionLoadError(EddingtonException):  # noqa: D101
    pass


class FitFunctionSaveError(EddingtonException):  # noqa: D101
    pass


class FitFunctionRuntimeError(EddingtonException):  # noqa: D101
    pass


# Fit Data Errors


class FitDataError(EddingtonException):  # noqa: D101
    pass


class FitDataInvalidFile(FitDataError):  # noqa: D101
    pass


class FitDataInvalidFileSyntax(FitDataInvalidFile):  # noqa: D101
    def __init__(self, file_name, sheet=None):  # noqa: D107
        sheet_msg = "" if sheet is None else f' in sheet "{sheet}"'
        msg = f'"{file_name}" has invalid syntax{sheet_msg}.'
        super(FitDataInvalidFileSyntax, self).__init__(msg)


class FitDataColumnsLengthError(FitDataInvalidFile):  # noqa: D101
    msg = "All columns in FitData should have the same length"

    def __init__(self):  # noqa: D107
        super(FitDataColumnsLengthError, self).__init__(self.msg)


class FitDataColumnIndexError(FitDataError):  # noqa: D101
    def __init__(self, index, max_index):  # noqa: D107
        super(FitDataColumnIndexError, self).__init__(
            f"No column number {index} in data. "
            f"index should be between 1 and {max_index}"
        )


class FitDataColumnExistenceError(FitDataError):  # noqa: D101
    def __init__(self, column):  # noqa: D107
        super(FitDataColumnExistenceError, self).__init__(
            f'Could not find column "{column}" in data'
        )


class FitDataColumnsSelectionError(FitDataError):  # noqa: D101
    pass


class FitDataInvalidSyntax(FitDataError):  # noqa: D101
    def __init__(self, col, row, value):  # noqa: D107
        msg = f'the cell at row:"{row}", column:"{col}" has invalid syntax{value}.'
        super(FitDataInvalidSyntax, self).__init__(msg)


class FitDataColumnAlreadyExists(FitDataError):  # noqa: D101
    def __init__(self, col):  # noqa: D107
        msg = f'the column name:"{col}" is already used.'
        super(FitDataColumnAlreadyExists, self).__init__(msg)
