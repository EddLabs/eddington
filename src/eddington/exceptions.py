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
        super().__init__(msg)


class FitDataColumnsLengthError(FitDataInvalidFile):  # noqa: D101
    msg = "All columns in FitData should have the same length"

    def __init__(self):  # noqa: D107
        super().__init__(self.msg)


class FitDataColumnIndexError(FitDataError):  # noqa: D101
    def __init__(self, index, max_index):  # noqa: D107
        super().__init__(
            f"No column number {index} in data. "
            f"index should be between 1 and {max_index}"
        )


class FitDataColumnExistenceError(FitDataError):  # noqa: D101
    def __init__(self, column):  # noqa: D107
        super().__init__(f'Could not find column "{column}" in data')


class FitDataColumnsSelectionError(FitDataError):  # noqa: D101
    pass
