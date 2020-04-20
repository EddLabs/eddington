class EddingtonException(Exception):
    pass


class InvalidGeneratorInitialization(EddingtonException):
    pass


class FitFunctionLoadError(EddingtonException):
    pass


class FitFunctionRuntimeError(EddingtonException):
    pass


# Column Errors


class ColumnError(EddingtonException):
    pass


class ColumnIndexError(ColumnError):
    def __init__(self, index, max_index):
        super(ColumnIndexError, self).__init__(
            f"No column number {index} in data. "
            f"index should be between 1 and {max_index}"
        )


class ColumnExistenceError(ColumnError):
    def __init__(self, column):
        super(ColumnExistenceError, self).__init__(
            f'Could not find column "{column}" in data'
        )
