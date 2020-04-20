class EddingtonException(Exception):
    pass


class InvalidGeneratorInitialization(EddingtonException):
    pass


class FitFunctionLoadError(EddingtonException):
    pass


class FitFunctionRuntimeError(EddingtonException):
    pass


class InvalidDataFile(EddingtonException):
    def __init__(self, file_name, sheet=None):
        sheet_msg = "" if sheet is None else f' in sheet "{sheet}"'
        msg = f'"{file_name}" has invalid syntax{sheet_msg}.'
        super(InvalidDataFile, self).__init__(msg)


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
