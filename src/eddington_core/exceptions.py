class EddingtonException(Exception):
    pass


# Fit Function Errors


class InvalidGeneratorInitialization(EddingtonException):
    pass


class FitFunctionLoadError(EddingtonException):
    pass


class FitFunctionRuntimeError(EddingtonException):
    pass


# Fit Data Errors


class FitDataError(EddingtonException):
    pass


class FitDataInvalidFile(FitDataError):
    pass


class FitDataInvalidFileSyntax(FitDataInvalidFile):
    def __init__(self, file_name, sheet=None):
        sheet_msg = "" if sheet is None else f' in sheet "{sheet}"'
        msg = f'"{file_name}" has invalid syntax{sheet_msg}.'
        super(FitDataInvalidFileSyntax, self).__init__(msg)


class FitDataColumnsLengthError(FitDataInvalidFile):
    msg = "All columns in FitData should have the same length"

    def __init__(self):
        super(FitDataColumnsLengthError, self).__init__(self.msg)


class FitDataColumnIndexError(FitDataError):
    def __init__(self, index, max_index):
        super(FitDataColumnIndexError, self).__init__(
            f"No column number {index} in data. "
            f"index should be between 1 and {max_index}"
        )


class FitDataColumnExistenceError(FitDataError):
    def __init__(self, column):
        super(FitDataColumnExistenceError, self).__init__(
            f'Could not find column "{column}" in data'
        )
