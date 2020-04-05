class EddingtonException(Exception):
    pass


class InvalidGeneratorInitialization(EddingtonException):
    pass


class FitFunctionLoadError(EddingtonException):
    pass


class FitFunctionRuntimeError(EddingtonException):
    pass
