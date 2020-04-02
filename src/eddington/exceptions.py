class EddingtonException(Exception):
    pass


class InvalidGeneratorInitialization(EddingtonException):
    pass


class FitFunctionLoadError(EddingtonException):
    pass
