from eddington_core.fit_functions.fit_functions_registry import FitFunctionsRegistry


class FitFunctionGenerator:
    def __init__(self, generator_func, name, syntax=None, parameters=None):
        self.__generator_func = generator_func
        self.__name = name
        self.__syntax = syntax
        self.__parameters = parameters
        self.__signature = f"{name}({FitFunctionGenerator.__param_string(parameters)})"
        FitFunctionsRegistry.add(self)

    def __call__(self, *args, **kwargs):
        return self.__generator_func(*args, **kwargs)

    @property
    def name(self):
        return self.__name

    @property
    def syntax(self):
        return self.__syntax

    @property
    def parameters(self):
        return self.__parameters

    @property
    def signature(self):
        return self.__signature

    @classmethod
    def is_generator(cls):
        return True

    @classmethod
    def __param_string(cls, parameters):
        if isinstance(parameters, str):
            return parameters
        return ", ".join(parameters)


def fit_function_generator(parameters, name=None, syntax=None):
    def wrapper(generator):
        generator_name = generator.__name__ if name is None else name
        return FitFunctionGenerator(
            generator_func=generator,
            name=generator_name,
            syntax=syntax,
            parameters=parameters,
        )

    return wrapper
