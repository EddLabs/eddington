"""Registry containing predefined fit functions and fit functions generators."""
from prettytable import PrettyTable

from eddington_core.exceptions import FitFunctionLoadError, FitFunctionSaveError


class FitFunctionsRegistry:  # noqa: D415,D213,D205
    """A singleton class containing all saved :class:`FitFunction` and
    :class:`FitFunctionGenerator` isntances.
    """

    __name_to_func_dict = dict()  # type: ignore

    @classmethod
    def add(cls, func):
        """Add a fit function."""
        if func.name in cls.__name_to_func_dict:
            raise FitFunctionSaveError(
                f'Cannot save "{func.name}" to registry'
                " since there is another fit function with this name"
            )
        cls.__name_to_func_dict[func.name] = func

    @classmethod
    def remove(cls, func_name):
        """Remove a fit function."""
        del cls.__name_to_func_dict[func_name]

    @classmethod
    def clear(cls):
        """Clear all fit functions and generators."""
        cls.__name_to_func_dict.clear()

    @classmethod
    def all(cls):
        """Get all fit functions and generators."""
        return cls.__name_to_func_dict.values()

    @classmethod
    def names(cls):
        """Names of all fit functions and generators."""
        return cls.__name_to_func_dict.keys()

    @classmethod
    def load(cls, name):
        """Get a fit function and generators by name."""
        if not cls.exists(name):
            raise FitFunctionLoadError(f"No fit function named {name}")
        return cls.__name_to_func_dict[name]

    @classmethod
    def exists(cls, func_name):
        """Checks whether a fit function exist."""
        return func_name in cls.__name_to_func_dict

    @classmethod
    def list(cls):
        """Prints all fit functions and generators in a pretty table."""
        table = PrettyTable(field_names=["Function", "Syntax"])
        for func in FitFunctionsRegistry.all():
            table.add_row([func.signature, func.syntax])
        return table

    @classmethod
    def syntax(cls, functions):
        """Prints chosen fit functions and generators in a pretty table."""
        table = PrettyTable(field_names=["Function", "Syntax"])
        for func_name in functions:
            if cls.exists(func_name):
                func = cls.load(func_name)
                table.add_row([func.signature, func.syntax])
        return table
