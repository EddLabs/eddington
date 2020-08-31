"""Registry containing predefined fit functions and fit functions generators."""
from typing import List

from eddington.exceptions import FitFunctionLoadError, FitFunctionSaveError


class FitFunctionsRegistry:  # noqa: D415,D213,D205
    """A singleton class containing all saved :class:`FitFunction` instances."""

    __name_to_func_dict = dict()  # type: ignore

    @classmethod
    def add(cls, func):  # type: ignore
        """
        Add a fit function.

        :param func: fit function to add to registry
        :type func: :class:`FitFunction`
        """
        if func.name in cls.__name_to_func_dict:
            raise FitFunctionSaveError(
                f'Cannot save "{func.name}" to registry'
                " since there is another fit function with this name"
            )
        cls.__name_to_func_dict[func.name] = func

    @classmethod
    def remove(cls, func_name: str) -> None:
        """
        Remove a fit function.

        :param func_name: Name of the function to remove.
        :type func_name: str
        """
        del cls.__name_to_func_dict[func_name]

    @classmethod
    def clear(cls) -> None:
        """Clear all fit functions and generators."""
        cls.__name_to_func_dict.clear()

    @classmethod
    def all(cls):  # type: ignore
        """
        Get all fit functions and generators.

        :returns: list of :class:`FitFunction`
        """
        return list(cls.__name_to_func_dict.values())

    @classmethod
    def names(cls) -> List[str]:
        """Names of all fit functions and generators."""
        return list(cls.__name_to_func_dict.keys())

    @classmethod
    def load(cls, func_name):  # type: ignore
        """
        Get a fit function and generators by name.

        :param func_name: Name of the function to load.
        :type func_name: str
        :returns: :class:`FitFunction`
        """
        if not cls.exists(func_name):
            raise FitFunctionLoadError(f"No fit function named {func_name}")
        return cls.__name_to_func_dict[func_name]

    @classmethod
    def exists(cls, func_name: str) -> bool:
        """
        Checks whether a fit function exist.

        :param func_name: Name of the function to load.
        :type func_name: str
        :returns: bool
        """
        return func_name in cls.__name_to_func_dict
