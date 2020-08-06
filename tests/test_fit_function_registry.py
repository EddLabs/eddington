import pytest

from eddington import (
    FitFunctionLoadError,
    FitFunctionsRegistry,
    fit_function,
)
from eddington.exceptions import FitFunctionSaveError


def dummy_function(value, save=True):
    @fit_function(
        n=2,
        name=f"dummy_function{value}",
        syntax=f"syntax_dummy_function{value}",
        save=save,
    )
    def dummy_func(a, x):  # pylint: disable=W0613
        return value

    return dummy_func


@pytest.fixture
def dummy_functions():
    backup = list(FitFunctionsRegistry.all())
    FitFunctionsRegistry.clear()
    func1 = dummy_function(1)
    func2 = dummy_function(2, save=False)
    func3 = dummy_function(3)
    func4 = dummy_function(4)
    func5 = dummy_function(5, save=False)

    yield dict(
        saved_funcs=dict(func1=func1, func3=func3, func4=func4),
        unsaved_funcs=[func2, func5],
    )

    FitFunctionsRegistry.clear()
    for func in backup:
        FitFunctionsRegistry.add(func)


def test_load_function(dummy_functions):
    for func in dummy_functions["saved_funcs"].values():
        actual_func = FitFunctionsRegistry.load(func.name)
        assert (
            func == actual_func
        ), "Registry get name function returns different function than expected."


def test_exists(dummy_functions):
    for func in dummy_functions["saved_funcs"].values():
        assert FitFunctionsRegistry.exists(
            func.name
        ), f"Expected {func.name} to exists. It does not."


def test_remove(dummy_functions):
    for func in dummy_functions["saved_funcs"].values():
        FitFunctionsRegistry.remove(func.name)
        assert not FitFunctionsRegistry.exists(
            func.name
        ), f"Expected {func.name} to not exist. It does."


def test_all(dummy_functions):
    assert list(dummy_functions["saved_funcs"].values()) == list(
        FitFunctionsRegistry.all()
    ), "Functions are different than expected"


def test_names(dummy_functions):
    assert [func.name for func in dummy_functions["saved_funcs"].values()] == list(
        FitFunctionsRegistry.names()
    ), "Functions names are different than expected"


def test_syntax_of_one_function(dummy_functions):
    syntax = FitFunctionsRegistry.syntax([dummy_functions["saved_funcs"]["func1"].name])
    expected_string = """
+-----------------+------------------------+
|     Function    |         Syntax         |
+-----------------+------------------------+
| dummy_function1 | syntax_dummy_function1 |
+-----------------+------------------------+"""
    assert expected_string[1:] == str(
        syntax
    ), "Syntax of one function is different than expected."


def test_syntax_of_two_functions(dummy_functions):
    syntax = FitFunctionsRegistry.syntax(
        [
            dummy_functions["saved_funcs"]["func1"].name,
            dummy_functions["saved_funcs"]["func3"].name,
        ]
    )
    expected_string = """
+-----------------+------------------------+
|     Function    |         Syntax         |
+-----------------+------------------------+
| dummy_function1 | syntax_dummy_function1 |
| dummy_function3 | syntax_dummy_function3 |
+-----------------+------------------------+"""
    assert expected_string[1:] == str(
        syntax
    ), "Syntax of two functions is different than expected."


def test_list(dummy_functions):
    syntax = FitFunctionsRegistry.list()
    expected_string = """
+-----------------+------------------------+
|     Function    |         Syntax         |
+-----------------+------------------------+
| dummy_function1 | syntax_dummy_function1 |
| dummy_function3 | syntax_dummy_function3 |
| dummy_function4 | syntax_dummy_function4 |
+-----------------+------------------------+"""
    assert (
        expected_string.strip() == str(syntax).strip()
    ), "Syntax of two functions is different than expected."


def test_add_dummy_function_without_saving(dummy_functions):
    for func in dummy_functions["unsaved_funcs"]:
        assert not FitFunctionsRegistry.exists(
            func.name
        ), "Function was saved, even though it wasn't supposed to"


def test_load_non_existing_function(dummy_functions):
    for func in dummy_functions["unsaved_funcs"]:
        with pytest.raises(
            FitFunctionLoadError, match=f"^No fit function named {func.name}$"
        ):
            FitFunctionsRegistry.load(func.name)


def test_saving_two_fit_functions_with_the_same_name(dummy_functions):
    with pytest.raises(
        FitFunctionSaveError,
        match='^Cannot save "dummy_function3" to registry '
        "since there is another fit function with this name$",
    ):
        dummy_function(value=3, save=True)
