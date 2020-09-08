from pytest_cases import THIS_MODULE, case, parametrize_with_cases

from eddington.cli import eddington_cli
from tests.util import dummy_function


def case_list_empty_registry(clear_functions_registry):
    args = []
    output = """
+----------+--------+
| Function | Syntax |
+----------+--------+
+----------+--------+
"""
    return args, output


def case_registry_with_one_function(clear_functions_registry):
    dummy_function("dummy_function1", "This is a short syntax")
    args = []
    output = """
+-----------------+------------------------+
|     Function    |         Syntax         |
+-----------------+------------------------+
| dummy_function1 | This is a short syntax |
+-----------------+------------------------+
"""
    return args, output


def case_registry_with_two_functions(clear_functions_registry):
    dummy_function("dummy_function1", "This is a short syntax")
    dummy_function("dummy_function2", "This is a much much much longer syntax")
    args = []
    output = """
+-----------------+----------------------------------------+
|     Function    |                 Syntax                 |
+-----------------+----------------------------------------+
| dummy_function1 |         This is a short syntax         |
| dummy_function2 | This is a much much much longer syntax |
+-----------------+----------------------------------------+
"""
    return args, output


def case_registry_with_three_functions(clear_functions_registry):
    dummy_function("dummy_function1", "This is a short syntax")
    dummy_function("dummy_function2", "This is a much much much longer syntax")
    dummy_function("blob", "I'm tiny")
    args = []
    output = """
+-----------------+----------------------------------------+
|     Function    |                 Syntax                 |
+-----------------+----------------------------------------+
| dummy_function1 |         This is a short syntax         |
| dummy_function2 | This is a much much much longer syntax |
|       blob      |                I'm tiny                |
+-----------------+----------------------------------------+
"""
    return args, output


def case_list_with_regex_flag(clear_functions_registry):
    dummy_function("shark", "Shark syntax")
    dummy_function("short", "Short syntax")
    dummy_function("sharp", "Sharp syntax")
    args = ["--regex=har"]
    output = """
+----------+--------------+
| Function |    Syntax    |
+----------+--------------+
|  shark   | Shark syntax |
|  sharp   | Sharp syntax |
+----------+--------------+
"""
    return args, output


@parametrize_with_cases(argnames=["args", "output"], cases=THIS_MODULE)
def test_eddington_list(args, output, cli_runner):
    result = cli_runner.invoke(eddington_cli, ["list", *args])
    assert result.exit_code == 0, "Result code should be successful"
    assert result.output == output[1:], "output is different than expected"
