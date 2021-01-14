from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pytest
from pytest_cases import THIS_MODULE, case, parametrize_with_cases

from eddington import FittingDataInvalidFile
from eddington.raw_data_builder import RawDataBuilder
from tests.fitting_data import (
    COLUMNS,
    CONTENT,
    NUMBER_OF_COLUMNS,
    NUMBER_OF_RECORDS,
    ROWS,
)
from tests.util import assert_dict_equal

EPSILON = 1e-5
SUCCESS = "success"
FAILURE = "failure"
COMMENT = "THIS IS A COMMENT"


def random_blank_string(max_len=5, zero_included=True):
    str_len = np.random.randint(max_len)
    if not zero_included:
        str_len += 1
    return " " * str_len


# Successful cases


@case(tags=SUCCESS)
def case_simple_data_build_with_headers():
    rows = ROWS
    data = COLUMNS
    return rows, data


@case(tags=SUCCESS)
def case_simple_data_build_without_headers():
    rows = CONTENT
    data = {str(i): COLUMNS[key] for i, key in enumerate(COLUMNS)}
    return rows, data


@case(tags=SUCCESS)
def case_strings_data_build_without_headers():
    rows = [[str(val) for val in row] for row in CONTENT]
    data = {str(i): COLUMNS[key] for i, key in enumerate(COLUMNS)}
    return rows, data


@case(tags=SUCCESS)
def case_data_build_with_comment_after_headers():
    rows = deepcopy(ROWS)
    rows[0].extend([None, COMMENT])
    for i in range(1, NUMBER_OF_RECORDS + 1):
        rows[i].extend([None, None])
    data = COLUMNS
    return rows, data


@case(tags=SUCCESS)
def case_data_build_handle_spaces_as_blank():
    rows = deepcopy(ROWS)
    rows[0].extend([random_blank_string(), COMMENT])
    for i in range(1, NUMBER_OF_RECORDS + 1):
        rows[i].extend([random_blank_string(), random_blank_string()])
    data = COLUMNS
    return rows, data


@case(tags=SUCCESS)
def case_data_build_with_comments_row():
    rows = deepcopy(ROWS)
    rows.append([None for _ in range(NUMBER_OF_COLUMNS)])
    data = COLUMNS
    return rows, data


@parametrize_with_cases(argnames=["rows", "data"], cases=THIS_MODULE, has_tag=SUCCESS)
def test_successful_raw_data_build(rows, data):
    actual_data = RawDataBuilder.build_raw_data(rows)
    assert isinstance(actual_data, OrderedDict), "Data should be an ordered dictionary."
    assert_dict_equal(actual_data, data, EPSILON)


# Failure cases


@case(tags=FAILURE)
def case_data_build_fail_with_empty_rows():
    rows = []
    exception_class = FittingDataInvalidFile
    exception_regex = "^All rows are empty.$"
    return rows, exception_class, exception_regex


@case(tags=FAILURE)
def case_data_build_fail_with_a_none_value_in_row_middle():
    rows = deepcopy(ROWS)
    row = 7
    column = 8
    rows[row][column] = None
    exception_class = FittingDataInvalidFile
    exception_regex = f"^Empty cell at row {row} column {column}.$"
    return rows, exception_class, exception_regex


@case(tags=FAILURE)
def case_data_build_fail_with_a_empty_value_in_row_middle():
    rows = deepcopy(ROWS)
    row = 7
    column = 8
    rows[row][column] = random_blank_string(zero_included=False)
    exception_class = FittingDataInvalidFile
    exception_regex = f"^Empty cell at row {row} column {column}.$"
    return rows, exception_class, exception_regex


@case(tags=FAILURE)
def case_data_build_fail_with_a_non_number_value_in_row_middle():
    rows = deepcopy(ROWS)
    value = "blip"
    row = 7
    column = 8
    rows[row][column] = value
    exception_class = FittingDataInvalidFile
    exception_regex = (
        f'^Cell should be a number at row {row} column {column}, got "{value}".$'
    )
    return rows, exception_class, exception_regex


@case(tags=FAILURE)
def case_data_build_fail_with_a_none_value_in_row_end():
    rows = deepcopy(ROWS)
    row = 7
    rows[row][NUMBER_OF_COLUMNS - 1] = None
    exception_class = FittingDataInvalidFile
    exception_regex = f"^Empty cell at row {row} column {NUMBER_OF_COLUMNS - 1}.$"
    return rows, exception_class, exception_regex


@case(tags=FAILURE)
def case_data_build_fail_with_comment_right_after_header():
    rows = deepcopy(ROWS)
    commented_row = 4
    for i in range(1, NUMBER_OF_RECORDS + 1):
        if i == commented_row:
            rows[i].append(COMMENT)
        else:
            rows[i].append(random_blank_string())
    exception_class = FittingDataInvalidFile
    exception_regex = (
        f"^Cell should be empty at row {commented_row} "
        f"column {NUMBER_OF_COLUMNS + 1}.$"
    )
    return rows, exception_class, exception_regex


@case(tags=FAILURE)
def case_data_build_fail_because_of_duplicate_header():
    rows = deepcopy(ROWS)
    header = "z"
    col1, col2 = 2, 6
    rows[0][col1] = header
    rows[0][col2] = header
    exception_class = FittingDataInvalidFile
    exception_regex = f"^The following headers appear more than once: {header}$"
    return rows, exception_class, exception_regex


@parametrize_with_cases(
    argnames=["rows", "exception_class", "exception_regex"],
    cases=THIS_MODULE,
    has_tag=FAILURE,
)
def test_failed_raw_data_build(rows, exception_class, exception_regex):
    with pytest.raises(exception_class, match=exception_regex):
        RawDataBuilder.build_raw_data(rows)
