import random

from pytest_cases import parametrize

from eddington import FittingData
from tests.fitting_data import COLUMNS, COLUMNS_NAMES, NUMBER_OF_RECORDS
from tests.util import assert_dict_equal, assert_equal_item

EPSILON = 1e-5


def test_copy_all_elements():
    fitting_data = FittingData(
        COLUMNS, x_column="c", xerr_column="a", y_column="e", yerr_column="g"
    )
    copied_data = fitting_data.copy()

    assert_dict_equal(copied_data.data, COLUMNS, rel=EPSILON)
    assert copied_data.used_columns == fitting_data.used_columns


@parametrize("header_name", COLUMNS_NAMES)
def test_rename_header_doesnt_affect_copy(header_name):
    new_header = "I am new"
    fitting_data = FittingData(
        COLUMNS, x_column="c", xerr_column="a", y_column="e", yerr_column="g"
    )
    copied_data = fitting_data.copy()
    fitting_data.set_header(header_name, new_header)

    assert_equal_item(
        copied_data.column_data(header_name),
        fitting_data.column_data(new_header),
        rel=EPSILON,
    )


@parametrize("column_type", ["x_column", "y_column", "xerr_column", "yerr_column"])
@parametrize("header_name", COLUMNS_NAMES)
def test_setting_header_as_column_doesnt_affect_copy(column_type, header_name):
    new_header = random.choice(COLUMNS_NAMES)
    fitting_data = FittingData(
        COLUMNS, x_column="c", xerr_column="a", y_column="e", yerr_column="g"
    )
    setattr(fitting_data, column_type, header_name)
    copied_data = fitting_data.copy()
    setattr(fitting_data, column_type, new_header)

    assert getattr(fitting_data, column_type) == new_header
    assert getattr(copied_data, column_type) == header_name


@parametrize("header_name", COLUMNS_NAMES)
def test_set_cell_doesnt_affect_copy(header_name):
    index = random.randint(1, NUMBER_OF_RECORDS)
    value = random.uniform(0, 1)
    fitting_data = FittingData(
        COLUMNS, x_column="c", xerr_column="a", y_column="e", yerr_column="g"
    )
    old_value = fitting_data.column_data(header_name)[index - 1]
    copied_data = fitting_data.copy()
    fitting_data.set_cell(column_name=header_name, index=index, value=value)

    assert fitting_data.cell_data(column_name=header_name, index=index) == value
    assert copied_data.cell_data(column_name=header_name, index=index) == old_value
