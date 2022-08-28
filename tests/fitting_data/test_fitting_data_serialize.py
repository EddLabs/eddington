from collections import OrderedDict
from typing import Any, Dict

import numpy as np
import pytest_cases
from pytest_cases import THIS_MODULE

from eddington import FittingData
from tests.util import random_selected_records


def random_raw_data(columns, size):
    return {column: np.random.uniform(100, size=size).tolist() for column in columns}


def case_simple_data_serialization():
    columns = ["a", "b", "c", "d", "e", "f", "g"]
    size = 10
    raw_data = random_raw_data(columns=columns, size=size)
    x_column, xerr_column, y_column, yerr_column = np.random.choice(
        columns, size=4, replace=False
    )
    fitting_data = FittingData(
        data=raw_data,
        x_column=x_column,
        xerr_column=xerr_column,
        y_column=y_column,
        yerr_column=yerr_column,
    )
    serialized_data = dict(
        data=raw_data,
        x_column=x_column,
        xerr_column=xerr_column,
        y_column=y_column,
        yerr_column=yerr_column,
        indices=[True] * size,
    )
    return fitting_data, serialized_data


def case_data_serialization_with_selected_records():
    columns = ["a", "b", "c", "d", "e", "f", "g"]
    size = 10
    raw_data = random_raw_data(columns=columns, size=size)
    x_column, xerr_column, y_column, yerr_column = np.random.choice(
        columns, size=4, replace=False
    )
    records_indices = random_selected_records(records_num=size)
    fitting_data = FittingData(
        data=raw_data,
        x_column=x_column,
        xerr_column=xerr_column,
        y_column=y_column,
        yerr_column=yerr_column,
    )
    fitting_data.records_indices = records_indices
    serialized_data = dict(
        data=raw_data,
        x_column=x_column,
        xerr_column=xerr_column,
        y_column=y_column,
        yerr_column=yerr_column,
        indices=records_indices,
    )
    return fitting_data, serialized_data


def case_data_serialization_with_no_xerr_column():
    columns = ["a", "b", "c", "d", "e", "f", "g"]
    size = 10
    raw_data = random_raw_data(columns=columns, size=size)
    x_column, y_column, yerr_column = np.random.choice(columns, size=3, replace=False)
    fitting_data = FittingData(
        data=raw_data,
        x_column=x_column,
        y_column=y_column,
        yerr_column=yerr_column,
        search=False,
    )
    serialized_data = dict(
        data=raw_data,
        x_column=x_column,
        xerr_column=None,
        y_column=y_column,
        yerr_column=yerr_column,
        indices=[True] * size,
    )
    return fitting_data, serialized_data


def case_data_serialization_with_no_yerr_column():
    columns = ["a", "b", "c", "d", "e", "f", "g"]
    size = 10
    raw_data = random_raw_data(columns=columns, size=size)
    x_column, y_column, xerr_column = np.random.choice(columns, size=3, replace=False)
    fitting_data = FittingData(
        data=raw_data,
        x_column=x_column,
        y_column=y_column,
        xerr_column=xerr_column,
        search=False,
    )
    serialized_data = dict(
        data=raw_data,
        x_column=x_column,
        xerr_column=xerr_column,
        y_column=y_column,
        yerr_column=None,
        indices=[True] * size,
    )
    return fitting_data, serialized_data


@pytest_cases.parametrize_with_cases(
    argnames=["fitting_data", "serialized_data"], cases=THIS_MODULE
)
def test_serialize_fitting_data(
    fitting_data: FittingData, serialized_data: Dict[str, Any]
):
    actual_serialized_data = fitting_data.serialize()
    assert isinstance(actual_serialized_data, dict)
    assert set(actual_serialized_data.keys()) == {
        "data",
        "x_column",
        "y_column",
        "xerr_column",
        "yerr_column",
        "indices",
    }
    actual_raw_data = actual_serialized_data["data"]
    assert isinstance(actual_raw_data, OrderedDict)
    assert list(actual_raw_data.keys()) == list(serialized_data["data"].keys())
    for column in actual_serialized_data["data"].keys():
        assert isinstance(actual_raw_data[column], list)
        assert not isinstance(actual_raw_data[column], np.ndarray)
        assert len(actual_raw_data[column]) == len(serialized_data["data"][column])
        np.testing.assert_almost_equal(
            actual_raw_data[column], serialized_data["data"][column]
        )
    assert actual_serialized_data["x_column"] == serialized_data["x_column"]
    assert actual_serialized_data["xerr_column"] == serialized_data["xerr_column"]
    assert actual_serialized_data["y_column"] == serialized_data["y_column"]
    assert actual_serialized_data["yerr_column"] == serialized_data["yerr_column"]
    assert actual_serialized_data["indices"] == serialized_data["indices"]


@pytest_cases.parametrize_with_cases(
    argnames=["fitting_data", "serialized_data"], cases=THIS_MODULE
)
def test_deserialize_fitting_data(
    fitting_data: FittingData, serialized_data: Dict[str, Any]
):
    actual_fitting_data = FittingData.deserialize(serialized_data)
    assert isinstance(actual_fitting_data, FittingData)
    assert actual_fitting_data.number_of_records == fitting_data.number_of_records
    assert actual_fitting_data.all_columns == fitting_data.all_columns
    for column in fitting_data.all_columns:
        np.testing.assert_almost_equal(
            actual_fitting_data.column_data(column, only_selected=False),
            fitting_data.column_data(column, only_selected=False),
        )
    assert actual_fitting_data.x_column == fitting_data.x_column
    assert actual_fitting_data.xerr_column == fitting_data.xerr_column
    assert actual_fitting_data.y_column == fitting_data.y_column
    assert actual_fitting_data.yerr_column == fitting_data.yerr_column
    assert actual_fitting_data.records_indices == fitting_data.records_indices
