import numpy as np
import pytest
from pytest_cases import fixture, unpack_fixture

from eddington import FitData, linear

a = np.array([1, 2])


@fixture
def make_data():
    data = FitData.random(linear, a=a)
    return data, data.residuals(linear, a)


data, residuals_data = unpack_fixture(  # pylint: disable=unbalanced-tuple-unpacking
    "data, residuals_data", make_data
)


def test_columns_names(data, residuals_data):
    assert (
        data.x_column == residuals_data.x_column
    ), "X column name is different than expected."
    assert (
        data.xerr_column == residuals_data.xerr_column
    ), "X error column name is different than expected."
    assert (
        data.y_column == residuals_data.y_column
    ), "Y column name is different than expected."
    assert (
        data.yerr_column == residuals_data.yerr_column
    ), "Y error column name is different than expected."


def test_columns_values(data, residuals_data):
    y_residuals = data.y - linear(a, data.x)

    assert data.x == pytest.approx(
        residuals_data.x
    ), "X column is different than expected."
    assert data.xerr == pytest.approx(
        residuals_data.xerr
    ), "X error column is different than expected."
    assert y_residuals == pytest.approx(
        residuals_data.y
    ), "Y column is different than expected."
    assert data.yerr == pytest.approx(
        residuals_data.yerr
    ), "Y error column is different than expected."
