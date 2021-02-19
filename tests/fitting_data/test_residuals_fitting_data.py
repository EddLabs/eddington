import numpy as np
import pytest

from eddington import FittingDataError, linear, random_data

A = np.array([1, 2])


def test_residuals_data_columns_names():
    data = random_data(linear, a=A)
    residuals_data = data.residuals(fit_func=linear, a=A)
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


def test_residuals_data_columns_values():
    data = random_data(linear, a=A)
    residuals_data = data.residuals(fit_func=linear, a=A)
    y_residuals = data.y - linear(A, data.x)

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


def test_residuals_data_without_xerr_column():
    data = random_data(linear, a=A)
    data.xerr_column = None
    residuals_data = data.residuals(fit_func=linear, a=A)
    y_residuals = data.y - linear(A, data.x)

    assert data.x == pytest.approx(
        residuals_data.x
    ), "X column is different than expected."
    assert data.xerr is None, "X error column should be None."
    assert y_residuals == pytest.approx(
        residuals_data.y
    ), "Y column is different than expected."
    assert data.yerr == pytest.approx(
        residuals_data.yerr
    ), "Y error column is different than expected."


def test_residuals_data_without_yerr_column():
    data = random_data(linear, a=A)
    data.yerr_column = None
    residuals_data = data.residuals(fit_func=linear, a=A)
    y_residuals = data.y - linear(A, data.x)

    assert data.x == pytest.approx(
        residuals_data.x
    ), "X column is different than expected."
    assert data.xerr == pytest.approx(
        residuals_data.xerr
    ), "X error column is different than expected."
    assert y_residuals == pytest.approx(
        residuals_data.y
    ), "Y column is different than expected."
    assert data.yerr is None, "Y error column should be None."


def test_residuals_data_without_x_column():
    data = random_data(linear, a=A)
    data.x_column = None

    with pytest.raises(
        FittingDataError,
        match="^Could not calculate residuals data without x values.$",
    ):
        data.residuals(fit_func=linear, a=A)


def test_residuals_data_without_y_column():
    data = random_data(linear, a=A)
    data.y_column = None

    with pytest.raises(
        FittingDataError,
        match="^Could not calculate residuals data without y values.$",
    ):
        data.residuals(fit_func=linear, a=A)
