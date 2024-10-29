import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from utilities.grad_huberl2 import grad_huber_l2  

def test_scalar_above_rho():
    result = grad_huber_l2(2.0, 1.0)
    expected = np.array([[1.0]])
    assert_array_almost_equal(result, expected)

def test_scalar_below_rho():
    result = grad_huber_l2(0.5, 1.0)
    expected = np.array([[0.5]])
    assert_array_almost_equal(result, expected)

def test_vector_mixed_values():
    result = grad_huber_l2(np.array([-2, 1, 0.3, 4]), 1.0)
    expected = np.array([[-1.0, 1.0, 0.3, 1.0]])
    assert_array_almost_equal(result, expected)

def test_matrix_values():
    result = grad_huber_l2(np.array([[1, 2], [-1, -2]]), 2.0)
    expected = np.array([[0.5, 1.0], [-0.5, -1.0]])
    assert_array_almost_equal(result, expected)

def test_zero_input():
    result = grad_huber_l2(0, 1.0)
    expected = np.array([[0.0]])
    assert_array_almost_equal(result, expected)

def test_rho_type_error():
    with pytest.raises(TypeError, match="rho must be a number"):
        grad_huber_l2(1.0, "invalid")

def test_rho_value_error():
    with pytest.raises(ValueError, match="rho must be positive"):
        grad_huber_l2(1.0, -1.0)

def test_non_finite_values():
    with pytest.raises(ValueError, match="Input contains non-finite values"):
        grad_huber_l2(np.array([1.0, np.inf]), 1.0)
    with pytest.raises(ValueError, match="Input contains non-finite values"):
        grad_huber_l2(np.array([1.0, np.nan]), 1.0)

def test_1d_array_input():
    result = grad_huber_l2(np.array([1, 2, 3]), 2.0)
    expected = np.array([[0.5, 1.0, 1.0]])
    assert_array_almost_equal(result, expected)
