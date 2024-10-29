# File: test_l2_estimation.py
import numpy as np
import pytest
from utilities.estimate_l2 import estimate_l2


@pytest.fixture
def random_data():
    """Fixture to provide random test data."""
    n1, n2 = 3, 2
    x = np.random.rand(n1, n2)
    b = np.random.rand(n1 * n2)
    M = np.eye(n1 * n2)
    return x, b, M


def test_basic_functionality(random_data):
    """Test basic functionality with identity matrix."""
    x, b, M = random_data
    n1, n2 = x.shape
    
    criterion, gradient = estimate_l2(x, b, M)
    
    # Manual calculations for verification
    x_flat = x.flatten()
    expected_criterion = 0.5 * np.sum((x_flat - b)**2)
    expected_gradient = (x_flat - b).reshape(n1, n2)
    
    np.testing.assert_almost_equal(criterion, expected_criterion)
    np.testing.assert_array_almost_equal(gradient, expected_gradient)


def test_custom_matrix():
    """Test with a custom M matrix."""
    n1, n2 = 2, 2
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([1.0, 2.0, 3.0, 4.0])
    M = np.array([
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 2.0]
    ])
    
    criterion, gradient = estimate_l2(x, b, M)
    
    # Manual calculations for verification
    x_flat = x.flatten()
    expected_criterion = 0.5 * np.sum((2 * x_flat - b)**2)
    expected_gradient = (2 * (2 * x_flat - b)).reshape(n1, n2)
    
    np.testing.assert_almost_equal(criterion, expected_criterion)
    np.testing.assert_array_almost_equal(gradient, expected_gradient)


def test_invalid_dimensions():
    """Test error handling for incompatible dimensions."""
    x = np.random.rand(3, 2)
    b = np.random.rand(5)  # Wrong dimension
    M = np.eye(5)  # Wrong dimension
    
    with pytest.raises(ValueError):
        estimate_l2(x, b, M)


@pytest.mark.parametrize("n1,n2", [(2,3), (3,2), (4,4)])
def test_different_dimensions(n1: int, n2: int):
    """Test function with different matrix dimensions."""
    x = np.random.rand(n1, n2)
    b = np.random.rand(n1 * n2)
    M = np.eye(n1 * n2)
    
    criterion, gradient = estimate_l2(x, b, M)
    
    assert criterion.ndim == 0  # Scalar
    assert gradient.shape == (n1, n2)


def test_zero_input():
    """Test with zero inputs."""
    n1, n2 = 2, 2
    x = np.zeros((n1, n2))
    b = np.zeros(n1 * n2)
    M = np.eye(n1 * n2)
    
    criterion, gradient = estimate_l2(x, b, M)
    
    assert criterion == 0
    np.testing.assert_array_equal(gradient, np.zeros((n1, n2)))


def test_numerical_stability():
    """Test numerical stability with very large and very small numbers."""
    n1, n2 = 2, 2
    x = np.array([[1e-10, 1e-10], [1e-10, 1e-10]])
    b = np.array([1e-10, 1e-10, 1e-10, 1e-10])
    M = np.eye(n1 * n2)
    
    criterion, gradient = estimate_l2(x, b, M)
    
    assert np.isfinite(criterion)
    assert np.all(np.isfinite(gradient))