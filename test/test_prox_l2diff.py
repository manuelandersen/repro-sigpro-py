import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from utilities.prox_l2diff import prox_l2diff


@pytest.fixture
def test_data():
    """Fixture providing basic test data."""
    n1, n2 = 3, 2
    x = np.random.rand(n1, n2)
    b = np.random.rand(n1 * n2)
    M = np.eye(n1 * n2)
    gamma = 0.5
    Minv = 1.0 / (1.0 + gamma)
    return x, b, M, gamma, Minv


def test_basic_functionality(test_data):
    """Test basic functionality with identity matrix."""
    x, b, M, gamma, Minv = test_data
    n1, n2 = x.shape
    
    result = prox_l2diff(x, b, M, gamma, Minv)
    
    # Manual calculation
    expected = Minv * (gamma * M.T @ b + x.flatten())
    expected = expected.reshape(n1, n2)
    
    assert_array_almost_equal(result, expected)
    assert result.shape == (n1, n2)


def test_scaled_matrix():
    """Test with scaled identity matrix."""
    n1, n2 = 3, 2
    x = np.random.rand(n1, n2)
    b = np.random.rand(n1 * n2)
    M = 2.0 * np.eye(n1 * n2)
    gamma = 0.3
    Minv = 1.0 / (1.0 + gamma * 4.0)  # 4 comes from M.T @ M
    
    result = prox_l2diff(x, b, M, gamma, Minv)
    
    # Manual calculation
    expected = Minv * (gamma * M.T @ b + x.flatten())
    expected = expected.reshape(n1, n2)
    
    assert_array_almost_equal(result, expected)


def test_different_dimensions():
    """Test with different matrix dimensions."""
    n1, n2 = 4, 3
    x = np.random.rand(n1, n2)
    b = np.random.rand(n1 * n2)
    M = np.eye(n1 * n2)
    gamma = 0.7
    Minv = 1.0 / (1.0 + gamma)
    
    result = prox_l2diff(x, b, M, gamma, Minv)
    assert result.shape == (n1, n2)


def test_zero_inputs():
    """Test with zero inputs."""
    n1, n2 = 3, 2
    x = np.zeros((n1, n2))
    b = np.zeros(n1 * n2)
    M = np.eye(n1 * n2)
    gamma = 0.5
    Minv = 1.0 / (1.0 + gamma)
    
    result = prox_l2diff(x, b, M, gamma, Minv)
    assert_array_equal(result, np.zeros((n1, n2)))


def test_numerical_stability():
    """Test numerical stability with very small numbers."""
    n1, n2 = 3, 2
    x = np.full((n1, n2), 1e-10)
    b = np.full(n1 * n2, 1e-10)
    M = np.eye(n1 * n2)
    gamma = 0.5
    Minv = 1.0 / (1.0 + gamma)
    
    result = prox_l2diff(x, b, M, gamma, Minv)
    
    assert np.all(np.isfinite(result))
    assert not np.any(np.isnan(result))


def test_invalid_parameters():
    """Test error handling for invalid parameters."""
    n1, n2 = 3, 2
    x = np.random.rand(n1, n2)
    b = np.random.rand(n1 * n2)
    M = np.eye(n1 * n2)
    
    with pytest.raises(ValueError):
        prox_l2diff(x, b, M, -1.0, 1.0)  # negative gamma
    
    with pytest.raises(ValueError):
        prox_l2diff(x, b, M, 1.0, -1.0)  # negative Minv


def test_invalid_dimensions():
    """Test error handling for incompatible dimensions."""
    n1, n2 = 3, 2
    x = np.random.rand(n1, n2)
    b = np.random.rand(n1 * n2 + 1)  # wrong size
    M = np.eye(n1 * n2)
    gamma = 0.5
    Minv = 1.0 / (1.0 + gamma)
    
    with pytest.raises(ValueError):
        prox_l2diff(x, b, M, gamma, Minv)


def test_non_square_matrix():
    """Test error handling for non-square matrix."""
    n1, n2 = 3, 2
    x = np.random.rand(n1, n2)
    b = np.random.rand(n1 * n2)
    M = np.random.rand(n1 * n2, n1 * n2 + 1)  # non-square matrix
    gamma = 0.5
    Minv = 1.0 / (1.0 + gamma)
    
    with pytest.raises(ValueError):
        prox_l2diff(x, b, M, gamma, Minv)