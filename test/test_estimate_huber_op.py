import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from utilities.estimate_huber_op import (
    create_difference_matrix,
    create_operators,
    estimate_huber_op
)

def test_create_difference_matrix():
    # Test with a small size for easy verification
    n = 3
    D = create_difference_matrix(n)
    expected = np.array([
        [0.5, -0.5, 0.0],
        [0.0, 0.5, -0.5],
        [0.0, 0.0, 0.5]
    ])
    assert_array_almost_equal(D, expected)


def test_estimate_huber_op():
    n = 4
    x = np.array([1, 2, 3, 4])
    b = np.array([0.5, 1.5, 2.5, 3.5])
    rho = 1.0
    
    # Create operator
    op, _, _ = create_operators(n)
    
    # Compute estimate_huber_op
    crit, grad = estimate_huber_op(x, b, rho, op)
    
    # Check criterion output
    assert isinstance(crit, float), "Criterion should be a float"
    
    # Check gradient output shape
    assert grad.shape == x.shape, "Gradient shape should match input x"
    
    # Check values (use approximate since it's mathematical)
    assert crit >= 0, "Criterion should be non-negative"
    assert np.all(np.isfinite(grad)), "Gradient should have finite values"

def test_estimate_huber_op_with_difference_operator():
    n = 5
    op, op1, op2 = create_operators(n)

    # Test with the primary operator (difference operator)
    x = np.array([1, 2, 3, 4, 5])  # Input signal
    b = np.array([1, 1, 1, 1, 1])   # Offset signal
    rho = 1.0

    crit, grad = estimate_huber_op(x, b, rho, op)

    # Add assertions for crit and grad based on expected outcomes
    assert crit is not None  # Check criterion is computed
    assert grad.shape == x.shape  # Gradient should match input shape

#TODO: add tests for the other operators

def test_estimate_huber_op_with_zero_offset():
    n = 5
    op, op1, op2 = create_operators(n)

    x = np.array([1, 2, 3, 4, 5])
    b = np.zeros(5)  # Zero offset signal
    rho = 1.0

    crit, grad = estimate_huber_op(x, b, rho, op)

    assert crit is not None
    assert grad.shape == x.shape  # Gradient should match input shape
