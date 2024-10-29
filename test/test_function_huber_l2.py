import numpy as np
import pytest
from utilities.function_huber_l2 import function_huber_l2

@pytest.mark.parametrize("x, rho, expected", [
    (2.0, 1.0, 1.5),
    (np.array([-2, 1, 0.3, 4]), 1.0, 5.545), 
    (np.array([[1, 2, 3], [-1, -2, -3]]), 2.0, 6.5), 
    (np.array([-1.5, -0.5, 0, 0.5, 1.5]), 1.0, 2.25),  
])
def test_function_huber_l2(x, rho, expected):
    result = function_huber_l2(x, rho)
    assert pytest.approx(result, rel=1e-3) == expected

@pytest.mark.parametrize("x, rho", [
    (np.array([-1.5, -0.5, 0, 0.5, 1.5]), -1.0),
])
def test_function_huber_l2_invalid_rho(x, rho):
    with pytest.raises(ValueError, match="rho must be positive"):
        function_huber_l2(x, rho)

@pytest.mark.parametrize("x, rho", [
    ("string", 1.0),
])
def test_function_huber_l2_invalid_x(x, rho):
    with pytest.raises(TypeError):
        function_huber_l2(x, rho)
