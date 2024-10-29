import numpy as np
from numpy.typing import NDArray
from typing import Union, overload

class HuberParameterError(Exception):
    """Exception raised for invalid Huber parameter."""
    pass

def grad_huber_l2(x: Union[float, NDArray[np.float64]], 
                 rho: float) -> NDArray[np.float64]:
    """
    Compute the gradient of the Huber L2 function with MATLAB-like behavior.
    
    The gradient is defined as:
        - x/|x|        if |x| > rho
        - x/rho        if |x| ≤ rho
    
    Args:
        x: Input value or array. Can be a scalar, vector, or matrix.
           1D arrays are treated as row vectors (1xN).
        rho: Huber parameter controlling the threshold. Must be positive.
    
    Returns:
        ndarray: Gradient values with same shape as input, 
                1D inputs are returned as (1,N) arrays.
    
    Raises:
        TypeError: If rho is not a number.
        ValueError: If rho is non-positive or x contains non-finite values.
    
    Examples:
        >>> grad_huber_l2(2.0, 1.0)
        array([[1.]])
        >>> grad_huber_l2(np.array([-2, 1, 0.3, 4]), 1.0)
        array([[-1.,  1.,  0.3,  1.]])
        >>> grad_huber_l2(np.array([[1, 2], [-1, -2]]), 2.0)
        array([[ 0.5,  1. ],
               [-0.5, -1. ]])
    """

    # Input validation
    if not isinstance(rho, (int, float)):
        raise TypeError("rho must be a number")
    if rho <= 0:
        raise ValueError("rho must be positive")
    
    # Handle scalar input
    if np.isscalar(x):
        abs_x = abs(float(x))
        result = np.array([[0.0]]) if abs_x == 0 else np.array([[x/rho if abs_x <= rho else x/abs_x]])
        return result
    
    # Convert input to numpy array
    x_array = np.asarray(x, dtype=np.float64)
    
    # Determine shape handling
    is_1d = x_array.ndim == 1
    if is_1d:
        x_array = x_array.reshape(1, -1)  # Convert to 2D row vector
        
    # Check for non-finite values
    if not np.all(np.isfinite(x_array)):
        raise ValueError("Input contains non-finite values")
    
    # Initialize output array
    p = np.zeros_like(x_array)
    
    # Handle non-zero values
    nonzero_mask = x_array != 0
    p[nonzero_mask] = x_array[nonzero_mask]/np.abs(x_array[nonzero_mask])
    
    # Apply threshold condition
    threshold_mask = np.abs(x_array) <= rho
    p[threshold_mask] = x_array[threshold_mask]/rho
    
    return p