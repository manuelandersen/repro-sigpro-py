import numpy as np
from numpy.typing import NDArray
from typing import Union


def function_huber_l2(x: Union[float, NDArray[np.float64]], 
                     rho: float) -> float:
    """
    Compute the Huber L2 function value for given input and parameter.
    
    The Huber L2 function is defined as:
        - |x| - rho/2        if |x| > rho
        - x^2/(2*rho)        if |x| ≤ rho
    
    Args:
        x: Input value or array. Can be a scalar, vector, or matrix.
        rho: Huber parameter controlling the threshold between 
             quadratic and linear regions. Must be positive.
    
    Returns:
        float: Sum of the Huber L2 function values.
    
    Raises:
        ValueError: If rho is not positive or if x contains non-finite values.
        TypeError: If rho is not a number.
    """
    # Input validation
    if not isinstance(rho, (int, float)):
        raise TypeError("rho must be a number")
    if rho <= 0:
        raise ValueError("rho must be positive")
    
    # Ensure `x` is numeric
    if np.isscalar(x) and not isinstance(x, (int, float)):
        raise TypeError("Input x must be numeric")
    if not np.isscalar(x) and not np.issubdtype(np.asarray(x).dtype, np.number):
        raise TypeError("Input x must be numeric")
    
    # # Ensure `x` is numeric
    # if not np.isscalar(x) and not np.issubdtype(np.asarray(x).dtype, np.number):
    #     raise TypeError("Input x must be numeric")

    # Handle scalar input
    if np.isscalar(x):
        abs_x = abs(float(x))
        if abs_x <= rho:
            return abs_x**2 / (2*rho)
        return abs_x - rho/2
    
    # Convert input to numpy array if needed
    x_array = np.asarray(x, dtype=np.float64)
    
    # Check for non-finite values
    if not np.all(np.isfinite(x_array)):
        raise ValueError("Input contains non-finite values")
    
    # Compute absolute values
    abs_x = np.abs(x_array)
    
    # Initialize output array with the linear part
    p = abs_x - rho/2
    
    # Find indices where |x| ≤ rho and apply quadratic form
    quad_indices = abs_x <= rho
    p[quad_indices] = abs_x[quad_indices]**2 / (2*rho)
    
    # Return sum of all elements
    return float(np.sum(p))