import numpy as np
from numpy.typing import NDArray
from typing import Union

def prox_huber_l2(
    x: NDArray[np.float64], 
    rho: float, 
    gamma: Union[float, NDArray[np.float64]]
) -> NDArray[np.float64]:
    """
    Compute the proximal operator of the Huber L2 function.
    
    Args:
        x: Input array of shape (n, m) or (n,)
        rho: Huber parameter (must be positive)
        gamma: Step size parameter - can be scalar or array matching x's shape
    
    Returns:
        NDArray[np.float64]: Proximal operator result with same shape as x
        
    Raises:
        ValueError: If rho is not positive or input dimensions are incompatible
    """
    # Input validation
    if rho <= 0:
        raise ValueError("rho must be positive")
    
    # Store original shape for later reshaping
    original_shape = x.shape
    x_flat = x.ravel()
    
    # Handle gamma as array or scalar
    if isinstance(gamma, np.ndarray):
        gamma_flat = gamma.ravel()
        if gamma_flat.shape != x_flat.shape:
            raise ValueError("When gamma is an array, it must match the shape of x")
    else:
        gamma_flat = gamma
        
    # Initialize output array
    p = x_flat.copy()
    
    # Handle non-zero values to avoid division by zero
    nonzero_mask = x_flat != 0
    if isinstance(gamma_flat, np.ndarray):
        p[nonzero_mask] = x_flat[nonzero_mask] - gamma_flat[nonzero_mask] * x_flat[nonzero_mask] / np.abs(x_flat[nonzero_mask])
    else:
        p[nonzero_mask] = x_flat[nonzero_mask] - gamma_flat * x_flat[nonzero_mask] / np.abs(x_flat[nonzero_mask])
    
    # Apply threshold condition
    if isinstance(gamma_flat, np.ndarray):
        threshold_mask = np.abs(x_flat) <= (gamma_flat + rho)
        p[threshold_mask] = rho * x_flat[threshold_mask] / (gamma_flat[threshold_mask] + rho)
    else:
        threshold_mask = np.abs(x_flat) <= (gamma_flat + rho)
        p[threshold_mask] = rho * x_flat[threshold_mask] / (gamma_flat + rho)
    
    # Restore original shape
    return p.reshape(original_shape)
