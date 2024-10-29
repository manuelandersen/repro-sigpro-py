import numpy as np
from numpy.typing import NDArray
from typing import Union

def prox_huber_l2_matrix(
    x: NDArray[np.float64], 
    rho: float, 
    gamma: Union[float, NDArray[np.float64]]
) -> NDArray[np.float64]:
    """
    Compute the proximal operator of the Huber function.
    
    Args:
        x: Input array of shape (n, m) or (n,)
        rho: Huber parameter
        gamma: Step size (scalar or array matching x's shape)
    
    Returns:
        NDArray[np.float64]: Proximal operator result with same shape as x
    """
    original_shape = x.shape
    x_flat = x.ravel()
    
    if isinstance(gamma, np.ndarray):
        gamma_flat = gamma.ravel()
        p = x_flat - gamma_flat * x_flat / np.abs(x_flat + np.finfo(float).eps)
        mask = np.abs(x_flat) <= (gamma_flat + rho)
        p[mask] = rho * x_flat[mask] / (gamma_flat[mask] + rho)
    else:
        p = x_flat - gamma * x_flat / np.abs(x_flat + np.finfo(float).eps)
        mask = np.abs(x_flat) <= (gamma + rho)
        p[mask] = rho * x_flat[mask] / (gamma + rho)
    
    return p.reshape(original_shape)
