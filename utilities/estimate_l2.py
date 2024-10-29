# import numpy as np
# from numpy.typing import NDArray
# from typing import Tuple


# def estimate_l2(
#     x: NDArray[np.float64], 
#     b: NDArray[np.float64], 
#     M: NDArray[np.float64]
# ) -> Tuple[float, NDArray[np.float64]]:
#     """
#     Calculate L2 criterion and its gradient for the expression 1/2 * ||Mx - b||^2.
    
#     Args:
#         x: Input matrix of shape (n1, n2)
#         b: Target vector of shape (n1*n2,)
#         M: Linear operator matrix of shape (n1*n2, n1*n2)
    
#     Returns:
#         Tuple containing:
#             - criterion: Scalar value of 1/2 * ||Mx - b||^2
#             - gradient: Matrix of shape (n1, n2) containing the gradient
    
#     Raises:
#         ValueError: If input dimensions are incompatible
#     """
#     n1, n2 = x.shape
#     x_flat = x.flatten()
    
#     # Validate input dimensions
#     if M.shape[0] != len(b):
#         raise ValueError(f"M rows ({M.shape[0]}) must match b length ({len(b)})")
#     if M.shape[1] != len(x_flat):
#         raise ValueError(f"M columns ({M.shape[1]}) must match flattened x length ({len(x_flat)})")
    
#     # Calculate criterion
#     diff = M @ x_flat - b
#     criterion = 0.5 * np.sum(diff**2)
    
#     # Calculate gradient
#     gradient = (M.T @ diff).reshape(n1, n2)
    
#     return criterion, gradient

import numpy as np
from numpy.typing import NDArray
from typing import Tuple

def estimate_l2(
    x: NDArray[np.float64], 
    b: NDArray[np.float64], 
    M: NDArray[np.float64] | float
) -> Tuple[float, NDArray[np.float64]]:
    """
    Calculate L2 criterion and its gradient for the expression 1/2 * ||Mx - b||^2.
    
    Args:
        x: Input array of shape (n,) or (n1, n2)
        b: Target array of shape (n,) or (n1*n2,)
        M: Linear operator matrix of shape (n, n) or (n1*n2, n1*n2), or scalar
    
    Returns:
        Tuple containing:
            - criterion: Scalar value of 1/2 * ||Mx - b||^2
            - gradient: Array with same shape as input x containing the gradient
    
    Raises:
        ValueError: If input dimensions are incompatible
    """
    # Handle scalar M (identity scaling)
    if np.isscalar(M):
        M = float(M)
        x_flat = x.ravel()
        
        # Calculate criterion
        diff = M * x_flat - b
        criterion = 0.5 * np.sum(diff**2)
        
        # Calculate gradient
        gradient = M * diff
        
        return criterion, gradient.reshape(x.shape)
    
    # Handle matrix M
    x_flat = x.ravel()
    
    # Validate input dimensions
    if M.shape[0] != len(b):
        raise ValueError(f"M rows ({M.shape[0]}) must match b length ({len(b)})")
    if M.shape[1] != len(x_flat):
        raise ValueError(f"M columns ({M.shape[1]}) must match flattened x length ({len(x_flat)})")
    
    # Calculate criterion
    diff = M @ x_flat - b
    criterion = 0.5 * np.sum(diff**2)
    
    # Calculate gradient
    gradient = (M.T @ diff).reshape(x.shape)
    
    return criterion, gradient