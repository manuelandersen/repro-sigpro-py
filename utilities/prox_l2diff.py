# import numpy as np
# from numpy.typing import NDArray
# from typing import Union, cast


# def prox_l2diff(
#     x: NDArray[np.float64],
#     b: NDArray[np.float64],
#     M: NDArray[np.float64],
#     gamma: float,
#     Minv: float
# ) -> NDArray[np.float64]:
#     """
#     Compute the proximal operator for L2 difference term with matrix M.
    
#     This computes: p = reshape(Minv*(gamma*M'*b + x(:)), n1, n2)
    
#     Args:
#         x: Input matrix of shape (n1, n2)
#         b: Target vector of shape (n1*n2,)
#         M: Linear operator matrix of shape (n1*n2, n1*n2)
#         gamma: Step size parameter (positive float)
#         Minv: Inverse scaling parameter (positive float)
    
#     Returns:
#         NDArray[np.float64]: Proximal operator result with same shape as x
        
#     Raises:
#         ValueError: If input dimensions are incompatible or parameters are invalid
#     """
#     # Input validation
#     if gamma <= 0 or Minv <= 0:
#         raise ValueError("gamma and Minv must be positive")
    
#     n1, n2 = x.shape
#     x_flat = x.flatten()
    
#     if M.shape[0] != M.shape[1]:
#         raise ValueError("M must be a square matrix")
#     if M.shape[0] != len(b):
#         raise ValueError(f"M dimensions ({M.shape[0]}) must match b length ({len(b)})")
#     if M.shape[1] != len(x_flat):
#         raise ValueError(f"M dimensions ({M.shape[1]}) must match flattened x length ({len(x_flat)})")
    
#     # Compute proximal operator
#     result = Minv * (gamma * M.T @ b + x_flat)
#     return result.reshape(n1, n2)

import numpy as np
from numpy.typing import NDArray
from typing import Union, cast

def prox_l2diff(
    x: NDArray[np.float64],
    b: NDArray[np.float64],
    M: Union[NDArray[np.float64], float],
    gamma: float,
    Minv: float
) -> NDArray[np.float64]:
    """
    Compute the proximal operator for L2 difference term with matrix M.
    
    This computes: p = Minv*(gamma*M'*b + x)
    
    Args:
        x: Input array of shape (n,) or (n1, n2)
        b: Target vector of shape matching x if M is scalar, or matching M.shape[0]
        M: Linear operator matrix of shape (n, n) or (n1*n2, n1*n2), or scalar
        gamma: Step size parameter (positive float)
        Minv: Inverse scaling parameter (positive float)
    
    Returns:
        NDArray[np.float64]: Proximal operator result with same shape as x
        
    Raises:
        ValueError: If input dimensions are incompatible or parameters are invalid
    """
    # Input validation
    if gamma <= 0 or Minv <= 0:
        raise ValueError("gamma and Minv must be positive")
    
    # Store original shape for later reshaping
    original_shape = x.shape
    x_flat = x.ravel()
    
    # Handle scalar M (identity scaling)
    if np.isscalar(M):
        if len(b) != len(x_flat):
            raise ValueError(f"When M is scalar, b length ({len(b)}) must match x length ({len(x_flat)})")
        result = Minv * (gamma * M * b + x_flat)
        return result.reshape(original_shape)
    
    # Handle matrix M
    if M.shape[0] != M.shape[1]:
        raise ValueError("M must be a square matrix")
    if M.shape[0] != len(b):
        raise ValueError(f"M dimensions ({M.shape[0]}) must match b length ({len(b)})")
    if M.shape[1] != len(x_flat):
        raise ValueError(f"M dimensions ({M.shape[1]}) must match flattened x length ({len(x_flat)})")
    
    # Compute proximal operator
    result = Minv * (gamma * M.T @ b + x_flat)
    return result.reshape(original_shape)