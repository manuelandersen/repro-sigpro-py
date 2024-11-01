from typing import Callable, Protocol, Tuple, NamedTuple
import numpy as np
from numpy.typing import NDArray

from function_huber_l2 import function_huber_l2
from grad_huberl2 import grad_huber_l2

class Operator(NamedTuple):
    """Class representing direct and adjoint operators."""
    direct: Callable[[NDArray], NDArray]
    adjoint: Callable[[NDArray], NDArray]

def create_difference_matrix(n: int) -> NDArray:
    """
    Create the first-order discrete difference operator matrix.
    
    Args:
        n: Size of the signal
        
    Returns:
        NDArray: Difference matrix D
    """
    # Create difference matrix D where (Dx)_n = (x_n - x_{n-1})/2
    D = np.zeros((n, n))
    np.fill_diagonal(D, 0.5)  # Main diagonal
    np.fill_diagonal(D[:, 1:], -0.5)  # Superdiagonal
    return D

def create_operators(n: int) -> Tuple[Operator, Operator, Operator]:
    """
    Create the three operators (op, op1, op2) using just numpy.
    
    Args:
        n: Size of the signal
        
    Returns:
        Tuple of three Operators (op, op1, op2)
    """
    # Create difference matrix D
    D = create_difference_matrix(n)
    
    def adjoint_op(x):
        # Ensure x is the right shape (convert from 2D to 1D if necessary)
        if x.ndim == 2 and x.shape[0] == 1:
            x = x.flatten()
        return D.T @ x
    
    op = Operator(
        direct=lambda x: D @ x,
        adjoint=adjoint_op
    )

    D1 = D[::2]  # Take odd-indexed rows
    def adjoint_op1(x):
        if x.ndim == 2 and x.shape[0] == 1:
            x = x.flatten()
        return D1.T @ x
    
    op1 = Operator(
        direct=lambda x: D1 @ x,
        adjoint=adjoint_op1
    )
    
    D2 = D[1::2]  # Take even-indexed rows
    def adjoint_op2(x):
        if x.ndim == 2 and x.shape[0] == 1:
            x = x.flatten()
        return D2.T @ x
    
    op2 = Operator(
        direct=lambda x: D2 @ x,
        adjoint=adjoint_op2
    )
    
    return op, op1, op2

def estimate_huber_op(
    x: NDArray, 
    b: NDArray, 
    rho: float, 
    op: Operator
) -> Tuple[float, NDArray]:
    """
    Compute the Huber estimate and its gradient.
    
    Args:
        x: Input signal (1D array)
        b: Offset signal (1D array)
        rho: Huber parameter
        op: Operator containing direct and adjoint callables
        
    Returns:
        Tuple containing:
        - criterion value (float)
        - gradient array (NDArray)
    """
    # Ensure inputs are 1D
    x = np.asarray(x).ravel()
    b = np.asarray(b).ravel()
    
    # Compute direct operation and difference
    Dx = op.direct(x)

    # Important: b should match the size of Dx, not x
    if b.shape != Dx.shape:
        raise ValueError(f"Shape mismatch: operator output has shape {Dx.shape}, but target b has shape {b.shape}. "
                       f"For op1 and op2, b should be half the size of the input signal.")
    

    Dx_minus_b = Dx - b
    
    # Compute criterion
    crit = function_huber_l2(Dx_minus_b, rho)
    
    # Compute gradient of Huber function and apply adjoint
    huber_grad = grad_huber_l2(Dx_minus_b, rho)  # This returns a (1,N) array
    # The adjoint operator will handle the conversion from (1,N) to 1D
    grad = op.adjoint(huber_grad)
    
    return crit, grad
