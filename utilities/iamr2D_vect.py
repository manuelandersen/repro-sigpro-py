import numpy as np
import pywt
from typing import Tuple, Union
from numpy.typing import NDArray


def iamr2D_vect(
    coeffs: NDArray[np.float64],
    decomp_level: int,
    wavelet: str
) -> NDArray[np.float64]:
    """
    Perform 2D inverse discrete wavelet transform reconstruction.
    
    This function reconstructs the original image from wavelet coefficients
    using the inverse discrete wavelet transform (IDWT).
    
    Args:
        coeffs: Input coefficient matrix of shape (Nx, Mx)
        decomp_level: Number of decomposition levels (rm in MATLAB code)
        wavelet: Wavelet type to use (e.g., 'db1', 'haar', 'sym4')
        
    Returns:
        NDArray[np.float64]: Reconstructed image/signal
        
    Raises:
        ValueError: If input dimensions are not compatible with decomposition level
        ValueError: If wavelet type is not supported
        ValueError: If coeffs is not a 2D array
    """
    # Input validation
    if not isinstance(coeffs, np.ndarray) or coeffs.ndim != 2:
        raise ValueError("Input coefficients must be a 2D array")
        
    Nx, Mx = coeffs.shape
    
    # Validate decomposition levels
    max_level = min(int(np.log2(Nx)), int(np.log2(Mx)))
    if decomp_level > max_level:
        raise ValueError(
            f"Decomposition level {decomp_level} is too high for input shape {coeffs.shape}. "
            f"Maximum possible level is {max_level}"
        )
        
    # Validate wavelet type
    if wavelet not in pywt.wavelist():
        raise ValueError(f"Wavelet type '{wavelet}' is not supported")
    
    # Initialize approximation coefficients with the lowest resolution part
    xa_rows = Nx // (2 ** decomp_level)
    xa_cols = Mx // (2 ** decomp_level)
    xa = coeffs[:xa_rows, :xa_cols].copy()
    
    # Reconstruct image from coarse to fine
    for j in range(decomp_level, 0, -1):
        # Calculate dimensions for current level
        rows = Nx // (2 ** j)
        cols = Mx // (2 ** j)
        
        # Extract detail coefficients
        xh = coeffs[:rows, cols:2*cols]         # Horizontal details
        xv = coeffs[rows:2*rows, :cols]         # Vertical details
        xd = coeffs[rows:2*rows, cols:2*cols]   # Diagonal details
        
        # Perform single level 2D inverse DWT
        xa = pywt.idwt2(
            (xa, (xh, xv, xd)),  # Coefficients in PyWavelets format
            wavelet,             # Wavelet type
            mode='periodization' # Use periodic extension at boundaries
        )
        
        # Handle potential None return from idwt2
        if xa is None:
            raise RuntimeError("Failed to perform inverse wavelet transform")
    
    return xa