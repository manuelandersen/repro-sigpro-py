from typing import Tuple
import numpy as np
from numpy.typing import NDArray
import pywt  # PyWavelets library for wavelet transforms

# def amr2D_vect(
#     x: NDArray[np.float64], 
#     rm: int, 
#     wav: str
# ) -> NDArray[np.float64]:
#     """
#     Perform 2D wavelet transform on an input matrix for multiple levels.
    
#     Args:
#         x: Input matrix of shape (Nx, Mx)
#         rm: Number of decomposition levels
#         wav: Wavelet type (e.g., 'db1', 'haar', 'sym4')
        
#     Returns:
#         NDArray[np.float64]: Matrix containing wavelet coefficients
        
#     Raises:
#         ValueError: If input dimensions are not compatible with the requested 
#                    number of decomposition levels
        
#     Notes:
#         This function performs a multi-level 2D discrete wavelet transform.
#         For each level j:
#         - The approximation coefficients (xa) and detail coefficients (xh,xv,xd)
#           are computed
#         - The coefficients are arranged in the output matrix according to the
#           standard wavelet decomposition layout
#     """
#     # Get input dimensions
#     Nx, Mx = x.shape
    
#     # Input validation
#     min_size = 2 ** rm
#     if Nx < min_size or Mx < min_size:
#         raise ValueError(
#             f"Input dimensions ({Nx}, {Mx}) too small for {rm} levels of decomposition. "
#             f"Minimum size required: {min_size}x{min_size}"
#         )
    
#     # Initialize working and output arrays
#     xa = x.copy()
#     wx = np.zeros((Nx, Mx), dtype=np.float64)
    
#     # Perform wavelet transform for each level
#     for j in range(1, rm + 1):
#         # Compute 2D DWT - returns tuple of (coeffs_ll, (coeffs_lh, coeffs_hl, coeffs_hh))
#         coeffs = pywt.dwt2(xa, wav)
#         xa, (xh, xv, xd) = coeffs
        
#         # Calculate size of current level's coefficients
#         current_Nx = Nx // (2 ** (j - 1))
#         current_Mx = Mx // (2 ** (j - 1))
#         half_Nx = current_Nx // 2
#         half_Mx = current_Mx // 2
        
#         # Arrange coefficients in output matrix
#         wx[:half_Nx, :half_Mx] = xa
#         wx[:half_Nx, half_Mx:current_Mx] = xh
#         wx[half_Nx:current_Nx, :half_Mx] = xv
#         wx[half_Nx:current_Nx, half_Mx:current_Mx] = xd
    
#     return wx

def amr2D_vect(
    x: NDArray[np.float64], 
    rm: int, 
    wav: str
) -> NDArray[np.float64]:
    """
    Perform 2D wavelet transform on an input matrix for multiple levels.
    """
    # Get input dimensions
    Nx, Mx = x.shape
    
    # Input validation
    min_size = 2 ** rm
    if Nx < min_size or Mx < min_size:
        raise ValueError(
            f"Input dimensions ({Nx}, {Mx}) too small for {rm} levels of decomposition. "
            f"Minimum size required: {min_size}x{min_size}"
        )
    
    # Initialize working and output arrays
    xa = x.copy()
    wx = np.zeros((Nx, Mx), dtype=np.float64)
    
    # Perform wavelet transform for each level
    for j in range(1, rm + 1):
        # Compute 2D DWT with mode='periodization' to ensure power-of-2 dimensions
        coeffs = pywt.dwt2(xa, wav, mode='periodization')
        xa, (xh, xv, xd) = coeffs
        
        # Calculate size of current level's coefficients
        current_Nx = Nx // (2 ** (j - 1))
        current_Mx = Mx // (2 ** (j - 1))
        half_Nx = current_Nx // 2
        half_Mx = current_Mx // 2
        
        # Get actual sizes of wavelet coefficients
        xa_shape = xa.shape
        xh_shape = xh.shape
        xv_shape = xv.shape
        xd_shape = xd.shape
        
        # Arrange coefficients in output matrix, using actual sizes
        wx[:xa_shape[0], :xa_shape[1]] = xa
        wx[:xh_shape[0], half_Mx:half_Mx+xh_shape[1]] = xh
        wx[half_Nx:half_Nx+xv_shape[0], :xv_shape[1]] = xv
        wx[half_Nx:half_Nx+xd_shape[0], half_Mx:half_Mx+xd_shape[1]] = xd
    
    return wx