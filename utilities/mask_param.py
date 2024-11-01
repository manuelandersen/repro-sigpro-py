import numpy as np
import pywt

def mask_param(N, rm, wav):
    # Initialize zero matrix of size 2^N x 2^N
    xa = np.zeros((2**N, 2**N))
    wx = np.zeros((2**N, 2**N))
    
    # Perform rm levels of wavelet decomposition
    for j in range(1, rm + 1):
        # Perform 2D wavelet transform
        coeffs = pywt.dwt2(xa, wav)
        xa1 = coeffs[0]  # Approximation coefficients
        
        # Calculate size for current level
        size = int(2**N/(2**(j-1)))
        # Fill wx with scaled values
        wx[:size, :size] = (2**(N-j)/2) * np.ones((size, size))
        xa = xa1
    
    # Set center region to zero
    final_size = int(2**N/(2**rm))
    wx[:final_size, :final_size] = 0
    
    # Normalize by maximum value
    wx = wx/np.max(wx)
    
    return wx