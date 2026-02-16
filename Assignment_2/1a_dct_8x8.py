import numpy as np
from scipy.fft import dct

def dct_8x8(block):
    """
    Perform 2D DCT on an 8x8 block
    """
    block = np.asarray(block, dtype=float)

    if block.shape != (8, 8):
        raise ValueError("Input must be an 8x8 array")

    # Apply DCT on rows, then columns
    return dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')

