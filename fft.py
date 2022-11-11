import numpy as np


def fft(x):
    r"""function to compute the fast fourier transformation of a vector x.
    Args:
        x: shape 1d-array. 
    out:
        f: fourier transformation of the vector x.
    """
    n = len(x)//2
    f = np.zeros(len(x),dtype=complex)
    k = np.arange(n)
    
    for m in range(n):
        # calculate the for even k's the contribution 
        even = np.sum(x[::2]*np.exp(-2 * np.pi * 1j * m * k / n))
        
        # calculate for uneven k's the contriution
        factor =  np.exp(-np.pi * 1j * m / n)
        uneven =  factor * np.sum(x[1::2] * np.exp(-2 * np.pi * 1j * k * m / n))
        
        # calculate the resulting array
        f[m] = even + uneven
        f[m+n] = even - uneven
        
    return f
