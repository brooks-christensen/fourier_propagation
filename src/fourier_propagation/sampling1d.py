
import numpy as np

def spatial_grid_1d(n:int, dx:float):
    """Return 1D spatial coordinate grid x of length n with spacing dx, centered at 0."""
    x = (np.arange(n) - n//2) * dx
    return x

def freq_grid_1d(n:int, dx:float):
    """Return 1D frequency grid fx (cycles/m) aligned with fftshift convention."""
    fx = np.fft.fftfreq(n, d=dx)
    return np.fft.fftshift(fx)

def pad_to_length_1d(field:np.ndarray, n_out:int):
    """Zero-pad (or center-crop) a 1D field to target length n_out."""
    n = field.size
    out = np.zeros(n_out, dtype=field.dtype)
    m = min(n, n_out)
    i0_out = n_out//2 - m//2
    i0_in  = n//2     - m//2
    out[i0_out:i0_out+m] = field[i0_in:i0_in+m]
    return out
