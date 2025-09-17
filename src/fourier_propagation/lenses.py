
import numpy as np
from .sampling import spatial_grid

def thin_lens_phase(nx, ny, dx, dy, wavelength, f):
    """Return thin-lens complex phase (multiplicative), ignoring aperture stop."""
    k = 2*np.pi / wavelength
    x, y, X, Y = spatial_grid(nx, ny, dx, dy)
    r2 = X**2 + Y**2
    return np.exp(-1j * k * r2 / (2*f))
