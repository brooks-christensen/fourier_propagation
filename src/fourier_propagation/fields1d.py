
import numpy as np
from .sampling1d import spatial_grid_1d

def plane_wave_1d(n, dx, wavelength, angle=0.0):
    """Unit-amplitude plane wave tilted by angle (radians) in x-z plane."""
    k = 2*np.pi / wavelength
    x = spatial_grid_1d(n, dx)
    return np.exp(1j * k * np.sin(angle) * x)

def gaussian_1d(n, dx, w0, wavelength, z=0.0):
    """1D Gaussian beam cross-section with waist w0 at z=0, paraxial phase evolution along z."""
    k = 2*np.pi / wavelength
    x = spatial_grid_1d(n, dx)
    zr = np.pi * w0**2 / wavelength
    wz = w0 * np.sqrt(1 + (z/zr)**2)
    Rz = np.inf if z==0 else z*(1 + (zr/z)**2)
    psi = np.arctan2(z, zr)
    amp = (w0 / wz) * np.exp(-(x**2) / (wz**2))
    phase = np.exp(1j*(k*z + k*(x**2)/(2*Rz) - psi))
    return amp * phase

def thin_lens_phase_1d(n, dx, wavelength, f):
    """1D thin-lens phase along x (ignores aperture stop)."""
    k = 2*np.pi / wavelength
    x = spatial_grid_1d(n, dx)
    return np.exp(-1j * k * (x**2) / (2*f))
