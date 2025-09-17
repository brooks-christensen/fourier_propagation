
import numpy as np
from .sampling import spatial_grid

def plane_wave(nx, ny, dx, dy, wavelength, angle_x=0.0, angle_y=0.0):
    """Unit-amplitude plane wave tilted by angles (radians)."""
    k = 2*np.pi / wavelength
    x, y, X, Y = spatial_grid(nx, ny, dx, dy)
    return np.exp(1j * k * (np.sin(angle_x)*X + np.sin(angle_y)*Y))

def gaussian_beam(nx, ny, dx, dy, w0, wavelength, z=0.0):
    """TEM00 Gaussian beam (paraxial), waist w0 at z=0, evaluated at z."""
    k = 2*np.pi / wavelength
    x, y, X, Y = spatial_grid(nx, ny, dx, dy)
    zr = np.pi * w0**2 / wavelength  # Rayleigh range
    wz = w0 * np.sqrt(1 + (z/zr)**2)
    Rz = np.inf if z==0 else z*(1 + (zr/z)**2)
    psi = np.arctan2(z, zr)
    r2 = X**2 + Y**2
    amp = (w0 / wz) * np.exp(-r2 / wz**2)
    phase = np.exp(1j*(k*z + k*r2/(2*Rz) - psi))
    return amp * phase
