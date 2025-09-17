
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from .sampling import freq_grid

def _transfer_fresnel(nx, ny, dx, dy, wavelength, z):
    fx, fy, FX, FY = freq_grid(nx, ny, dx, dy)
    k = 2*np.pi / wavelength
    H = np.exp(1j * k * z) * np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))
    return H

def fresnel_propagate_fft(U0, dx, dy, wavelength, z):
    """
    One-FFT Fresnel propagation (Fourier method).
    U(z) = FT^{-1} { FT[U0] * H_Fresnel }.
    """
    ny, nx = U0.shape
    H = _transfer_fresnel(nx, ny, dx, dy, wavelength, z)
    U1 = ifft2( ifftshift( H ) * fftshift( fft2(U0) ) )
    return U1

def fraunhofer_propagate(U0, dx, dy, wavelength, z):
    """
    Fraunhofer (far-field) pattern ~ Fourier transform of field at aperture.
    Returns field on sensor plane with sample spacings (dx', dy') = (lambda z / (N*dx), lambda z / (M*dy)).
    """
    ny, nx = U0.shape
    U1 = fftshift( fft2( ifftshift(U0) ) )
    scale = np.exp(1j*2*np.pi*z/wavelength) / (1j * wavelength * z) * dx * dy
    return scale * U1

def angular_spectrum_propagate(U0, dx, dy, wavelength, z, bandlimit=True):
    """
    Angular Spectrum Method (ASM). Optionally band-limit evanescent components.
    """
    ny, nx = U0.shape
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)
    FX, FY = np.meshgrid(fx, fy, indexing='xy')
    k = 2*np.pi / wavelength
    kx = 2*np.pi*FX
    ky = 2*np.pi*FY
    kz_sq = (k**2 - kx**2 - ky**2)
    kz = np.sqrt(np.maximum(0.0, kz_sq)) - 1j*np.sqrt(np.maximum(0.0, -kz_sq))
    H = np.exp(1j * kz * z)
    if bandlimit:
        H = H * (kz_sq >= 0)  # remove evanescent growth
    U1 = np.fft.ifft2( np.fft.fft2(U0) * H )
    return U1
