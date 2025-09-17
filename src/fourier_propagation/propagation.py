
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from .sampling import freq_grid, pad_to_shape

def _transfer_fresnel(nx, ny, dx, dy, wavelength, z):
    fx, fy, FX, FY = freq_grid(nx, ny, dx, dy)
    k = 2*np.pi / wavelength
    H = np.exp(1j * k * z) * np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))
    return H

def fresnel_propagate_fft(U0, dx, dy, wavelength, z, pad_factor: int = 2, crop: bool = True):
    """
    One-FFT Fresnel propagation with optional zero-padding to suppress circular wrap-around.
    Returns propagated field on the same sampling as input (after cropping).
    """
    ny0, nx0 = U0.shape
    if pad_factor and pad_factor > 1:
        Ny = int(pad_factor * ny0)
        Nx = int(pad_factor * nx0)
        U = pad_to_shape(U0, (Ny, Nx))
        _dx, _dy = dx, dy
    else:
        U = U0
        _dx, _dy = dx, dy

    ny, nx = U.shape
    H = _transfer_fresnel(nx, ny, _dx, _dy, wavelength, z)
    U1_full = ifft2( ifftshift(H) * fftshift( fft2(U) ) )

    if crop and (U.shape != U0.shape):
        y0 = ny//2 - ny0//2; x0 = nx//2 - nx0//2
        U1 = U1_full[y0:y0+ny0, x0:x0+nx0]
    else:
        U1 = U1_full
    return U1

def fraunhofer_propagate(U0, dx, dy, wavelength, z, return_sampling: bool = False, pad_factor: int = 2):
    """
    Fraunhofer (far-field) pattern ~ Fourier transform of field at aperture.
    Output plane sample spacings are:
        dx_out = lambda * z / (N * dx)
        dy_out = lambda * z / (M * dy)
    Optionally pads the input to reduce spectral leakage.
    """
    ny0, nx0 = U0.shape
    if pad_factor and pad_factor > 1:
        Ny = int(pad_factor * ny0)
        Nx = int(pad_factor * nx0)
        U = pad_to_shape(U0, (Ny, Nx))
        px, py = dx, dy
        ny, nx = U.shape
    else:
        U = U0
        px, py = dx, dy
        ny, nx = ny0, nx0

    U1 = fftshift( fft2( ifftshift(U) ) )
    # Physical scaling
    scale = np.exp(1j*2*np.pi*z/wavelength) / (1j * wavelength * z) * px * py
    U1 = scale * U1

    dx_out = wavelength * z / (nx * px)
    dy_out = wavelength * z / (ny * py)

    if return_sampling:
        return U1, dx_out, dy_out
    else:
        return U1

def angular_spectrum_propagate(U0, dx, dy, wavelength, z, bandlimit=True, pad_factor: int = 2, crop: bool = True):
    """
    Angular Spectrum Method (ASM). Optionally band-limit evanescent components.
    Includes padding to suppress wrap-around.
    """
    ny0, nx0 = U0.shape
    if pad_factor and pad_factor > 1:
        Ny = int(pad_factor * ny0)
        Nx = int(pad_factor * nx0)
        U = pad_to_shape(U0, (Ny, Nx))
        _dx, _dy = dx, dy
    else:
        U = U0
        _dx, _dy = dx, dy

    ny, nx = U.shape
    fx = np.fft.fftfreq(nx, d=_dx)
    fy = np.fft.fftfreq(ny, d=_dy)
    FX, FY = np.meshgrid(fx, fy, indexing='xy')
    k = 2*np.pi / wavelength
    kx = 2*np.pi*FX
    ky = 2*np.pi*FY
    kz_sq = (k**2 - kx**2 - ky**2)
    # Proper branch for forward propagation: real for propagating, imag for evanescent
    kz = np.sqrt(np.maximum(0.0, kz_sq)) - 1j*np.sqrt(np.maximum(0.0, -kz_sq))
    H = np.exp(1j * kz * z)
    if bandlimit:
        H = H * (kz_sq >= 0)

    U1_full = np.fft.ifft2( np.fft.fft2(U) * H )

    if crop and (U.shape != U0.shape):
        y0 = ny//2 - ny0//2; x0 = nx//2 - nx0//2
        U1 = U1_full[y0:y0+ny0, x0:x0+nx0]
    else:
        U1 = U1_full
    return U1
