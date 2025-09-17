
import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift
from .sampling1d import freq_grid_1d, pad_to_length_1d

def _transfer_fresnel_1d(n, dx, wavelength, z):
    fx = freq_grid_1d(n, dx)
    k = 2*np.pi / wavelength
    H = np.exp(1j * k * z) * np.exp(-1j * np.pi * wavelength * z * (fx**2))
    return H

def fresnel_propagate_fft_1d(U0, dx, wavelength, z, pad_factor:int=2, crop:bool=True):
    """One-FFT Fresnel propagation in 1D with optional zero-padding and center crop."""
    n0 = U0.size
    if pad_factor and pad_factor > 1:
        n = int(pad_factor * n0)
        U = pad_to_length_1d(U0, n)
        _dx = dx
    else:
        U = U0
        n = n0
        _dx = dx
    H = _transfer_fresnel_1d(n, _dx, wavelength, z)
    U1_full = ifft( ifftshift(H) * fftshift( fft(U) ) )
    if crop and n != n0:
        i0 = n//2 - n0//2
        U1 = U1_full[i0:i0+n0]
    else:
        U1 = U1_full
    return U1

def fraunhofer_propagate_1d(U0, dx, wavelength, z, return_sampling:bool=False, pad_factor:int=2):
    """1D Fraunhofer propagation. Returns correct output sampling if requested."""
    n0 = U0.size
    if pad_factor and pad_factor > 1:
        n = int(pad_factor * n0)
        U = pad_to_length_1d(U0, n)
        _dx = dx
    else:
        n = n0
        U = U0
        _dx = dx
    U1 = fftshift( fft( ifftshift(U) ) )
    scale = np.exp(1j*2*np.pi*z/wavelength) / (1j * wavelength * z) * _dx
    U1 = scale * U1
    dx_out = wavelength * z / (n * _dx)
    if return_sampling:
        return U1, dx_out
    else:
        return U1

def angular_spectrum_propagate_1d(U0, dx, wavelength, z, bandlimit:bool=True, pad_factor:int=2, crop:bool=True):
    """1D Angular Spectrum Method with optional band-limiting and padding."""
    n0 = U0.size
    if pad_factor and pad_factor > 1:
        n = int(pad_factor * n0)
        U = pad_to_length_1d(U0, n)
        _dx = dx
    else:
        n = n0
        U = U0
        _dx = dx
    fx = np.fft.fftfreq(n, d=_dx)
    k = 2*np.pi / wavelength
    kx = 2*np.pi * fx
    kz_sq = k**2 - kx**2
    kz = np.sqrt(np.maximum(0.0, kz_sq)) - 1j*np.sqrt(np.maximum(0.0, -kz_sq))
    H = np.exp(1j * kz * z)
    if bandlimit:
        H = H * (kz_sq >= 0)
    U1_full = np.fft.ifft( np.fft.fft(U) * H )
    if crop and n != n0:
        i0 = n//2 - n0//2
        U1 = U1_full[i0:i0+n0]
    else:
        U1 = U1_full
    return U1
