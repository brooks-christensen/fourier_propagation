
import numpy as np
from fourier_propagation import fresnel_propagate_fft

def test_energy_conservation_fresnel():
    nx = ny = 256
    dx = dy = 10e-6
    lam = 633e-9
    U0 = np.ones((ny, nx), dtype=complex)
    U1 = fresnel_propagate_fft(U0, dx, dy, lam, 0.02)
    e0 = np.sum(np.abs(U0)**2)
    e1 = np.sum(np.abs(U1)**2)
    assert np.isclose(e0, e1, rtol=1e-2)
