
# 1D x–z propagation: uniform field through a rectangular aperture
import numpy as np
import matplotlib.pyplot as plt
from fourier_propagation import rect_slit_1d, fresnel_propagate_fft_1d

# Parameters (from prompt)
Nx = 1024
lam = 500e-9
dx = lam / 2
dz = 2*dx
Lx = Nx * dx
aperture_width = 6 * lam
Nz = 1024  # spans z_max ~= 0.512 mm
# Nz = 256

# Input field: uniform * aperture
U_in = rect_slit_1d(Nx, dx, aperture_width).astype(complex)

# Build x–z intensity map
xz = np.zeros((Nz, Nx), dtype=np.float32)
for k in range(Nz):
    z = k * dz
    Uz = fresnel_propagate_fft_1d(U_in, dx, lam, z, pad_factor=2)
    xz[k, :] = np.abs(Uz)**2

# Plot x–z plane
extent = (-Lx/2, Lx/2, 0.0, Nz*dz)
plt.figure()
plt.imshow(xz, extent=extent, origin='lower', aspect='auto')
plt.xlabel('x [m]'); plt.ylabel('z [m]')
plt.title('Propagation: uniform field through rectangular aperture (1D)')
plt.colorbar(label='Intensity (a.u.)')
plt.tight_layout()
plt.show()
