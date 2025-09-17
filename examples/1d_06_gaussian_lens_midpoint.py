
# 1D x–z propagation: Gaussian beam through a thin (parabolic) glass lens at z mid-point
import numpy as np
import matplotlib.pyplot as plt
from fourier_propagation import gaussian_1d, fresnel_propagate_fft_1d, thin_lens_phase_1d

# Parameters
Nx = 1024
lam = 500e-9
n_lens = 1.5
w0 = 1.5 * lam

dx = lam/2
dz = 2*dx
Lx = Nx * dx
Nz = 1024
z_max = Nz * dz
z_lens = z_max / 2

# f = 100e-6
f = z_max / 4   # choose f so focus lands at z_max

# Input Gaussian at z=0
U0 = gaussian_1d(Nx, dx, w0, lam, z=0.0)

# Thin-lens phase for a parabolic lens: φ(x) = -k x^2 / (2f)
lens_phase = thin_lens_phase_1d(Nx, dx, lam, f)

# Precompute fields at the lens plane
U_before = fresnel_propagate_fft_1d(U0, dx, lam, z_lens, pad_factor=2)
U_after0 = U_before * lens_phase  # immediately after the lens

# Build x–z intensity map
xz = np.zeros((Nz, Nx), dtype=np.float32)
for k in range(Nz):
    z = k * dz
    if z <= z_lens:
        Uz = fresnel_propagate_fft_1d(U0, dx, lam, z, pad_factor=2)
    else:
        Uz = fresnel_propagate_fft_1d(U_after0, dx, lam, z - z_lens, pad_factor=2)
    xz[k, :] = np.abs(Uz)**2

# Plot x–z plane with a marker at the lens position
extent = (-Lx/2, Lx/2, 0.0, z_max)
plt.figure()
plt.imshow(xz, extent=extent, origin='lower', aspect='auto')
plt.xlabel('x [m]'); plt.ylabel('z [m]')
plt.title('Propagation: Gaussian beam with thin lens at z = Lz/2 (1D)')
plt.colorbar(label='Intensity (a.u.)')
plt.axhline(z_lens, linestyle='--')  # lens plane
plt.tight_layout()
plt.show()
