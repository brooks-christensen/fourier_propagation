
# Overlay input and output intensities for the same configuration as the x–z plot
import numpy as np
import matplotlib.pyplot as plt
from fourier_propagation import rect_slit_1d, fresnel_propagate_fft_1d

Nx = 1024
lam = 500e-9
dx = lam / 2
dz = 2*dx
Lx = Nx * dx
aperture_width = 6 * lam
# Nz = 1024
Nz = 256
z_out = Nz * dz

U_in = rect_slit_1d(Nx, dx, aperture_width).astype(complex)
U_out = fresnel_propagate_fft_1d(U_in, dx, lam, z_out, pad_factor=2)

x = (np.arange(Nx) - Nx//2) * dx
I_in = (np.abs(U_in)**2) # / (np.max(np.abs(U_in)**2) + 1e-12)
I_out = (np.abs(U_out)**2) # / (np.max(np.abs(U_out)**2) + 1e-12)

plt.figure()
plt.plot(x, I_in, label='Input z=0 (normalized)')
plt.plot(x, I_out, label=f'Output z={z_out:.3e} m (normalized)')
plt.xlabel('x [m]'); plt.ylabel('Intensity (a.u.)')
plt.title('Input vs Output Intensities (overlaid)')
plt.legend()
plt.tight_layout()
plt.show()
