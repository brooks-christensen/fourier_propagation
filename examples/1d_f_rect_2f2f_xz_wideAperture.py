import numpy as np
import matplotlib.pyplot as plt
from fourier_propagation import spatial_grid_1d, rect_slit_1d, thin_lens_phase_1d, angular_spectrum_propagate_1d

lam = 500e-9
dx  = lam/2
Nx  = 4096
x   = spatial_grid_1d(Nx, dx)
Lx  = Nx*dx

z_max  = Nx*dx*dx/lam
z_lens = z_max/2
f      = z_max/4
width  = Lx/5                       # ~ one-fifth of window

U0 = rect_slit_1d(Nx, dx, width).astype(complex)

def prop(u, z): return angular_spectrum_propagate_1d(u, dx, lam, z, bandlimit=True, pad_factor=2)
U_lens_in  = prop(U0, z_lens)
U_lens_out = U_lens_in * thin_lens_phase_1d(Nx, dx, lam, f)

dz = 2*dx; Nz = int(np.round(z_max/dz))
xz = np.zeros((Nz, Nx), dtype=np.float32)
for k in range(Nz):
    z = k*dz
    Uz = prop(U0, z) if z <= z_lens else prop(U_lens_out, z - z_lens)
    xz[k,:] = np.abs(Uz)**2

plt.figure()
plt.imshow(xz, extent=(x[0], x[-1], 0.0, z_max), origin='lower', aspect='auto')
plt.colorbar(label='Intensity (a.u.)')
plt.axhline(z_lens, ls='--', color='w')
plt.xlabel('x [m]'); plt.ylabel('z [m]')
plt.title('Rect (width ≈ Lx/5), 2f-2f imaging to z_max')
plt.tight_layout(); plt.show()