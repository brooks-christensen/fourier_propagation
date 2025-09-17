import numpy as np
import matplotlib.pyplot as plt
from fourier_propagation import spatial_grid_1d, rect_slit_1d, thin_lens_phase_1d
from fourier_propagation import fresnel_propagate_fft_1d

lam = 500e-9
dx  = lam/2
Nx  = 4096
x   = spatial_grid_1d(Nx, dx)

z_max  = Nx*dx*dx/lam
z_lens = z_max/2
f      = z_max/4
width  = 6*lam

U0 = rect_slit_1d(Nx, dx, width).astype(complex)
U_mid = fresnel_propagate_fft_1d(U0, dx, lam, z_lens)

U_out = fresnel_propagate_fft_1d(U_mid * thin_lens_phase_1d(Nx, dx, lam, f),
                                 dx, lam, z_max - z_lens)

I_mid = np.abs(U_mid)**2; I_mid /= I_mid.max()+1e-12
I_out = np.abs(U_out)**2; I_out /= I_out.max()+1e-12

plt.figure()
plt.plot(x, I_mid, label='Just before lens (z = z_max/2), norm')
plt.plot(x, I_out, label='Output (z = z_max), norm')
plt.xlabel('x [m]'); plt.ylabel('Intensity (a.u.)'); plt.legend()
plt.title('Rect + 2f-2f: midplane vs output')
plt.tight_layout(); plt.show()