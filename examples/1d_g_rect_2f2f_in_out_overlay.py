import numpy as np
import matplotlib.pyplot as plt
from fourier_propagation import spatial_grid_1d, rect_slit_1d, thin_lens_phase_1d
from fourier_propagation import fresnel_propagate_fft_1d

lam = 500e-9
dx  = lam/2
Nx  = 4096
x   = spatial_grid_1d(Nx, dx)
Lx  = Nx*dx

z_max  = Nx*dx*dx/lam
z_lens = z_max/2
f      = z_max/4
width  = Lx/5

U0 = rect_slit_1d(Nx, dx, width).astype(complex)

U_mid = fresnel_propagate_fft_1d(U0, dx, lam, z_lens)
U_out = fresnel_propagate_fft_1d(U_mid * thin_lens_phase_1d(Nx, dx, lam, f),
                                 dx, lam, z_max - z_lens)

I_in  = np.abs(U0)**2;  I_in  /= I_in.max()+1e-12
I_out = np.abs(U_out)**2; I_out /= I_out.max()+1e-12

plt.figure()
plt.plot(x, I_in,  label='Input (z = 0), norm')
plt.plot(x, I_out, label='Output (z = z_max), norm')
plt.xlabel('x [m]')
plt.ylabel('Intensity (a.u.)')
plt.legend()
plt.title('Rect (width ≈ Lx/5), 2f-2f imaging: input vs output')
plt.tight_layout() 
plt.show()