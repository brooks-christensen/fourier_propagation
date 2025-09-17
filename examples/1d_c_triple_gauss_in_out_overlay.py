import numpy as np
import matplotlib.pyplot as plt
from fourier_propagation import spatial_grid_1d, thin_lens_phase_1d
from fourier_propagation import fresnel_propagate_fft_1d

lam = 500e-9
dx  = lam/2
Nx  = 4096
x   = spatial_grid_1d(Nx, dx)

z_max  = Nx*dx*dx/lam
z_lens = z_max/2
f      = z_max/4
w0     = 1.5*lam
offset = 10*lam

U0 = ( np.exp(-(x**2)/(w0**2))
     + np.exp(-((x + offset)**2)/(w0**2))
     + np.exp(-((x + 2*offset)**2)/(w0**2)) ).astype(complex)

U_lens_in  = fresnel_propagate_fft_1d(U0, dx, lam, z_lens)
U_lens_out = U_lens_in * thin_lens_phase_1d(Nx, dx, lam, f)
U_out      = fresnel_propagate_fft_1d(U_lens_out, dx, lam, z_max - z_lens)

I_in  = (np.abs(U0)**2);  I_in  /= I_in.max() + 1e-12
I_out = (np.abs(U_out)**2); I_out /= I_out.max() + 1e-12

plt.figure()
plt.plot(x, I_in,  label='Input z = 0 (norm)')
plt.plot(x, I_out, label=f'Output z = {z_max:.3e} m (norm)')
plt.xlabel('x [m]'); plt.ylabel('Intensity (a.u.)'); plt.legend()
plt.title('Triple-Gaussian + thin lens (focus at z_max): input vs output')
plt.tight_layout(); plt.show()