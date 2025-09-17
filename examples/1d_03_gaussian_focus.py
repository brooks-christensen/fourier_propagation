
# 1D Gaussian beam through a thin lens (focus)
import numpy as np
import matplotlib.pyplot as plt
from fourier_propagation import gaussian_1d, thin_lens_phase_1d, angular_spectrum_propagate_1d, plot_intensity_1d

wavelength = 532e-9
n = 4096
dx = 2.5e-6
w0 = 200e-6
f = 0.2
z_after = f

U0 = gaussian_1d(n, dx, w0, wavelength, z=0.0)
lens = thin_lens_phase_1d(n, dx, wavelength, f)
U_lens = U0 * lens
U_focus = angular_spectrum_propagate_1d(U_lens, dx, wavelength, z_after, bandlimit=True, pad_factor=2)
plot_intensity_1d(U_focus, dx, title="1D Focused Gaussian intensity (near focus)")
plt.show()
