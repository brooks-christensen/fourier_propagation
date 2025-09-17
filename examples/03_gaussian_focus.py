
# Angular spectrum through a thin lens focusing a Gaussian beam
import numpy as np
import matplotlib.pyplot as plt
from fourier_propagation import gaussian_beam, thin_lens_phase, angular_spectrum_propagate, imshow_field

wavelength = 532e-9
nx = ny = 1024
dx = dy = 5e-6
w0 = 200e-6
f = 0.2  # 200 mm
z_after = f

U0 = gaussian_beam(nx, ny, dx, dy, w0, wavelength, z=0.0)
lens = thin_lens_phase(nx, ny, dx, dy, wavelength, f)
U_lens = U0 * lens
U_focus = angular_spectrum_propagate(U_lens, dx, dy, wavelength, z_after, bandlimit=True, pad_factor=2)
imshow_field(U_focus, dx, dy, title="Focused Gaussian intensity (near focus)")
plt.show()
