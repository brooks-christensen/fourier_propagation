
# Double-slit Fraunhofer pattern
import numpy as np
import matplotlib.pyplot as plt
from fourier_propagation import double_slit, fraunhofer_propagate, imshow_field, plot_intensity

wavelength = 633e-9  # 633 nm
nx = ny = 1024
dx = dy = 5e-6       # 5 microns pixel
slit_width = 50e-6
slit_height = 1e-3
center_sep = 200e-6
z = 1.0              # 1 m to screen

U0 = double_slit(nx, ny, dx, dy, slit_width, slit_height, center_sep)
U1, dx1, dy1 = fraunhofer_propagate(U0, dx, dy, wavelength, z, return_sampling=True, pad_factor=2)
imshow_field(U1, dx1, dy1, title="Double-slit Fraunhofer intensity")
plot_intensity(U1, dx1, dy1, axis='x', title="Central line profile")
plt.show()
