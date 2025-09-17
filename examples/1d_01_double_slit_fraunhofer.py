
# 1D double slit Fraunhofer pattern
import numpy as np
import matplotlib.pyplot as plt
from fourier_propagation import double_slit_1d, fraunhofer_propagate_1d, plot_intensity_1d

wavelength = 633e-9
n = 4096
dx = 2.5e-6
slit_width = 50e-6
center_sep = 200e-6
z = 1.0

U0 = double_slit_1d(n, dx, slit_width, center_sep)
U1, dx1 = fraunhofer_propagate_1d(U0, dx, wavelength, z, return_sampling=True, pad_factor=2)
plot_intensity_1d(U1, dx1, title="1D Double-slit Fraunhofer intensity")
plt.show()
