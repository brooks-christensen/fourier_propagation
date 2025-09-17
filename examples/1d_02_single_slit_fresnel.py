
# 1D single slit Fresnel propagation at multiple z
import numpy as np
import matplotlib.pyplot as plt
from fourier_propagation import rect_slit_1d, fresnel_propagate_fft_1d, plot_intensity_1d

wavelength = 532e-9
n = 4096
dx = 2.0e-6
width = 100e-6

U0 = rect_slit_1d(n, dx, width)
for z in [0.01, 0.05, 0.1]:
    U1 = fresnel_propagate_fft_1d(U0, dx, wavelength, z, pad_factor=2)
    plot_intensity_1d(U1, dx, title=f"1D Single slit Fresnel z={z} m")
plt.show()
