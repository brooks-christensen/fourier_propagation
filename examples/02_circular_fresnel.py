
# Fresnel propagation of circular aperture
import numpy as np
import matplotlib.pyplot as plt
from fourier_propagation import circ_aperture, fresnel_propagate_fft, imshow_field

wavelength = 532e-9
nx = ny = 1024
dx = dy = 4e-6
radius = 100e-6

U0 = circ_aperture(nx, ny, dx, dy, radius)
for z in [0.01, 0.05, 0.1]:
    U1 = fresnel_propagate_fft(U0, dx, dy, wavelength, z, pad_factor=2)
    imshow_field(U1, dx, dy, title=f"Circular aperture Fresnel z={z} m")
plt.show()
