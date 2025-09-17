
import numpy as np
import matplotlib.pyplot as plt

def plot_intensity_1d(U, dx, title=None):
    I = np.abs(U)**2
    x = (np.arange(I.size) - I.size//2) * dx
    plt.figure()
    plt.plot(x, I)
    plt.xlabel('x [m]'); plt.ylabel('Intensity (a.u.)')
    if title: plt.title(title)
    plt.tight_layout()
