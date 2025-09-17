
import numpy as np
import matplotlib.pyplot as plt

def imshow_field(U, dx, dy, title=None):
    I = np.abs(U)**2
    ext = [-(U.shape[1]//2)*dx, (U.shape[1]//2)*dx, -(U.shape[0]//2)*dy, (U.shape[0]//2)*dy]
    plt.figure()
    plt.imshow(np.fft.fftshift(I), extent=ext, origin='lower')
    plt.xlabel('x [m]'); plt.ylabel('y [m]')
    if title: plt.title(title)
    plt.colorbar(label='Intensity (a.u.)')
    plt.tight_layout()

def plot_intensity(U, dx, dy, axis='x', title=None):
    I = np.abs(U)**2
    I = np.fft.fftshift(I)
    if axis == 'x':
        prof = I[I.shape[0]//2, :]
        x = (np.arange(I.shape[1]) - I.shape[1]//2) * dx
        xlabel = 'x [m]'
    else:
        prof = I[:, I.shape[1]//2]
        x = (np.arange(I.shape[0]) - I.shape[0]//2) * dy
        xlabel = 'y [m]'
    plt.figure()
    plt.plot(x, prof)
    plt.xlabel(xlabel); plt.ylabel('Intensity (a.u.)')
    if title: plt.title(title)
    plt.tight_layout()
