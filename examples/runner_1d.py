# examples/runner_1d.py
import numpy as np
import matplotlib.pyplot as plt
from fourier_propagation import (
    spatial_grid_1d, rect_slit_1d, gaussian_1d, thin_lens_phase_1d,
    fresnel_propagate_fft_1d, angular_spectrum_propagate_1d
)

def default_params():
    lam = 500e-9
    dx  = lam/2
    Nx  = 4096
    x   = spatial_grid_1d(Nx, dx)
    z_max  = Nx*dx*dx/lam
    z_lens = z_max/2
    f      = z_max/4
    return dict(lam=lam, dx=dx, Nx=Nx, x=x, z_max=z_max, z_lens=z_lens, f=f)

# ---------- Field builders ----------
def field_gaussian(p, w0=1.5e0, center=0.0):
    lam = p['lam']
    if isinstance(w0, float):
        w0m = w0
    else:
        w0m = 1.5*p['lam']
    x = p['x']
    return np.exp(-((x-center)**2)/(w0m**2)).astype(complex)

def field_triple_gauss(p, w0=None, offset=10):
    lam = p['lam']; x = p['x']
    w0m = (1.5*lam) if w0 is None else w0
    off = offset*lam
    U = ( np.exp(-(x**2)/(w0m**2))
        + np.exp(-((x+off)**2)/(w0m**2))
        + np.exp(-((x+2*off)**2)/(w0m**2)) ).astype(complex)
    return U

def field_rect(p, width='6lam'):
    lam = p['lam']; dx = p['dx']; Nx = p['Nx']
    if width == 'Lx/5':
        return rect_slit_1d(Nx, dx, Nx*dx/5).astype(complex)
    elif width == '6lam':
        return rect_slit_1d(Nx, dx, 6*lam).astype(complex)
    else:
        return rect_slit_1d(Nx, dx, float(width)).astype(complex)

# ---------- Experiments ----------
def exp_overlay_in_out(U0, p, with_lens=False):
    lam, dx = p['lam'], p['dx']
    z_max, z_lens, f = p['z_max'], p['z_lens'], p['f']
    if with_lens:
        U_mid = fresnel_propagate_fft_1d(U0, dx, lam, z_lens)
        U_out = fresnel_propagate_fft_1d(U_mid * thin_lens_phase_1d(p['Nx'], dx, lam, f),
                                         dx, lam, z_max - z_lens)
    else:
        U_out = fresnel_propagate_fft_1d(U0, dx, lam, z_max)

    x = p['x']
    I_in  = (np.abs(U0 )**2); I_in  /= I_in.max()+1e-12
    I_out = (np.abs(U_out)**2); I_out /= I_out.max()+1e-12
    plt.figure(); plt.plot(x, I_in, label='input'); plt.plot(x, I_out, label='output')
    plt.xlabel('x [m]'); plt.ylabel('Intensity (a.u.)'); plt.legend(); plt.tight_layout(); plt.show()

def exp_xz(U0, p, with_lens=False):
    lam, dx = p['lam'], p['dx']
    z_max, z_lens, f = p['z_max'], p['z_lens'], p['f']
    prop = lambda U,z: angular_spectrum_propagate_1d(U, dx, lam, z, bandlimit=True, pad_factor=2)
    if with_lens:
        U_lens_out = prop(U0, z_lens) * thin_lens_phase_1d(p['Nx'], dx, lam, f)
    dz = 2*dx; Nz = int(np.round(z_max/dz))
    xz = np.zeros((Nz, p['Nx']), dtype=np.float32)
    for k in range(Nz):
        z = k*dz
        if not with_lens or z <= z_lens:
            Uz = prop(U0, z)
        else:
            Uz = prop(U_lens_out, z - z_lens)
        xz[k,:] = np.abs(Uz)**2
    x = p['x']; plt.figure()
    plt.imshow(xz, extent=(x[0], x[-1], 0.0, z_max), origin='lower', aspect='auto')
    if with_lens: plt.axhline(z_lens, ls='--', color='w')
    plt.colorbar(label='Intensity (a.u.)'); plt.xlabel('x [m]'); plt.ylabel('z [m]')
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    p = default_params()
    # Examples:
    U_gauss  = field_gaussian(p)                # single Gaussian
    U_triple = field_triple_gauss(p, offset=10) # 0, -10λ, -20λ
    U_rect   = field_rect(p, width='Lx/5')

    # Overlay, with lens
    exp_overlay_in_out(U_gauss,  p, with_lens=True)
    exp_overlay_in_out(U_triple, p, with_lens=True)
    exp_overlay_in_out(U_rect,   p, with_lens=True)

    # xz plots
    exp_xz(U_triple, p, with_lens=True)
    exp_xz(U_rect,   p, with_lens=True)
