
import numpy as np

def spatial_grid(nx:int, ny:int, dx:float, dy:float):
    """Return spatial coordinate grids (x, y) and meshgrid (X, Y)."""
    x = (np.arange(nx) - nx//2) * dx
    y = (np.arange(ny) - ny//2) * dy
    X, Y = np.meshgrid(x, y, indexing='xy')
    return x, y, X, Y

def freq_grid(nx:int, ny:int, dx:float, dy:float):
    """Return frequency grids (fx, fy) and meshgrid (FX, FY) in cycles per meter."""
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)
    FX, FY = np.meshgrid(np.fft.fftshift(fx), np.fft.fftshift(fy), indexing='xy')
    return np.fft.fftshift(fx), np.fft.fftshift(fy), FX, FY

def pad_to_shape(field:np.ndarray, shape):
    """Zero-pad (or crop) a 2D field to target shape centered."""
    ny, nx = field.shape
    Ny, Nx = shape
    out = np.zeros((Ny, Nx), dtype=field.dtype)
    sy = min(ny, Ny)
    sx = min(nx, Nx)
    y0 = Ny//2 - sy//2; x0 = Nx//2 - sx//2
    out[y0:y0+sy, x0:x0+sx] = field[ny//2 - sy//2: ny//2 - sy//2 + sy,
                                    nx//2 - sx//2: nx//2 - sx//2 + sx]
    return out
