
import numpy as np
from .sampling import spatial_grid

def rect_aperture(nx, ny, dx, dy, width_x, width_y, center=(0.0, 0.0)):
    x, y, X, Y = spatial_grid(nx, ny, dx, dy)
    cx, cy = center
    return ((np.abs(X - cx) <= width_x/2) & (np.abs(Y - cy) <= width_y/2)).astype(float)

def circ_aperture(nx, ny, dx, dy, radius, center=(0.0, 0.0)):
    x, y, X, Y = spatial_grid(nx, ny, dx, dy)
    cx, cy = center
    return ( (X - cx)**2 + (Y - cy)**2 <= radius**2 ).astype(float)

def double_slit(nx, ny, dx, dy, slit_width, slit_height, center_sep):
    """Two rectangular slits separated along x by center_sep."""
    a = rect_aperture(nx, ny, dx, dy, slit_width, slit_height, center=(-center_sep/2, 0.0))
    b = rect_aperture(nx, ny, dx, dy, slit_width, slit_height, center=(+center_sep/2, 0.0))
    return a + b
