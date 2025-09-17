
import numpy as np
from .sampling1d import spatial_grid_1d

def rect_slit_1d(n, dx, width, center=0.0):
    x = spatial_grid_1d(n, dx)
    return (np.abs(x - center) <= width/2).astype(float)

def double_slit_1d(n, dx, slit_width, center_sep):
    a = rect_slit_1d(n, dx, slit_width, center=-center_sep/2)
    b = rect_slit_1d(n, dx, slit_width, center=+center_sep/2)
    return a + b
