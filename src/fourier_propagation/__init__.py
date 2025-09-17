
from .sampling import spatial_grid, freq_grid, pad_to_shape
from .fields import gaussian_beam, plane_wave
from .apertures import rect_aperture, circ_aperture, double_slit
from .lenses import thin_lens_phase
from .propagation import fresnel_propagate_fft, fraunhofer_propagate, angular_spectrum_propagate
from .plotting import imshow_field, plot_intensity
