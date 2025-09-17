
# fourier-propagation

Numerical scalar diffraction and Fourier propagation in Python using FFTs.

Implements:
- Fresnel propagation (FFT-based)
- Fraunhofer (far-field) propagation
- Angular Spectrum Method (ASM) with optional band-limiting
- Thin-lens phase
- Common apertures (slits, circular, rectangular)
- Sampling helpers and Nyquist conditions

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .
python examples/01_double_slit.py
```

See `examples/` for figure reproduction templates.
