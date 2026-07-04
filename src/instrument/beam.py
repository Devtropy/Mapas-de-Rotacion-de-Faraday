from __future__ import annotations

import math


def fwhm_to_sigma_pixels(fwhm, pixel_size):
    """
    Convierte el FWHM del beam (en unidades físicas, p.ej. kpc) a sigma en
    píxeles, la cantidad que de verdad necesita el filtro gaussiano.

    La relación FWHM = 2*sqrt(2*ln(2)) * sigma es una propiedad geométrica
    de la gaussiana, no algo específico de esta simulación.
    """
    sigma_fisico = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    return sigma_fisico / pixel_size


def _gaussian_filter(xp):
    if xp.__name__ == "cupy":
        from cupyx.scipy.ndimage import gaussian_filter
    else:
        from scipy.ndimage import gaussian_filter
    return gaussian_filter


def apply_beam(map2d, fwhm, pixel_size, xp=None):
    """Convoluciona un mapa 2D con un beam gaussiano de FWHM dado."""
    if xp is None:
        import numpy as xp
    sigma_pix = float(fwhm_to_sigma_pixels(fwhm, pixel_size))
    gaussian_filter = _gaussian_filter(xp)
    return gaussian_filter(map2d, sigma=sigma_pix, mode="nearest")


def apply_beam_stokes(i_map, q_map, u_map, fwhm, pixel_size, xp=None):
    """
    Aplica el mismo beam a I, Q y U. Se convolucionan por separado porque
    cada uno es un mapa independiente: el beam no "sabe" que Q y U describen
    juntos un vector de polarización, solo difumina cada canal como
    cualquier otro mapa de intensidad.
    """
    if xp is None:
        import numpy as xp
    i_beam = apply_beam(i_map, fwhm, pixel_size, xp=xp)
    q_beam = apply_beam(q_map, fwhm, pixel_size, xp=xp)
    u_beam = apply_beam(u_map, fwhm, pixel_size, xp=xp)
    return i_beam, q_beam, u_beam
