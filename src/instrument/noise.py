from __future__ import annotations


def add_gaussian_noise(map_, sigma, xp=None, rng=None):
    """Suma ruido gaussiano de desviación estándar `sigma` al mapa dado."""
    if xp is None:
        import numpy as xp
    random = xp.random if rng is None else rng
    ruido = random.normal(loc=0.0, scale=sigma, size=map_.shape)
    return map_ + ruido
