from __future__ import annotations


def polarization_fraction(i_map, q_map, u_map, xp=None, epsilon=1e-12):
    """Devuelve (P, P/I) a partir de los mapas de Stokes."""
    if xp is None:
        import numpy as xp
    p_map = xp.sqrt(q_map**2 + u_map**2)
    frac_pol = p_map / (i_map + epsilon)
    return p_map, frac_pol


def depolarization(i_map, q_map, u_map, i_beam, q_beam, u_beam, xp=None, epsilon=1e-12):
    """
    DP = (P/I)_beam / (P/I)_sin_beam, la razón entre la fracción de
    polarización observada con el beam y la que tendría el instrumento con
    resolución infinita.
    """
    _, frac_sin_beam = polarization_fraction(
        i_map, q_map, u_map, xp=xp, epsilon=epsilon
    )
    _, frac_con_beam = polarization_fraction(
        i_beam, q_beam, u_beam, xp=xp, epsilon=epsilon
    )
    return frac_con_beam / (frac_sin_beam + epsilon)
