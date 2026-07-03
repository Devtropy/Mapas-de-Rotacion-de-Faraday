import numpy as np

from faradaymr.instrument import apply_beam, depolarization, fwhm_to_sigma_pixels


def test_beam_conserva_el_flujo_total():
    #  convolucionar con un beam redistribuye la señal
    # entre píxeles vecinos, pero un beam gaussiano normalizado no crea ni
    # destruye energía. El flujo total (la suma del mapa) debe conservarse,
    # salvo efectos de borde pequeños.
    rng = np.random.RandomState(0)
    mapa = rng.normal(loc=10.0, scale=1.0, size=(64, 64))

    mapa_beam = apply_beam(mapa, fwhm=6.0, pixel_size=1.0, xp=np)

    assert np.isclose(mapa.sum(), mapa_beam.sum(), rtol=1e-3)


def test_sin_diferencia_entre_mapas_la_despolarizacion_es_uno():
    # Si el mapa "con beam" es idéntico al mapa "sin beam" (caso límite de
    # una fuente perfectamente uniforme, donde el beam no tiene nada que
    # promediar), la despolarización debe ser exactamente 1: no se pierde
    # señal de polarización.
    i_map = np.full((8, 8), 5.0)
    q_map = np.full((8, 8), 1.0)
    u_map = np.full((8, 8), 0.5)

    dp = depolarization(i_map, q_map, u_map, i_map, q_map, u_map, xp=np)

    assert np.allclose(dp, 1.0)


def test_fwhm_a_sigma_es_geometria_pura():
    sigma_pix = fwhm_to_sigma_pixels(fwhm=2.3548 * 2.0, pixel_size=2.0)
    assert np.isclose(sigma_pix, 1.0, atol=1e-3)
