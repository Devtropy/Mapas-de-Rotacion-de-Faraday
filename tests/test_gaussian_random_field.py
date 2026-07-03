import os

import numpy as np

from faradaymr.fields import GaussianRandomVectorField


def test_campo_es_solenoidal():
    #  si el campo se construyó como rotacional de un
    # potencial vectorial, no debería haber "fuentes" de campo magnético
    # (monopolos), solo el ruido numérico propio de la FFT en una malla
    # discreta y finita.
    campo = GaussianRandomVectorField(
        n=16, dx=1.0, spectral_index=3.0, scale_min=2.0, scale_max=8.0
    )
    bx, by, bz = campo.sample(use_gpu=False, rng=np.random.RandomState(0))

    div = GaussianRandomVectorField.divergence(bx, by, bz, dx=1.0, xp=np)
    escala_campo = np.sqrt(np.mean(bx**2 + by**2 + bz**2))

    assert np.mean(np.abs(div)) < 0.5 * escala_campo


def test_normalizacion_fija_el_rms():
    campo = GaussianRandomVectorField(
        n=16, dx=1.0, spectral_index=3.0, scale_min=2.0, scale_max=8.0
    )
    bx, by, bz = campo.sample(use_gpu=False, rng=np.random.RandomState(1))

    b0 = 5.0
    bx_n, by_n, bz_n = GaussianRandomVectorField.normalize_to_rms(bx, by, bz, b0, xp=np)
    b_rms = np.sqrt(np.mean(bx_n**2 + by_n**2 + bz_n**2))

    assert np.isclose(b_rms, b0, rtol=1e-6)


def test_sample_cached_recupera_de_disco_sin_recalcular(tmp_path):
    # Con la misma semilla y los mismos parámetros espectrales, la segunda
    # llamada tiene que devolver *exactamente* la malla guardada en la
    # primera (no una nueva realización aleatoria).
    campo = GaussianRandomVectorField(
        n=8, dx=1.0, spectral_index=3.0, scale_min=2.0, scale_max=4.0
    )

    bx1, by1, bz1 = campo.sample_cached(str(tmp_path), use_gpu=False, seed=0)

    archivos = [f for f in os.listdir(tmp_path) if f.endswith(".npz")]
    assert len(archivos) == 1

    bx2, by2, bz2 = campo.sample_cached(str(tmp_path), use_gpu=False, seed=0)

    assert np.array_equal(bx1, bx2)
    assert np.array_equal(by1, by2)
    assert np.array_equal(bz1, bz2)
    # Sigue habiendo un solo archivo de caché: la segunda llamada no generó
    # (ni guardó) una malla nueva.
    assert len([f for f in os.listdir(tmp_path) if f.endswith(".npz")]) == 1


def test_sample_cached_distingue_parametros_distintos(tmp_path):
    # Dos configuraciones espectrales distintas no deben compartir archivo
    # de caché: cada combinación de parámetros necesita su propia malla
    # guardada, o un estudio paramétrico terminaría reusando por error la
    # malla de otra corrida.
    campo_a = GaussianRandomVectorField(
        n=8, dx=1.0, spectral_index=2.0, scale_min=2.0, scale_max=4.0
    )
    campo_b = GaussianRandomVectorField(
        n=8, dx=1.0, spectral_index=3.0, scale_min=2.0, scale_max=4.0
    )

    campo_a.sample_cached(str(tmp_path), use_gpu=False, seed=0)
    campo_b.sample_cached(str(tmp_path), use_gpu=False, seed=0)

    archivos = [f for f in os.listdir(tmp_path) if f.endswith(".npz")]
    assert len(archivos) == 2


def test_save_y_load_recuperan_el_mismo_campo(tmp_path):
    campo = GaussianRandomVectorField(
        n=8, dx=1.0, spectral_index=3.0, scale_min=2.0, scale_max=4.0
    )
    bx, by, bz = campo.sample(use_gpu=False, rng=np.random.RandomState(0))

    ruta = str(tmp_path / "malla.npz")
    campo.save(ruta, bx, by, bz, seed=0)
    bx_cargado, by_cargado, bz_cargado = GaussianRandomVectorField.load(
        ruta, use_gpu=False
    )

    assert np.array_equal(bx, bx_cargado)
    assert np.array_equal(by, by_cargado)
    assert np.array_equal(bz, bz_cargado)
