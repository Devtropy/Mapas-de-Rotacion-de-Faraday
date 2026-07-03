import numpy as np
import pytest

from faradaymr import (
    BetaModel,
    DensityProfile,
    DoubleBetaModel,
    NFWModel,
    TabulatedProfile,
    beta_model,
)


def test_beta_model_en_el_centro_es_n0():
    # en r=0 el término (1+(r/rc)^2) vale 1 sin
    # importar r_core ni beta, así que n_e(0) debe ser exactamente n0.
    perfil = BetaModel(n0=1e-3, r_core=400.0, beta=0.6)
    assert np.isclose(perfil.density(np.array([0.0]))[0], 1e-3)


def test_beta_model_es_monotona_decreciente():
    perfil = BetaModel(n0=1e-3, r_core=400.0, beta=0.6)
    r = np.linspace(0, 2000, 100)
    assert np.all(np.diff(perfil.density(r)) <= 0)


def test_beta_model_coincide_con_la_funcion_retrocompatible():
    # La función `beta_model` se dejó como envoltorio de BetaModel: deben
    # dar exactamente el mismo resultado para los mismos parámetros.
    r = np.linspace(1.0, 1000.0, 50)
    perfil = BetaModel(n0=2e-3, r_core=250.0, beta=0.7)
    assert np.allclose(perfil.density(r), beta_model(r, 2e-3, 250.0, 0.7))


def test_beta_model_es_llamable_como_funcion():
    perfil = BetaModel(n0=1e-3, r_core=400.0, beta=0.6)
    r = np.linspace(0, 500, 20)
    assert np.allclose(perfil(r), perfil.density(r))


def test_double_beta_es_la_suma_de_sus_dos_componentes():
    # en r=0 ambas componentes valen su n0
    # respectivo, así que la suma debe ser exactamente n0_1 + n0_2.
    doble = DoubleBetaModel(
        n0_1=5e-3, r_core_1=50.0, beta_1=0.5, n0_2=1e-3, r_core_2=400.0, beta_2=0.7
    )
    assert np.isclose(doble.density(np.array([0.0]))[0], 6e-3)

    componente_1 = BetaModel(n0=5e-3, r_core=50.0, beta=0.5)
    componente_2 = BetaModel(n0=1e-3, r_core=400.0, beta=0.7)
    r = np.linspace(0, 1000, 50)
    assert np.allclose(
        doble.density(r), componente_1.density(r) + componente_2.density(r)
    )


def test_nfw_en_r_igual_a_rs_vale_n0_sobre_4():
    # en r = r_s, x=1, así que n0/[(1)*(1+1)^2] = n0/4
    # es un punto de referencia analítico simple para verificar la fórmula.
    perfil = NFWModel(n0=1e-2, r_s=300.0)
    assert np.isclose(perfil.density(np.array([300.0]))[0], 1e-2 / 4.0)


def test_nfw_no_diverge_en_r_cero():
    # El perfil NFW diverge formalmente en r=0 (~1/r); la implementación
    # debe devolver un valor finito (saturado) en vez de un NaN o inf que
    # arruinaría el resto del pipeline de integración.
    perfil = NFWModel(n0=1e-2, r_s=300.0)
    valor = perfil.density(np.array([0.0]))[0]
    assert np.isfinite(valor)
    assert valor > 0


def test_tabulated_reproduce_los_puntos_de_la_tabla():
    tabla = TabulatedProfile(
        r_table=[1, 10, 100, 1000], n_e_table=[1.0, 1e-1, 1e-2, 1e-3]
    )
    assert np.allclose(tabla.density(np.array([10.0, 100.0])), [1e-1, 1e-2])


def test_tabulated_interpola_una_ley_de_potencia_exactamente():
    # si la tabla sigue una ley de potencia exacta
    # (como aquí, n_e ∝ r^-1), la interpolación log-log debe reproducirla
    # sin error en cualquier punto intermedio, no solo en los nodos de la
    # tabla (a diferencia de una interpolación lineal en espacio real).
    r_tabla = np.array([1.0, 10.0, 100.0, 1000.0])
    n_tabla = 1.0 / r_tabla
    tabla = TabulatedProfile(r_table=r_tabla, n_e_table=n_tabla)

    r_consulta = np.array([2.0, 5.0, 31.6227766, 500.0])
    assert np.allclose(tabla.density(r_consulta), 1.0 / r_consulta, rtol=1e-6)


def test_tabulated_sin_extrapolar_satura_al_borde():
    tabla = TabulatedProfile(
        r_table=[1, 10, 100], n_e_table=[1.0, 0.1, 0.01], extrapolate=False
    )
    assert np.isclose(tabla.density(np.array([0.001]))[0], 1.0)
    assert np.isclose(tabla.density(np.array([1e6]))[0], 0.01)


def test_tabulated_con_extrapolar_continua_la_ley_de_potencia():
    r_tabla = np.array([1.0, 10.0, 100.0])
    n_tabla = 1.0 / r_tabla  # ley de potencia exacta, pendiente log-log = -1
    tabla = TabulatedProfile(r_table=r_tabla, n_e_table=n_tabla, extrapolate=True)
    assert np.isclose(tabla.density(np.array([1000.0]))[0], 1e-3, rtol=1e-6)


def test_tabulated_rechaza_valores_no_positivos():
    with pytest.raises(ValueError):
        TabulatedProfile(r_table=[0.0, 10.0], n_e_table=[1.0, 0.1])
    with pytest.raises(ValueError):
        TabulatedProfile(r_table=[1.0, 10.0], n_e_table=[1.0, -0.1])


def test_todos_los_perfiles_respetan_la_interfaz_densityprofile():
    perfiles = [
        BetaModel(n0=1e-3, r_core=400.0, beta=0.6),
        DoubleBetaModel(
            n0_1=5e-3, r_core_1=50.0, beta_1=0.5, n0_2=1e-3, r_core_2=400.0, beta_2=0.7
        ),
        NFWModel(n0=1e-2, r_s=300.0),
        TabulatedProfile(r_table=[1, 10, 100], n_e_table=[1.0, 0.1, 0.01]),
    ]
    r = np.linspace(1, 500, 30)
    for perfil in perfiles:
        assert isinstance(perfil, DensityProfile)
        resultado = perfil.density(r)
        assert resultado.shape == r.shape
        assert np.all(np.isfinite(resultado))
