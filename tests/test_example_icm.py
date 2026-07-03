import os
import sys

import numpy as np

EXAMPLE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "examples", "icm_faraday_rotation"
)


def _cargar_ejemplo():
    # El ejemplo se escribió como script (config.py/model.py/run.py sueltos,
    # como en la simulación original) en vez de paquete, así que se agrega
    # su carpeta al path para importarlo, igual que si uno se parara ahí a
    # correrlo a mano.
    if EXAMPLE_DIR not in sys.path:
        sys.path.insert(0, EXAMPLE_DIR)
    import config
    import run

    return config, run


def test_pipeline_icm_corre_de_punta_a_punta(tmp_path):
    cfg, run = _cargar_ejemplo()
    # Perspectiva de físico: no importa reproducir el estudio de 128^3
    # celdas en una prueba automática; alcanza con una malla chica que
    # corra en segundos y siga obedeciendo las mismas leyes físicas.
    cfg.N_BASE = 24

    run.RUTA_RESULTADOS = str(tmp_path)
    run.ejecutar_corrida(
        n_spec=3.0, b0_microgauss=1.0, ruta_destino=str(tmp_path), use_gpu=False
    )

    rm_map = np.load(tmp_path / "rm_mapa.npy")
    i_map = np.load(tmp_path / "intensidad.npy")

    assert rm_map.shape == (cfg.N_BASE, cfg.N_BASE)
    # La turbulencia tiene media cero por construcción (potencial vectorial
    # aleatorio de media cero), así que <RM> debe ser mucho más chico que su
    # propia dispersión, no sistemáticamente positivo o negativo.
    assert abs(rm_map.mean()) < rm_map.std()
    # La intensidad sincrotrón es una suma de cantidades no negativas
    # (emisividad por longitud de celda): no puede haber intensidad negativa.
    assert np.all(i_map >= 0)


def test_mayor_b0_produce_mayor_dispersion_de_rm(tmp_path):
    # sigma_RM ∝ n_e * B, así que a igual densidad,
    # un campo magnético diez veces más intenso debe producir una dispersión
    # de RM claramente mayor. Es la prueba de sanidad física más directa que
    # se le puede pedir a este pipeline.
    cfg, run = _cargar_ejemplo()
    cfg.N_BASE = 24

    rng = np.random.RandomState(42)

    ruta_debil = tmp_path / "b0_debil"
    ruta_fuerte = tmp_path / "b0_fuerte"

    # Misma semilla de turbulencia en ambos casos: la única diferencia debe
    # ser la intensidad de campo, no la realización aleatoria.
    import model

    for ruta, b0 in [(ruta_debil, 1.0), (ruta_fuerte, 10.0)]:
        bx, by, bz, ne, ne_rel, r = model.construir_escenario(
            n_spec=3.0, b0_microgauss=b0, use_gpu=False, rng=np.random.RandomState(42)
        )
        from faradaymr import ObservationConfig, ObservationPipeline
        from faradaymr.io import save_maps

        observacion = ObservationConfig(
            pixel_size=cfg.DX_BASE_KPC,
            dl=cfg.DX_BASE_PC,
            frequency=cfg.NU_HZ,
            wavelength=cfg.LAMBDA_ONDA_M,
            p_index=cfg.P_SPEC,
            beam_fwhm=None,
            noise_sigma=None,
        )
        resultado = ObservationPipeline(config=observacion).run(bx, by, bz, ne, ne_rel)
        save_maps(str(ruta), {"rm_mapa": resultado.rm_map})

    rm_debil = np.load(ruta_debil / "rm_mapa.npy")
    rm_fuerte = np.load(ruta_fuerte / "rm_mapa.npy")

    assert rm_fuerte.std() > rm_debil.std()


def test_construir_escenario_acepta_cualquier_perfil_de_densidad(tmp_path):
    # Perspectiva de físico: este es el objetivo central de la
    cfg, run = _cargar_ejemplo()
    cfg.N_BASE = 16

    import model

    from faradaymr import BetaModel, DoubleBetaModel, NFWModel

    rng_semilla = 7
    _, _, _, ne_beta, _, r = model.construir_escenario(
        n_spec=3.0,
        b0_microgauss=1.0,
        density_profile=BetaModel(n0=cfg.N0_CM3, r_core=cfg.RC_KPC, beta=cfg.BETA),
        use_gpu=False,
        rng=np.random.RandomState(rng_semilla),
    )
    _, _, _, ne_nfw, _, _ = model.construir_escenario(
        n_spec=3.0,
        b0_microgauss=1.0,
        density_profile=NFWModel(n0=1e-2, r_s=300.0),
        use_gpu=False,
        rng=np.random.RandomState(rng_semilla),
    )
    _, _, _, ne_doble, _, _ = model.construir_escenario(
        n_spec=3.0,
        b0_microgauss=1.0,
        density_profile=DoubleBetaModel(
            n0_1=5e-3, r_core_1=50.0, beta_1=0.5, n0_2=1e-3, r_core_2=400.0, beta_2=0.7
        ),
        use_gpu=False,
        rng=np.random.RandomState(rng_semilla),
    )

    # Misma malla de radios, misma semilla de turbulencia: la única
    # diferencia entre las tres corridas es el objeto de densidad, así que
    # los tres campos n_e resultantes deben ser distintos entre sí.
    assert ne_beta.shape == ne_nfw.shape == ne_doble.shape == r.shape
    assert not np.allclose(ne_beta, ne_nfw)
    assert not np.allclose(ne_beta, ne_doble)
    # Y si no se pasa ningún density_profile, debe recuperarse exactamente
    # el comportamiento anterior (perfil beta con los parámetros de config).
    _, _, _, ne_por_defecto, _, _ = model.construir_escenario(
        n_spec=3.0,
        b0_microgauss=1.0,
        use_gpu=False,
        rng=np.random.RandomState(rng_semilla),
    )
    assert np.allclose(ne_beta, ne_por_defecto)
