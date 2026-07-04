from __future__ import annotations

import os

import config as cfg
from model import construir_escenario
from plots import generar_graficos_estudio

from faradaymr import (
    BetaModel,
    DensityProfile,
    ObservationConfig,
    ObservationPipeline,
    get_backend,
)
from faradaymr.io import save_maps

RUTA_RESULTADOS = os.path.join(
    os.path.dirname(__file__), "..", "..", "results", "estudio_parametrico_icm"
)


def ejecutar_corrida(
    n_spec: float,
    b0_microgauss: float,
    ruta_destino: str,
    use_gpu=True,
    density_profile: DensityProfile | None = None,
):
    if density_profile is None:
        density_profile = BetaModel(n0=cfg.N0_CM3, r_core=cfg.RC_KPC, beta=cfg.BETA)

    xp = get_backend(use_gpu)

    bx, by, bz, ne, ne_rel, r = construir_escenario(
        n_spec, b0_microgauss, density_profile=density_profile, use_gpu=use_gpu
    )

    observacion = ObservationConfig(
        pixel_size=cfg.DX_BASE_KPC,
        dl=cfg.DX_BASE_PC,
        frequency=cfg.NU_HZ,
        wavelength=cfg.LAMBDA_ONDA_M,
        p_index=cfg.P_SPEC,
        beam_fwhm=cfg.BEAM_FWHM_KPC,
        noise_sigma=cfg.DESV_EST_RUIDO if cfg.AGREGAR_RUIDO else None,
    )
    pipeline = ObservationPipeline(config=observacion)
    resultado = pipeline.run(bx, by, bz, ne, ne_rel)

    save_maps(
        ruta_destino,
        {
            "rm_mapa": resultado.rm_map,
            "intensidad": resultado.i_map,
            "stokes_q": resultado.q_map,
            "stokes_u": resultado.u_map,
            "rm_mapa_beam": resultado.rm_beam,
            "i_beam": resultado.i_beam,
            "q_beam": resultado.q_beam,
            "u_beam": resultado.u_beam,
            "intensidad_de_polaridad": resultado.p_beam,
            "fraccion_de_polarizacion": resultado.frac_pol_beam,
            "despolarizacion": resultado.depolarization_map,
        },
    )

    generar_graficos_estudio(ruta_destino, n_spec=n_spec, b0_microgauss=b0_microgauss)


ejecutar_corrida(cfg.N_SPEC, cfg.B0_MG, RUTA_RESULTADOS)
