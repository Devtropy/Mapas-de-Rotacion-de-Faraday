import cupy as cp
import os
import config_values as cfg
from grid import generar_capa_campo
from simulation.faraday import calcular_sincrotron
from simulation.Polaridad import calcular_mapas_polarizacion
from visualizer import generar_graficos_estudio
from observatorio.beam import aplicar_observacion


def ejecutar_simulacion(n_val, b0_val, ruta_destino):
    cfg.N_SPEC = n_val
    b0_mg = b0_val

    bx, by, bz,_,_,_ = generar_capa_campo(cfg.N_BASE, cfg.DX_BASE_KPC)
    b_rms = cp.sqrt(cp.mean(bx**2 + by**2 + bz**2))
    bx /= b_rms
    by /= b_rms
    bz /= b_rms

    eje = cp.linspace(-cfg.N_BASE / 2, cfg.N_BASE / 2, cfg.N_BASE) * cfg.DX_BASE_KPC
    xx, yy, zz = cp.meshgrid(eje, eje, eje, indexing="ij")
    r = cp.sqrt(xx**2 + yy**2 + zz**2)

    ne = (cfg.N0_CM3 * (1.0 + (r / cfg.RC_KPC) ** 2) ** (-1.5 * cfg.BETA)).astype(
        cp.float32
    )

    perfil_b = b0_mg * cp.power(ne / cfg.N0_CM3, cfg.MU)

    bx *= perfil_b
    by *= perfil_b
    bz *= perfil_b

    ne_rel = cfg.N0_CM3* cp.power(perfil_b / b0_mg, 2)

    i_map, j_nu = calcular_sincrotron(bx, by, ne_rel)
    
    q_map, u_map = calcular_mapas_polarizacion(bx, by, bz, j_nu, ne)

    rm_map = cp.sum(0.812 * ne * bz * cfg.DX_BASE_PC, axis=2)
    
    rm_beam, i_beam, q_beam, u_beam, p_beam, frac_pol, dp = aplicar_observacion(
        rm_map, i_map, q_map, u_map, cfg.BEAM_ARCSEC_VAL, cfg.D_A_MPC_VAL, cfg.DX_BASE_KPC, cfg.NOISE_RMS_VAL
    )

    ruta_datos = os.path.join(ruta_destino, "data")
    ruta_graficos = os.path.join(ruta_destino, "plots")

    os.makedirs(ruta_datos, exist_ok=True)

    cp.save(os.path.join(ruta_datos, "rm_mapa.npy"), rm_map.get())
    cp.save(os.path.join(ruta_datos, "intensidad.npy"), i_map.get())
    cp.save(os.path.join(ruta_datos, "stokes_q.npy"), q_map.get())
    cp.save(os.path.join(ruta_datos, "stokes_u.npy"), u_map.get())
    cp.save(os.path.join(ruta_datos, "rm_mapa_beam.npy"), rm_beam.get())
    cp.save(os.path.join(ruta_datos, "i_beam.npy"), i_beam.get())
    cp.save(os.path.join(ruta_datos, "q_beam.npy"), q_beam.get())
    cp.save(os.path.join(ruta_datos, "u_beam.npy"), u_beam.get())
    cp.save(os.path.join(ruta_datos, "intensidad_de_polaridad.npy"), p_beam.get())
    cp.save(os.path.join(ruta_datos, "fraccion_de_polarizacion.npy"), frac_pol.get())
    cp.save(os.path.join(ruta_datos, "despolarizacion.npy"), dp.get())

    del (
        bx, by, bz, r, ne, ne_rel, perfil_b, j_nu,
        i_map, q_map, u_map, rm_map,
        rm_beam, i_beam, q_beam, u_beam, p_beam, frac_pol, dp,
    )

    cp.get_default_memory_pool().free_all_blocks()

    generar_graficos_estudio(ruta_datos, ruta_graficos)


def estudio_parametrico():
    valores_n = [2.0, 3.0, 4.0]
    valores_b0 = [1.0, 10.0]

    for n in valores_n:
        for b0 in valores_b0:
            nombre_carpeta = f"n{int(n)}_b0_{int(b0)}"
            ruta = os.path.join("../results/estudio_parametrico", nombre_carpeta)
            print(f"Iniciando: n={n}, B0={b0}")
            ejecutar_simulacion(n, b0, ruta)


if __name__ == "__main__":
    estudio_parametrico()
