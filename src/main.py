import cupy as cp
import os
import config_values as cfg
from grid import generar_capa_campo
from simulation.faraday import calcular_sincrotron
from simulation.Polaridad import calcular_mapas_polarizacion
from visualizer import generar_graficos_estudio


def ejecutar_simulacion(n_val, b0_val, ruta_destino):
    cfg.N_SPEC = n_val
    b0_mg = b0_val

    bx, by, bz = generar_capa_campo(cfg.N_BASE, cfg.DX_BASE_KPC)

    b_rms = cp.sqrt(cp.mean(bx**2 + by**2 + bz**2))
    bx *= b0_mg / b_rms
    by *= b0_mg / b_rms
    bz *= b0_mg / b_rms

    eje = cp.linspace(-cfg.N_BASE / 2, cfg.N_BASE / 2, cfg.N_BASE) * cfg.DX_BASE_KPC
    xx, yy, zz = cp.meshgrid(eje, eje, eje, indexing="ij")
    r = cp.sqrt(xx**2 + yy**2 + zz**2)

    ne = (cfg.N0_VAL * (1.0 + (r / cfg.RC_KPC) ** 2) ** (-1.5 * cfg.BETA)).astype(
        cp.float32
    )
    ne_rel = ne * 0.01

    i_map, j_nu = calcular_sincrotron(bx, by, ne_rel)
    q_map, u_map = calcular_mapas_polarizacion(bx, by, bz, j_nu, ne)

    rm_map = cp.sum(0.812 * ne * bz * cfg.DX_BASE_PC, axis=2)

    os.makedirs(ruta_destino, exist_ok=True)
    cp.save(os.path.join(ruta_destino, "rm_mapa.npy"), rm_map.get())
    cp.save(os.path.join(ruta_destino, "intensidad.npy"), i_map.get())
    cp.save(os.path.join(ruta_destino, "stokes_q.npy"), q_map.get())
    cp.save(os.path.join(ruta_destino, "stokes_u.npy"), u_map.get())

    del bx, by, bz, r, ne, ne_rel, j_nu, i_map, q_map, u_map, rm_map
    cp.get_default_memory_pool().free_all_blocks()

    generar_graficos_estudio(ruta_destino)


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
