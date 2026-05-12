import cupy as cp
import os
import config as cfg
from grid import generar_capa_campo
from simulation.faraday import calcular_sincrotron
from simulation.Polaridad import calcular_mapas_polarizacion
from visualizer import generar_graficos_estudio


def ejecutar_simulacion(n_val, b0_val, ruta_destino):
    cfg.N_SPEC = n_val
    cfg.B0 = b0_val

    bx, by, bz = generar_capa_campo(cfg.N_BASE, cfg.DX_BASE)

    b_rms = cp.sqrt(cp.mean(bx**2 + by**2 + bz**2))
    bx *= cfg.B0 / b_rms
    by *= cfg.B0 / b_rms
    bz *= cfg.B0 / b_rms

    eje = cp.linspace(-cfg.N_BASE / 2, cfg.N_BASE / 2, cfg.N_BASE) * cfg.DX_BASE
    xx, yy, zz = cp.meshgrid(eje, eje, eje, indexing="ij")
    r = cp.sqrt(xx**2 + yy**2 + zz**2)

    ne = (cfg.N0 * (1.0 + (r / cfg.RC) ** 2) ** (-1.5 * cfg.BETA)).astype(cp.float32)
    ne_rel = ne * 0.01

    i_map, j_nu = calcular_sincrotron(bx, by, ne_rel)
    q_map, u_map = calcular_mapas_polarizacion(bx, by, bz, j_nu, ne)

    factor_unidades = 1000.0 if cfg.DX_BASE < 100 else 1.0
    rm_map = cp.sum(812.0 * ne * bz * (cfg.DX_BASE * factor_unidades), axis=2)

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
    valores_b0 = [1.0, 2.0]

    for n in valores_n:
        for b0 in valores_b0:
            nombre_carpeta = f"n{int(n)}_b0_{int(b0)}"
            ruta = os.path.join("../results/estudio_parametrico", nombre_carpeta)
            print(f"Iniciando: n={n}, B0={b0}")
            ejecutar_simulacion(n, b0, ruta)


if __name__ == "__main__":
    estudio_parametrico()
