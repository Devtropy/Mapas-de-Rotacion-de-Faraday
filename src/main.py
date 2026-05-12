import cupy as cp
import os
import config as cfg
from grid import obtener_malla_amr
from simulation.faraday import calcular_sincrotron
from simulation.Polaridad import calcular_mapas_polarizacion


def ejecutar():
    bx, by, bz, r = obtener_malla_amr()

    ne = cfg.N0 * (1.0 + (r / cfg.RC) ** 2) ** (-1.5 * cfg.BETA)
    ne_rel = ne * 0.01

    i_map, j_nu = calcular_sincrotron(bx, by, ne_rel)
    q_map, u_map = calcular_mapas_polarizacion(bx, by, bz, j_nu, ne)

    rm_map = cp.sum(812.0 * ne * bz * cfg.DX_BASE, axis=2)

    cp.save("../results/rm_mapa.npy", rm_map)
    cp.save("../results/intensidad.npy", i_map)
    cp.save("../results/stokes_q.npy", q_map)
    cp.save("../results/stokes_u.npy", u_map)


if __name__ == "__main__":
    if not os.path.exists("../results"):
        os.makedirs("../results")
    ejecutar()
