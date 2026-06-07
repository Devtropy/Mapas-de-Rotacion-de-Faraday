import cupy as cp
import config as cfg


def calcular_mapas_polarizacion(Bx, By, Bz, j_nu, ne):
    fp = (cfg.P_SPEC + 1) / (cfg.P_SPEC + 7 / 3)
    psi_0 = cp.arctan2(Bx, -By)

    rm_acumulada = cp.flip(
        cp.cumsum(cp.flip(812.0 * ne * Bz * cfg.DX_BASE, axis=2), axis=2), axis=2
    )
    psi_obs = psi_0 + rm_acumulada * (cfg.LAMBDA_ONDA**2)

    Q_tot = cp.sum((fp * j_nu) * cp.cos(2 * psi_obs) * cfg.DX_BASE, axis=2)
    U_tot = cp.sum((fp * j_nu) * cp.sin(2 * psi_obs) * cfg.DX_BASE, axis=2)

    return Q_tot, U_tot
