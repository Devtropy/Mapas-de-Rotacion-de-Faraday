import cupy as cp
import config_values as cfg


def calcular_sincrotron(Bx, By, ne_rel):
    B_perp = cp.sqrt(Bx**2 + By**2)
    j_nu = (
        ne_rel
        * cp.power(B_perp, (cfg.P_SPEC + 1) / 2)
        * cp.power(cfg.NU_HZ, -(cfg.P_SPEC - 1) / 2)
    )
    mapa_intensidad = cp.sum(j_nu * cfg.DX_BASE_KPC, axis=2)
    return mapa_intensidad, j_nu
