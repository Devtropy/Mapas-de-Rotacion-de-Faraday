import numpy as np


def densidad_beta(r, n0, rc, beta):
    return n0 * (1 + (r / rc) ** 2) ** (-3 * beta / 2)


def campo_magnetico(B, ne, n0, mu):
    return B * (ne / n0) ** mu


def emisividad(n_rel, B_perp, p, nu):
    return n_rel * (B_perp ** ((p + 1) / 2)) * nu ** (-(p - 1) / 2)


def angulo_intrinseco(Bx, By):
    return np.arctan2(By, Bx) + np.pi / 2
