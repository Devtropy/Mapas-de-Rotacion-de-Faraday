import numpy as np
from numerics.FFT import IFFT3D
from numerics.variables_aleatorias import Fase, Rayleigh
from config import Parametros
from grid import malla
from physics.physics import densidad_beta, campo_magnetico


def RM(ne, Bz, dz):
    return 812 * (ne * Bz).sum(axis=2) * dz


def turbulencia(Parametros):
    N = Parametros.N
    dx = Parametros.dx

    X, Y, Z = malla(N, dx)
    r = np.sqrt(X**2 + Y**2 + Z**2)

    kv = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    kx, ky, kz = np.meshgrid(kv, kv, kv, indexing="ij")
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)

    k_min = 2 * np.pi / Parametros.lambda_max
    k_max = 2 * np.pi / Parametros.lambda_min

    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_k = np.where(
            (k_mag > k_min) & (k_mag < k_max), k_mag ** (-(Parametros.n + 2) / 2), 0
        ).astype(np.float32)

    A_k = np.zeros((3, N, N, N), dtype=complex)
    for i in range(3):
        amp = Rayleigh(sigma_k)
        fase = Fase()
        A_k[i] = amp * (np.cos(fase) + 1j * np.sin(fase))

    k_vec = np.array([kx, ky, kz])
    Bk = 1j * np.cross(k_vec, A_k, axisa=0, axisb=0, axisc=0)

    del A_k, k_vec, k_mag, kx, ky, kz

    B_real = np.zeros((3, N, N, N), dtype=np.float32)
    for i in range(3):
        B_real[i] = IFFT3D(Bk[i])
    del Bk

    ne = densidad_beta(r, Parametros.n0, Parametros.rc, Parametros.beta)
    B_final = campo_magnetico(B_real, ne, Parametros.n0, Parametros.mu)

    B_mag = np.sqrt(np.sum(B_final**2, axis=0))
    B_centro = B_mag[N // 2, N // 2, N // 2]
    factor_norm = Parametros.B0 / B_centro

    return B_final * factor_norm, ne
