import cupy as cp
from cupyx.scipy.fft import fftn, ifftn
import config as cfg


def generar_capa_campo(n, dx):
    k_vec = cp.fft.fftfreq(n, d=dx) * 2 * cp.pi
    kx, ky, kz = cp.meshgrid(k_vec, k_vec, k_vec, indexing="ij")
    k_mag = cp.sqrt(kx**2 + ky**2 + kz**2 + 1e-10)

    k_min, k_max = (2.0 * cp.pi) / cfg.LAMBDA_MAX, (2.0 * cp.pi) / cfg.LAMBDA_MIN
    sigma_k = cp.where(
        (k_mag > k_min) & (k_mag < k_max), k_mag ** (-(cfg.N_SPEC + 2.0) / 2.0), 0.0
    )

    A = [
        sigma_k
        * cp.random.rayleigh(1.0, (n, n, n))
        * cp.exp(1j * 2 * cp.pi * cp.random.random((n, n, n)))
        for _ in range(3)
    ]

    Bx_k = 1j * (ky * A[2] - kz * A[1])
    By_k = 1j * (kz * A[0] - kx * A[2])
    Bz_k = 1j * (kx * A[1] - ky * A[0])

    return ifftn(Bx_k).real, ifftn(By_k).real, ifftn(Bz_k).real


def obtener_malla_amr():
    bx_b, by_b, bz_b = generar_capa_campo(cfg.N_BASE, cfg.DX_BASE)
    bx_r, by_r, bz_r = generar_capa_campo(cfg.N_REFINADO, cfg.DX_REFINADO)

    eje = cp.linspace(-cfg.N_BASE / 2, cfg.N_BASE / 2, cfg.N_BASE) * cfg.DX_BASE
    xx, yy, zz = cp.meshgrid(eje, eje, eje, indexing="ij")
    r = cp.sqrt(xx**2 + yy**2 + zz**2)

    mascara = r < cfg.RADIO_REFINAMIENTO

    def refinar(base, refinado):
        temp = cp.repeat(cp.repeat(cp.repeat(refinado, 2, 0), 2, 1), 2, 2)
        return cp.where(mascara, temp[: cfg.N_BASE, : cfg.N_BASE, : cfg.N_BASE], base)

    return refinar(bx_b, bx_r), refinar(by_b, by_r), refinar(bz_b, bz_r), r
