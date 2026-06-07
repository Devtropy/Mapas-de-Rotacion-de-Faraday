import cupy as cp
from cupyx.scipy.fft import fftn, ifftn
import config as cfg


def generar_capa_campo(n, dx):
    k_vec = cp.fft.fftfreq(n, d=dx) * 2.0 * cp.pi
    kx, ky, kz = cp.meshgrid(k_vec, k_vec, k_vec, indexing="ij")
    k_mag = cp.sqrt(kx**2 + ky**2 + kz**2)

    k_min = cp.pi / cfg.LAMBDA_MAX
    k_max = cp.pi / cfg.LAMBDA_MIN

    zeta = cfg.N_SPEC + 2.0
    k_safe = cp.where(k_mag == 0, 1e-20, k_mag)
    sigma_k = cp.where(
        (k_mag >= k_min) & (k_mag <= k_max), k_safe ** (-zeta / 2.0), 0.0
    )

    A_k = []
    for _ in range(3):
        fase = 2.0 * cp.pi * cp.random.random((n, n, n))
        amplitud = cp.random.rayleigh(1.0, (n, n, n))
        A_k.append(sigma_k * amplitud * cp.exp(1j * fase))

    bx_k = 1j * (ky * A_k[2] - kz * A_k[1])
    by_k = 1j * (kz * A_k[0] - kx * A_k[2])
    bz_k = 1j * (kx * A_k[1] - ky * A_k[0])

    return ifftn(bx_k).real, ifftn(by_k).real, ifftn(bz_k).real

def verificar_divergencia(bx, by, bz, dx, etiqueta="Campo"):

    dbx_dx = cp.gradient(bx, dx, axis=0)
    dby_dy = cp.gradient(by, dx, axis=1)
    dbz_dz = cp.gradient(bz, dx, axis=2)
    
    div_B = dbx_dx + dby_dy + dbz_dz
    
    div_mean = cp.mean(cp.abs(div_B))
    div_max = cp.max(cp.abs(div_B))
    
    print(f"[Física - {etiqueta}] Divergencia Media: {div_mean:.4e} | Máxima: {div_max:.4e}")
    return div_B

def obtener_malla_amr():
    bx_b, by_b, bz_b = generar_capa_campo(cfg.N_BASE, cfg.DX_BASE)
    b_rms_b = cp.sqrt(cp.mean(bx_b**2 + by_b**2 + bz_b**2))
    bx_b, by_b, bz_b = bx_b / b_rms_b, by_b / b_rms_b, bz_b / b_rms_b

    bx_r, by_r, bz_r = generar_capa_campo(cfg.N_REFINADO, cfg.DX_REFINADO)
    b_rms_r = cp.sqrt(cp.mean(bx_r**2 + by_r**2 + bz_r**2))
    bx_r, by_r, bz_r = bx_r / b_rms_r, by_r / b_rms_r, bz_r / b_rms_r

    eje = cp.linspace(-cfg.N_BASE / 2, cfg.N_BASE / 2, cfg.N_BASE) * cfg.DX_BASE
    xx, yy, zz = cp.meshgrid(eje, eje, eje, indexing="ij")
    r = cp.sqrt(xx**2 + yy**2 + zz**2)

    ancho_transicion = cfg.DX_BASE * 2.0

    peso_refinado = 1.0 / (1.0 + cp.exp((r - cfg.RADIO_REFINAMIENTO) / ancho_transicion))
    peso_base = 1.0 - peso_refinado

    def refinar_suave(base, refinado):
        zoom = cfg.N_REFINADO // cfg.N_BASE
        temp = cp.repeat(cp.repeat(cp.repeat(refinado, zoom, 0), zoom, 1), zoom, 2)

        start = (temp.shape[0] - cfg.N_BASE) // 2
        end = start + cfg.N_BASE
        cropped = temp[start:end, start:end, start:end]

        return base * peso_base + cropped * peso_refinado
    bx_final = refinar_suave(bx_b, bx_r)
    by_final = refinar_suave(by_b, by_r)
    bz_final = refinar_suave(bz_b, bz_r)
    
    verificar_divergencia(bx_final, by_final, bz_final, cfg.DX_BASE, etiqueta="Malla AMR Suavizada")

    return bx_final, by_final, bz_final, r
