import cupy as cp
from cupyx.scipy.fft import fftn, ifftn
import config_values as cfg


def generar_capa_campo(n, dx):
    k_vec = cp.fft.fftfreq(n, d=dx) * 2.0 * cp.pi
    kx, ky, kz = cp.meshgrid(k_vec, k_vec, k_vec, indexing="ij")
    k_mag = cp.sqrt(kx**2 + ky**2 + kz**2)

    k_min = cp.pi / cfg.LAMBDA_MAX_KPC
    k_max = cp.pi / cfg.LAMBDA_MIN_KPC

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

    ax=ifftn(A_k[0]).real
    ay=ifftn(A_k[1]).real
    az=ifftn(A_k[2]).real

    bx_k = 1j * (ky * A_k[2] - kz * A_k[1])
    by_k = 1j * (kz * A_k[0] - kx * A_k[2])
    bz_k = 1j * (kx * A_k[1] - ky * A_k[0])

    bx=ifftn(bx_k).real
    by = ifftn(by_k).real
    bz = ifftn(bz_k).real

    return bx,by,bz,ax,ay,az

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

    bx_b, by_b, bz_b, ax_b, ay_b, az_b = generar_capa_campo(cfg.N_BASE, cfg.DX_BASE_KPC)

    b_rms_b = cp.sqrt(cp.mean(bx_b**2 + by_b**2 + bz_b**2))

    ax_b, ay_b, az_b = ax_b / b_rms_b, ay_b / b_rms_b, az_b / b_rms_b

    bx_r, by_r, bz_r ,ax_r,ay_r,az_r= generar_capa_campo(cfg.N_REFINADO, cfg.DX_REFINADO_KPC)

    b_rms_r = cp.sqrt(cp.mean(bx_r**2 + by_r**2 + bz_r**2))

    ax_r, ay_r, az_r = ax_r / b_rms_r, ay_r / b_rms_r, az_r / b_rms_r

    eje = cp.linspace(-cfg.N_BASE / 2, cfg.N_BASE / 2, cfg.N_BASE) * cfg.DX_BASE_KPC
    xx, yy, zz = cp.meshgrid(eje, eje, eje, indexing="ij")
    r = cp.sqrt(xx**2 + yy**2 + zz**2)

    ancho_transicion = cfg.DX_BASE_KPC * 2.0

    peso_refinado = 1.0 / (1.0 + cp.exp((r - cfg.RADIO_REFINAMIENTO_KPC) / ancho_transicion))

    peso_base = 1.0 - peso_refinado

    def refinar_suave(base, refinado):
        zoom = cfg.N_REFINADO // cfg.N_BASE
        temp = cp.repeat(cp.repeat(cp.repeat(refinado, zoom, 0), zoom, 1), zoom, 2)

        start = (temp.shape[0] - cfg.N_BASE) // 2
        end = start + cfg.N_BASE
        cropped = temp[start:end, start:end, start:end]

        return base * peso_base + cropped * peso_refinado

    ax_final = refinar_suave(ax_b, ax_r)
    ay_final = refinar_suave(ay_b, ay_r)
    az_final = refinar_suave(az_b, az_r)

    bx_final = cp.gradient(az_final, cfg.DX_BASE_KPC, axis=1) - cp.gradient(
        ay_final, cfg.DX_BASE_KPC, axis=2
    )
    by_final = cp.gradient(ax_final, cfg.DX_BASE_KPC, axis=2) - cp.gradient(
        az_final, cfg.DX_BASE_KPC, axis=0
    )
    bz_final = cp.gradient(ay_final, cfg.DX_BASE_KPC, axis=0) - cp.gradient(
        ax_final, cfg.DX_BASE_KPC, axis=1
    )

    verificar_divergencia(bx_final, by_final, bz_final, cfg.DX_BASE_KPC, etiqueta="Malla AMR Suavizada")

    return bx_final, by_final, bz_final, r
