import numpy as np
import matplotlib.pyplot as plt
import os
import config as cfg


def calcular_perfil_radial(mapa, n_bins=20):
    n = mapa.shape[0]
    eje = np.linspace(-n / 2, n / 2, n) * cfg.DX_BASE
    xx, yy = np.meshgrid(eje, eje)
    r_mapa = np.sqrt(xx**2 + yy**2)

    r_max = n * cfg.DX_BASE / 2
    bins = np.linspace(0, r_max, n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    sigma_perfil = []
    mean_perfil = []

    for i in range(len(bins) - 1):
        mascara = (r_mapa >= bins[i]) & (r_mapa < bins[i + 1])
        if np.any(mascara):
            datos_bin = mapa[mascara]
            sigma_perfil.append(np.std(datos_bin))
            mean_perfil.append(np.abs(np.mean(datos_bin)))
        else:
            sigma_perfil.append(0)
            mean_perfil.append(0)

    return bin_centers, np.array(sigma_perfil), np.array(mean_perfil)


def calcular_funcion_estructura(mapa, n_muestras=500):
    n = mapa.shape[0]
    dists = []
    diff_sq = []

    indices = np.random.randint(0, n, size=(n_muestras, 2))

    for i in range(n_muestras):
        for j in range(i + 1, n_muestras):
            y1, x1 = indices[i]
            y2, x2 = indices[j]

            d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) * cfg.DX_BASE
            diff = (mapa[y1, x1] - mapa[y2, x2]) ** 2

            dists.append(d)
            diff_sq.append(diff)

    dists = np.array(dists)
    diff_sq = np.array(diff_sq)

    bins = np.logspace(np.log10(cfg.DX_BASE), np.log10(n * cfg.DX_BASE / 2), 20)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    s_l = []

    for i in range(len(bins) - 1):
        mascara = (dists >= bins[i]) & (dists < bins[i + 1])
        if np.any(mascara):
            s_l.append(np.mean(diff_sq[mascara]))
        else:
            s_l.append(np.nan)

    return bin_centers, np.array(s_l)


def formula_analitica_sigma(r_perp, b0, lambda_c):
    from scipy.special import gamma

    k = 441.0
    term1 = k * b0 * cfg.N0 * np.sqrt(cfg.RC * lambda_c)
    term2 = (1 + (r_perp / cfg.RC) ** 2) ** ((6 * cfg.BETA - 1) / 4)
    term3 = np.sqrt(gamma(3 * cfg.BETA - 0.5) / gamma(3 * cfg.BETA))
    return (term1 / term2) * term3


def generar_graficos_estudio(ruta_datos):
    rm_map = np.load(os.path.join(ruta_datos, "rm_mapa.npy"))

    r_cent, sigma_rm, mean_rm = calcular_perfil_radial(rm_map)
    dist_sf, s_l = calcular_funcion_estructura(rm_map)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    axs[0].plot(r_cent, sigma_rm, "ro-", label="Simulación $\sigma_{RM}$")
    analitico = formula_analitica_sigma(r_cent, cfg.B0, cfg.LAMBDA_MIN)
    axs[0].plot(
        r_cent, analitico, "k--", label=r"Analítico ($\Lambda_c = \Lambda_{min}$)"
    )
    axs[0].set_xlabel("Distancia al centro (kpc)")
    axs[0].set_ylabel(r"$\sigma_{RM}$ (rad m$^{-2}$)")
    axs[0].legend()
    axs[0].set_title("Perfil Radial de Fluctuaciones")

    ratio = mean_rm / (sigma_rm + 1e-10)
    axs[1].plot(r_cent, ratio, "go-")
    axs[1].axhline(y=np.mean(ratio), color="r", linestyle="--")
    axs[1].set_xlabel("Distancia al centro (kpc)")
    axs[1].set_ylabel(r"$|\langle RM \rangle| / \sigma_{RM}$")
    axs[1].set_title("Ratio de Escalas (Murgia Fig. 2)")

    axs[2].loglog(dist_sf, s_l, "bo-")
    axs[2].set_xlabel(r"Separación $\Lambda$ (kpc)")
    axs[2].set_ylabel(r"$S(\Lambda)$ (rad$^2$ m$^{-4}$)")
    axs[2].set_title("Función de Estructura (Murgia Fig. 9)")

    plt.tight_layout()
    plt.savefig(os.path.join(ruta_datos, "analisis_faraday.png"), dpi=300)
    plt.close()
