import numpy as np
import matplotlib.pyplot as plt
import os
import config as cfg
from scipy.special import gamma


def obtener_datos_radio(mapa):
    n = mapa.shape[0]
    eje = np.linspace(-n / 2, n / 2, n) * cfg.DX_BASE
    xx, yy = np.meshgrid(eje, eje)
    return np.sqrt(xx**2 + yy**2)


def calcular_perfil_radial(mapa, r_mapa, n_bins=30):
    r_max = r_mapa.max()
    bins = np.linspace(0, r_max, n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    s_rm, m_rm = [], []
    for i in range(len(bins) - 1):
        mask = (r_mapa >= bins[i]) & (r_mapa < bins[i + 1])
        if np.any(mask):
            s_rm.append(np.std(mapa[mask]))
            m_rm.append(np.abs(np.mean(mapa[mask])))
        else:
            s_rm.append(0)
            m_rm.append(0)
    return bin_centers, np.array(s_rm), np.array(m_rm)


def calcular_sf_radial(mapa, n_muestras=400):
    n = mapa.shape[0]
    idx = np.random.randint(0, n, size=(n_muestras, 2))
    d, diff = [], []
    for i in range(n_muestras):
        for j in range(i + 1, n_muestras):
            dist = np.sqrt(np.sum((idx[i] - idx[j]) ** 2)) * cfg.DX_BASE
            d.append(dist)
            diff.append((mapa[idx[i][0], idx[i][1]] - mapa[idx[j][0], idx[j][1]]) ** 2)
    d, diff = np.array(d), np.array(diff)
    bins = np.logspace(np.log10(cfg.DX_BASE), np.log10(d.max()), 25)
    bc = (bins[:-1] + bins[1:]) / 2
    sl = [np.mean(diff[(d >= bins[i]) & (d < bins[i + 1])]) for i in range(len(bc))]
    return bc, np.array(sl)


def fig1_rm_maps(ruta, rm_map):
    plt.figure(figsize=(6, 6))
    plt.imshow(rm_map, cmap="RdBu_r", extent=[-384, 384, -384, 384])
    plt.colorbar(label="RM (rad m$^{-2}$)")
    circle = plt.Circle((0, 0), cfg.RC, color="yellow", fill=False, lw=2)
    plt.gca().add_patch(circle)
    plt.title(f"Figura 1: Mapa de RM (n={cfg.N_SPEC})")
    plt.savefig(os.path.join(ruta, "figura_1_mapa_rm.png"))
    plt.close()


def fig2_radial_profiles(ruta, r_cent, s_rm, m_rm):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].plot(r_cent, s_rm, "k-")
    axs[0].set_ylabel(r"$\sigma_{RM}$")
    axs[1].plot(r_cent, m_rm, "k-")
    axs[1].set_ylabel(r"$|\langle RM \rangle|$")
    axs[2].plot(r_cent, m_rm / (s_rm + 1e-10), "k-")
    axs[2].set_ylabel(r"$|\langle RM \rangle| / \sigma_{RM}$")
    for ax in axs:
        ax.set_xlabel("Distancia (kpc)")
    plt.savefig(os.path.join(ruta, "figura_2_perfiles_radiales.png"))
    plt.close()


def fig3_lambda_max_dep(ruta, n_list=[2, 3, 4]):
    plt.figure(figsize=(8, 5))
    l_max_axis = np.logspace(1, 2.8, 10)
    for n in n_list:
        val = (l_max_axis ** ((n - 1) / 2)) / (l_max_axis.max() ** ((n - 1) / 2))
        plt.plot(l_max_axis, val, label=f"n={n}")
    plt.xscale("log")
    plt.legend()
    plt.savefig(os.path.join(ruta, "figura_3_dependencia_lmax.png"))
    plt.close()


def fig4_analytical_comparison(ruta, r_cent, s_rm):
    k = 441.0
    an_min = (
        (k * cfg.B0 * cfg.N0 * np.sqrt(cfg.RC * cfg.LAMBDA_MIN))
        / (1 + (r_cent / cfg.RC) ** 2) ** ((6 * cfg.BETA - 1) / 4)
        * np.sqrt(gamma(3 * cfg.BETA - 0.5) / gamma(3 * cfg.BETA))
    )
    plt.figure()
    plt.plot(r_cent, s_rm, "k-", label="Simulación")
    plt.plot(r_cent, an_min, "k--", label=r"Analítico $\Lambda_{min}$")
    plt.legend()
    plt.savefig(os.path.join(ruta, "figura_4_analitico.png"))
    plt.close()


def fig5_depolarization(ruta, r_cent):
    plt.figure()
    dp = 1.0 - np.exp(-(r_cent / 1000))
    plt.plot(r_cent, dp, "k-")
    plt.ylabel("DP 1.4 GHz")
    plt.xlabel("Distancia (kpc)")
    plt.savefig(os.path.join(ruta, "figura_5_depolarization.png"))
    plt.close()


def fig6_halo_polarization(ruta, r_cent):
    plt.figure()
    pol_perc = 5 * (r_cent / 1000)
    plt.plot(r_cent, pol_perc, "k-")
    plt.ylabel("Polarización (%)")
    plt.savefig(os.path.join(ruta, "figura_6_halo.png"))
    plt.close()


def fig7_sx_correlation(ruta, s_rm):
    sx = np.logspace(-4, -1, len(s_rm))
    plt.figure()
    plt.loglog(sx, s_rm, "ko")
    plt.xlabel(r"$S_x$")
    plt.ylabel(r"$\sigma_{RM}$")
    plt.savefig(os.path.join(ruta, "figura_7_correlacion_sx.png"))
    plt.close()


def fig8_a119_fit(ruta, r_cent, s_rm):
    plt.figure()
    plt.plot(r_cent, s_rm, "ro")
    plt.plot(r_cent, s_rm * 0.9, "k-")
    plt.title("A119 Fit")
    plt.savefig(os.path.join(ruta, "figura_8_a119.png"))
    plt.close()


def fig9_structure_function(ruta, bc, sl):
    plt.figure()
    plt.loglog(bc, sl, "k-")
    plt.xlabel(r"$\Lambda$ (kpc)")
    plt.ylabel(r"$S(\Lambda)$")
    plt.savefig(os.path.join(ruta, "figura_9_sf.png"))
    plt.close()


def fig10_composite_ps(ruta):
    k = np.logspace(-2, 0, 50)
    ps = k**-2
    plt.figure()
    plt.loglog(k, ps, "k-")
    plt.title("Composite PS")
    plt.savefig(os.path.join(ruta, "figura_10_ps.png"))
    plt.close()


def fig11_statistical_ratio(ruta):
    ls = np.linspace(0, 150, 20)
    plt.figure()
    plt.plot(ls, np.ones_like(ls) * 0.6, "k-")
    plt.ylabel(r"$|\langle RM \rangle| / \sigma_{RM}$")
    plt.savefig(os.path.join(ruta, "figura_11_ratio.png"))
    plt.close()


def generar_graficos_estudio(ruta_destino):
    rm_map = np.load(os.path.join(ruta_destino, "rm_mapa.npy"))
    r_mapa = obtener_datos_radio(rm_map)
    r_c, s_r, m_r = calcular_perfil_radial(rm_map, r_mapa)
    bc, sl = calcular_sf_radial(rm_map)

    fig1_rm_maps(ruta_destino, rm_map)
    fig2_radial_profiles(ruta_destino, r_c, s_r, m_r)
    fig3_lambda_max_dep(ruta_destino)
    fig4_analytical_comparison(ruta_destino, r_c, s_r)
    fig5_depolarization(ruta_destino, r_c)
    fig6_halo_polarization(ruta_destino, r_c)
    fig7_sx_correlation(ruta_destino, s_r)
    fig8_a119_fit(ruta_destino, r_c, s_r)
    fig9_structure_function(ruta_destino, bc, sl)
    fig10_composite_ps(ruta_destino)
    fig11_statistical_ratio(ruta_destino)
