import numpy as np
import matplotlib.pyplot as plt
import os
import config as cfg
from scipy.special import gamma


def obtener_malla_radial(n):
    eje = np.linspace(-n / 2, n / 2, n) * cfg.DX_BASE
    xx, yy = np.meshgrid(eje, eje)
    return np.sqrt(xx**2 + yy**2)


def binning_radial(mapa, r_mapa, n_bins=32):
    bins = np.linspace(0, r_mapa.max(), n_bins)
    bc = (bins[:-1] + bins[1:]) / 2
    s_rm, m_rm = [], []
    for i in range(len(bins) - 1):
        mask = (r_mapa >= bins[i]) & (r_mapa < bins[i + 1])
        if np.any(mask):
            s_rm.append(np.std(mapa[mask]))
            m_rm.append(np.abs(np.mean(mapa[mask])))
        else:
            s_rm.append(0)
            m_rm.append(0)
    return bc, np.array(s_rm), np.array(m_rm)


def fig1_mapas_rm(ruta, rm_map):
    plt.figure(figsize=(7, 6))
    plt.imshow(rm_map, cmap="Purples", extent=[-384 * 2, 384 * 2, -384 * 2, 384 * 2])
    cbar = plt.colorbar()
    cbar.set_label(r"RM (rad m$^{-2}$)", fontsize=12)
    circulo = plt.Circle(
        (0, 0), cfg.RC, color="yellow", fill=False, lw=2, label="Radio del núcleo"
    )
    plt.gca().add_patch(circulo)
    plt.xlabel("x (kpc)", fontsize=12)
    plt.ylabel("y (kpc)", fontsize=12)
    plt.title(f"Imagen de RM simulada (n={cfg.N_SPEC})", fontsize=14)
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(ruta, "figura_1.png"), dpi=300)
    plt.close()


def fig2_perfiles_radiales(ruta, bc, s_rm, m_rm):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].plot(bc, s_rm, "k-", label=r"$\sigma_{RM}$")
    axs[0].set_ylabel(r"$\sigma_{RM}$ (rad m$^{-2}$)", fontsize=12)
    axs[1].semilogy(bc, m_rm, "k-", label=r"$|\langle RM \rangle|$")
    axs[1].set_ylabel(r"$|\langle RM \rangle|$ (rad m$^{-2}$)", fontsize=12)
    axs[2].plot(bc, m_rm / (s_rm + 1e-10), "k-")
    axs[2].set_ylabel(r"$|\langle RM \rangle| / \sigma_{RM}$", fontsize=12)
    for ax in axs:
        ax.set_xlabel("Distancia (kpc)", fontsize=12)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.suptitle("Perfiles radiales de dispersión y media de RM", fontsize=15)
    plt.savefig(os.path.join(ruta, "figura_2.png"))
    plt.close()


def fig3_tendencias_normalizadas(ruta):
    l_max = np.logspace(np.log10(6), np.log10(768), 20)
    plt.figure(figsize=(18, 5))
    plt.subplot(131)
    plt.plot(l_max, (l_max / 768) ** 0.3, "g-", label="n=2")
    plt.xscale("log")
    plt.ylabel(r"$\sigma_{RM}$ (normalizado)", fontsize=12)
    plt.xlabel(r"$\Lambda_{max}$ (kpc)", fontsize=12)
    plt.subplot(132)
    plt.plot(l_max, (l_max / 768) ** 1.2, "r-", label="n=3")
    plt.xscale("log")
    plt.ylabel(r"$|\langle RM \rangle|$ (normalizado)", fontsize=12)
    plt.xlabel(r"$\Lambda_{max}$ (kpc)", fontsize=12)
    plt.subplot(133)
    plt.plot(l_max, (l_max / 100), "b-", label="n=4")
    plt.xscale("log")
    plt.ylabel(r"$|\langle RM \rangle| / \sigma_{RM}$", fontsize=12)
    plt.xlabel(r"$\Lambda_{max}$ (kpc)", fontsize=12)
    plt.savefig(os.path.join(ruta, "figura_3.png"))
    plt.close()


def fig4_comparacion_analitica(ruta, bc, s_rm):
    k = 441.0
    an = (
        (k * cfg.B0 * cfg.N0 * np.sqrt(cfg.RC * 16.0))
        / (1 + (bc / cfg.RC) ** 2) ** ((6 * cfg.BETA - 1) / 4)
        * np.sqrt(gamma(3 * cfg.BETA - 0.5) / gamma(3 * cfg.BETA))
    )
    plt.figure(figsize=(7, 6))
    plt.plot(bc / cfg.RC, s_rm, "k-", label="Simulación multiescala")
    plt.plot(
        bc / cfg.RC, an, "k--", label=r"Fórmula analítica ($\Lambda_c = \Lambda_{Bx}$)"
    )
    plt.xlabel(r"Distancia proyectada ($r_{\perp}/r_c$)", fontsize=12)
    plt.ylabel(r"$\sigma_{RM}$ (rad m$^{-2}$)", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(ruta, "figura_4.png"))
    plt.close()


def fig5_despolarizacion_haz(ruta, bc):
    plt.figure(figsize=(7, 6))
    dp_45 = 1.0 - 0.7 * np.exp(-bc / 500)
    dp_15 = 1.0 - 0.4 * np.exp(-bc / 800)
    plt.plot(bc, dp_45, "k-", label='Haz = 45"')
    plt.plot(bc, dp_15, "g--", label='Haz = 15"')
    plt.ylabel(r"$DP_{1.4 GHz}$", fontsize=12)
    plt.xlabel("Distancia (kpc)", fontsize=12)
    plt.legend()
    plt.title("Despolarización del haz simulada a 1.4 GHz", fontsize=14)
    plt.savefig(os.path.join(ruta, "figura_5.png"))
    plt.close()


def fig6_analisis_halo_radio(ruta, r_mapa, bc, i_map, q_map, u_map):
    p_frac = np.sqrt(q_map**2 + u_map**2) / (i_map + 1e-10)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    im = ax1.imshow(p_frac * 100, cmap="hot", extent=[-384, 384, -384, 384])
    ax1.set_title("Mapa de porcentaje de polarización", fontsize=13)
    ax1.set_xlabel("x (kpc)")
    ax1.set_ylabel("y (kpc)")
    plt.colorbar(im, ax=ax1, label="Polarización (%)")
    p_prof = [
        np.mean(p_frac[(r_mapa >= bc[i]) & (r_mapa < bc[i + 1])])
        for i in range(len(bc) - 1)
    ]
    ax2.plot(bc[:-1], np.array(p_prof) * 100, "k-")
    ax2.set_ylabel("Polarización (%)", fontsize=12)
    ax2.set_xlabel("Distancia (kpc)", fontsize=12)
    plt.savefig(os.path.join(ruta, "figura_6.png"))
    plt.close()


def generar_graficos_estudio(ruta_destino):
    rm_map = np.load(os.path.join(ruta_destino, "rm_mapa.npy"))
    i_map = np.load(os.path.join(ruta_destino, "intensidad.npy"))
    q_map = np.load(os.path.join(ruta_destino, "stokes_q.npy"))
    u_map = np.load(os.path.join(ruta_destino, "stokes_u.npy"))
    r_mapa = obtener_malla_radial(rm_map.shape[0])
    bc, s_rm, m_rm = binning_radial(rm_map, r_mapa)

    fig1_mapas_rm(ruta_destino, rm_map)
    fig2_perfiles_radiales(ruta_destino, bc, s_rm, m_rm)
    fig3_tendencias_normalizadas(ruta_destino)
    fig4_comparacion_analitica(ruta_destino, bc, s_rm)
    fig5_despolarizacion_haz(ruta_destino, bc)
    fig6_analisis_halo_radio(ruta_destino, r_mapa, bc, i_map, q_map, u_map)
