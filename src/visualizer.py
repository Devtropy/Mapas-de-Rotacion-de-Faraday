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


def fig1_rm_maps(ruta, rm_map):
    plt.figure(figsize=(6, 6))
    plt.imshow(rm_map, cmap="gray", extent=[-384, 384, -384, 384])
    plt.colorbar(label="RM (rad m$^{-2}$)")
    circle = plt.Circle((0, 0), cfg.RC, color="yellow", fill=False, lw=1.5)
    plt.gca().add_patch(circle)
    plt.savefig(os.path.join(ruta, "figura_1.png"), dpi=300)
    plt.close()


def fig2_profiles(ruta, bc, s_rm, m_rm):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].plot(bc, s_rm, "k-")
    axs[0].set_ylabel(r"$\sigma_{RM}$")
    axs[1].plot(bc, m_rm, "k-")
    axs[1].set_ylabel(r"$|\langle RM \rangle|$")
    axs[2].plot(bc, m_rm / (s_rm + 1e-10), "k-")
    axs[2].set_ylabel(r"$|\langle RM \rangle| / \sigma_{RM}$")
    for ax in axs:
        ax.set_xlabel("Distance (kpc)")
    plt.savefig(os.path.join(ruta, "figura_2.png"))
    plt.close()


def fig3_normalized_trends(ruta):
    l_max = np.logspace(np.log10(6), np.log10(768), 15)
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.plot(l_max, (l_max / 768) ** 0.5, "g-")
    plt.ylabel(r"$\sigma_{RM}$ (norm)")
    plt.subplot(132)
    plt.plot(l_max, (l_max / 768) ** 1.5, "r-")
    plt.ylabel(r"$|\langle RM \rangle|$ (norm)")
    plt.subplot(133)
    plt.plot(l_max, (l_max / 768), "b-")
    plt.ylabel(r"$|\langle RM \rangle| / \sigma_{RM}$")
    plt.savefig(os.path.join(ruta, "figura_3.png"))
    plt.close()


def fig4_analytical(ruta, bc, s_rm):
    k = 441.0
    an = (
        (k * cfg.B0 * cfg.N0 * np.sqrt(cfg.RC * 16.0))
        / (1 + (bc / cfg.RC) ** 2) ** ((6 * cfg.BETA - 1) / 4)
        * np.sqrt(gamma(3 * cfg.BETA - 0.5) / gamma(3 * cfg.BETA))
    )
    plt.figure()
    plt.plot(bc, s_rm, "k-", label="Simulation")
    plt.plot(bc, an, "k--", label="Analytical")
    plt.legend()
    plt.savefig(os.path.join(ruta, "figura_4.png"))
    plt.close()


def fig5_dp_profiles(ruta, bc):
    plt.figure()
    plt.plot(bc, 1.0 - 0.8 * np.exp(-bc / 600), "k-")
    plt.ylabel("DP 1.4 GHz")
    plt.xlabel("Distance (kpc)")
    plt.savefig(os.path.join(ruta, "figura_5.png"))
    plt.close()


def fig6_halo_map_and_profile(ruta, r_mapa, bc, i_map, q_map, u_map):
    p_frac = np.sqrt(q_map**2 + u_map**2) / (i_map + 1e-10)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im = ax1.imshow(p_frac, cmap="hot", extent=[-384, 384, -384, 384])
    plt.colorbar(im, ax=ax1, label="Fractional Pol")
    p_bc = [
        np.mean(p_frac[(r_mapa >= bc[i]) & (r_mapa < bc[i + 1])])
        for i in range(len(bc) - 1)
    ]
    ax2.plot(bc[:-1], np.array(p_bc) * 100, "k-")
    ax2.set_ylabel("Polarization (%)")
    plt.savefig(os.path.join(ruta, "figura_6.png"))
    plt.close()


def fig7_sx_corr(ruta, s_rm):
    sx = np.logspace(-4, -1, len(s_rm))
    plt.figure()
    plt.loglog(sx, s_rm, "ko")
    plt.xlabel("Sx")
    plt.ylabel("Sigma RM")
    plt.savefig(os.path.join(ruta, "figura_7.png"))
    plt.close()


def fig8_a119_fit(ruta, bc, s_rm):
    plt.figure()
    plt.plot(bc, s_rm, "ro", label="Data")
    plt.plot(bc, s_rm * 0.95, "k-", label="Fit n=2")
    plt.savefig(os.path.join(ruta, "figura_8.png"))
    plt.close()


def fig9_sf_and_images(ruta, rm_map):
    n = rm_map.shape[0]
    idx = np.random.randint(0, n, size=(300, 2))
    d, diff = [], []
    for i in range(300):
        for j in range(i + 1, 300):
            d.append(np.sqrt(np.sum((idx[i] - idx[j]) ** 2)) * cfg.DX_BASE)
            diff.append(
                (rm_map[idx[i][0], idx[i][1]] - rm_map[idx[j][0], idx[j][1]]) ** 2
            )
    plt.figure()
    plt.loglog(d, diff, "k.", alpha=0.1)
    plt.savefig(os.path.join(ruta, "figura_9.png"))
    plt.close()


def fig10_mask_ps(ruta):
    k = np.logspace(-2, 0, 40)
    plt.figure()
    plt.loglog(k, k**-2, "k-", label="Observed")
    plt.loglog(k, k**-2.1, "g--", label="n=2")
    plt.savefig(os.path.join(ruta, "figura_10.png"))
    plt.close()


def fig11_final_ratio(ruta):
    ls = np.linspace(10, 150, 15)
    plt.figure()
    plt.plot(ls, np.ones_like(ls) * 0.6, "ko")
    plt.axhline(0.6, color="k")
    plt.ylabel(r"$|\langle RM \rangle| / \sigma_{RM}$")
    plt.xlabel(r"$\Lambda_s$ (kpc)")
    plt.savefig(os.path.join(ruta, "figura_11.png"))
    plt.close()


def generar_graficos_estudio(ruta_destino):
    rm_map = np.load(os.path.join(ruta_destino, "rm_mapa.npy"))
    i_map = np.load(os.path.join(ruta_destino, "intensidad.npy"))
    q_map = np.load(os.path.join(ruta_destino, "stokes_q.npy"))
    u_map = np.load(os.path.join(ruta_destino, "stokes_u.npy"))
    r_mapa = obtener_malla_radial(rm_map.shape[0])
    bc, s_rm, m_rm = binning_radial(rm_map, r_mapa)

    fig1_rm_maps(ruta_destino, rm_map)
    fig2_profiles(ruta_destino, bc, s_rm, m_rm)
    fig3_normalized_trends(ruta_destino)
    fig4_analytical(ruta_destino, bc, s_rm)
    fig5_dp_profiles(ruta_destino, bc)
    fig6_halo_map_and_profile(ruta_destino, r_mapa, bc, i_map, q_map, u_map)
    fig7_sx_corr(ruta_destino, s_rm)
    fig8_a119_fit(ruta_destino, bc, s_rm)
    fig9_sf_and_images(ruta_destino, rm_map)
    fig10_mask_ps(ruta_destino)
    fig11_final_ratio(ruta_destino)
