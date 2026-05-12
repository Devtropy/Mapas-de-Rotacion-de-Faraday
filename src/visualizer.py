import numpy as np
import matplotlib.pyplot as plt
import os


def generar_graficos_individuales():
    ruta_base = "../results/"
    ruta_salida = os.path.join(ruta_base, "graficos_finales")

    if not os.path.exists(ruta_salida):
        os.makedirs(ruta_salida)

    rm = np.load(os.path.join(ruta_base, "rm_mapa.npy"))
    i_tot = np.load(os.path.join(ruta_base, "intensidad.npy"))
    q = np.load(os.path.join(ruta_base, "stokes_q.npy"))
    u = np.load(os.path.join(ruta_base, "stokes_u.npy"))

    p_frac = np.sqrt(q**2 + u**2) / (i_tot + 1e-15)
    psi = 0.5 * np.arctan2(u, q)
    x_dim = np.linspace(-768, 768, rm.shape[0])
    x, y = np.meshgrid(x_dim, x_dim)

    def guardar_mapa(data, nombre, titulo, cmap, label_cb, vmin=None, vmax=None):
        plt.figure(figsize=(8, 7))
        im = plt.imshow(
            data, cmap=cmap, extent=[-768, 768, -768, 768], vmin=vmin, vmax=vmax
        )
        plt.title(titulo)
        plt.xlabel("kpc")
        plt.ylabel("kpc")
        plt.colorbar(im, label=label_cb)
        plt.savefig(
            os.path.join(ruta_salida, f"{nombre}.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

    guardar_mapa(rm, "01_mapa_rm", "Medida de Rotación (RM)", "seismic", r"rad/m$^2$")
    guardar_mapa(
        np.log10(i_tot + 1e-15),
        "02_intensidad_log",
        "Log10 Intensidad Sincrotrón",
        "magma",
        "Log Intensity",
    )
    guardar_mapa(
        p_frac,
        "03_grado_polarizacion",
        "Grado de Polarización Fraccional",
        "viridis",
        "P",
        vmin=0,
        vmax=0.7,
    )

    plt.figure(figsize=(8, 8))
    paso = 4
    plt.quiver(
        x[::paso, ::paso],
        y[::paso, ::paso],
        p_frac[::paso, ::paso] * np.cos(psi[::paso, ::paso]),
        p_frac[::paso, ::paso] * np.sin(psi[::paso, ::paso]),
        pivot="middle",
        color="black",
        headwidth=0,
        scale=10,
    )
    plt.title("Vectores de Ángulo de Posición de Polarización")
    plt.xlabel("kpc")
    plt.ylabel("kpc")
    plt.savefig(
        os.path.join(ruta_salida, "04_vectores_polarizacion.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(rm.flatten(), bins=50, color="skyblue", edgecolor="black")
    plt.title("Distribución Estadística de RM")
    plt.xlabel(r"RM (rad/m$^2$)")
    plt.ylabel("Frecuencia")
    plt.savefig(
        os.path.join(ruta_salida, "05_histograma_rm.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    r_map = np.sqrt(x**2 + y**2).flatten()
    rm_flat = np.abs(rm).flatten()
    indices = np.argsort(r_map)
    r_sorted, rm_sorted = r_map[indices], rm_flat[indices]
    bins_r = np.linspace(0, 768, 20)
    bin_means = [
        rm_sorted[(r_sorted >= bins_r[j]) & (r_sorted < bins_r[j + 1])].mean()
        for j in range(len(bins_r) - 1)
    ]

    plt.figure(figsize=(8, 6))
    plt.plot(bins_r[:-1], bin_means, "ro-", linewidth=2)
    plt.title("Perfil Radial de la Dispersión de RM")
    plt.xlabel("Radio (kpc)")
    plt.ylabel(r"$\langle |RM| \rangle$")
    plt.grid(True, linestyle="--")
    plt.savefig(
        os.path.join(ruta_salida, "06_perfil_radial_rm.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    generar_graficos_individuales()
