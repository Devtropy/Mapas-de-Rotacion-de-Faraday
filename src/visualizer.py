import numpy as np
import matplotlib.pyplot as plt
import os
import config as cfg


def generar_graficos_estudio(ruta_data):
    # Crear carpetas de destino basadas en las screenshots
    dir_faraday = os.path.join(ruta_data, "Faraday")
    dir_polarizacion = os.path.join(ruta_data, "Polarizacion")

    for d in [dir_faraday, dir_polarizacion]:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    # Carga de datos con manejo de errores
    try:
        rm = np.load(os.path.join(ruta_data, "rm_mapa.npy"))
        i_tot = np.load(os.path.join(ruta_data, "intensidad.npy"))
        q = np.load(os.path.join(ruta_data, "stokes_q.npy"))
        u = np.load(os.path.join(ruta_data, "stokes_u.npy"))
    except FileNotFoundError as e:
        print(f"Error: No se encontró un archivo .npy en {ruta_data}")
        return

    p_frac = np.sqrt(q**2 + u**2) / (i_tot + 1e-15)
    psi = 0.5 * np.arctan2(u, q)
    x_dim = np.linspace(-768, 768, rm.shape[0])
    x, y = np.meshgrid(x_dim, x_dim)

    # --- Carpeta Faraday ---
    plt.figure(figsize=(8, 7))
    plt.imshow(rm, cmap="seismic", extent=[-768, 768, -768, 768])
    plt.title("Medida de Rotación (RM)")
    plt.colorbar(label="rad/m^2")
    plt.gca().add_patch(plt.Circle((0, 0), cfg.RC, color="yellow", fill=False, ls="--"))
    plt.savefig(os.path.join(dir_faraday, "01_mapa_rm.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(rm.flatten(), bins=50, color="skyblue", edgecolor="black", density=True)
    plt.title("Estadística de RM")
    plt.savefig(os.path.join(dir_faraday, "02_histograma.png"), dpi=300)
    plt.close()

    # --- Carpeta Polarizacion ---
    plt.figure(figsize=(8, 7))
    plt.imshow(np.log10(i_tot + 1e-15), cmap="inferno", extent=[-768, 768, -768, 768])
    plt.title("Intensidad Sincrotrón")
    plt.colorbar(label="Log10(I)")
    plt.savefig(os.path.join(dir_polarizacion, "01_intensidad.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.imshow(
        np.log10(i_tot + 1e-15), cmap="gray_r", extent=[-768, 768, -768, 768], alpha=0.5
    )
    skip = max(1, rm.shape[0] // 20)
    plt.quiver(
        x[::skip, ::skip],
        y[::skip, ::skip],
        p_frac[::skip, ::skip] * np.cos(psi[::skip, ::skip]),
        p_frac[::skip, ::skip] * np.sin(psi[::skip, ::skip]),
        pivot="middle",
        color="red",
        headwidth=0,
        scale=15,
    )
    plt.title("Vectores de Ángulo de Posición")
    plt.savefig(os.path.join(dir_polarizacion, "02_vectores.png"), dpi=300)
    plt.close()

    print(f"Gráficos generados con éxito en: {ruta_data}")


if __name__ == "__main__":
    # Si quieres ejecutarlo manualmente para una carpeta específica:
    generar_graficos_estudio("../results/estudio_parametrico/n3_b0_1")
