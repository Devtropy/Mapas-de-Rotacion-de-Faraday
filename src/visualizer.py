import numpy as np
import matplotlib.pyplot as plt
import os
import config as cfg


def generar_visualizacion_parametrica():
    ruta_estudio = "../results/estudio_parametrico"
    if not os.path.exists(ruta_estudio):
        return

    for carpeta in os.listdir(ruta_estudio):
        ruta_data = os.path.join(ruta_estudio, carpeta)
        ruta_graficos = os.path.join(ruta_data, "graficos")
        if not os.path.exists(ruta_graficos):
            os.makedirs(ruta_graficos)

        rm = np.load(os.path.join(ruta_data, "rm_mapa.npy"))
        i_tot = np.load(os.path.join(ruta_data, "intensidad.npy"))
        q = np.load(os.path.join(ruta_data, "stokes_q.npy"))
        u = np.load(os.path.join(ruta_data, "stokes_u.npy"))

        p_frac = np.sqrt(q**2 + u**2) / (i_tot + 1e-15)
        psi = 0.5 * np.arctan2(u, q)

        # Mapa RM Individual
        plt.figure(figsize=(8, 7))
        plt.imshow(rm, cmap="seismic", extent=[-768, 768, -768, 768])
        plt.title(f"Mapa RM - Condición {carpeta}")
        plt.colorbar(label="rad/m^2")
        plt.savefig(os.path.join(ruta_graficos, "01_rm.png"))
        plt.close()

        # Histograma Comparativo
        plt.figure(figsize=(8, 6))
        plt.hist(rm.flatten(), bins=50, density=True, alpha=0.7)
        plt.title(f"Distribución de RM (sigma={np.std(rm):.2f})")
        plt.savefig(os.path.join(ruta_graficos, "02_histograma.png"))
        plt.close()

        # Polarización y Vectores
        plt.figure(figsize=(8, 8))
        plt.imshow(
            np.log10(i_tot + 1e-15), cmap="gray_r", extent=[-768, 768, -768, 768]
        )
        x_dim = np.linspace(-768, 768, rm.shape[0])
        x, y = np.meshgrid(x_dim, x_dim)
        skip = max(1, rm.shape[0] // 16)
        plt.quiver(
            x[::skip, ::skip],
            y[::skip, ::skip],
            p_frac[::skip, ::skip] * np.cos(psi[::skip, ::skip]),
            p_frac[::skip, ::skip] * np.sin(psi[::skip, ::skip]),
            pivot="middle",
            color="red",
            headwidth=0,
        )
        plt.title("Intensidad y Vectores de Polarización")
        plt.savefig(os.path.join(ruta_graficos, "03_polarizacion.png"))
        plt.close()


if __name__ == "__main__":
    generar_visualizacion_parametrica()
