import numpy as np
import matplotlib.pyplot as plt
import os


def generar_plots():
    ruta_res = "../results/"
    archivos = {
        "RM": "rm_mapa.npy",
        "Intensidad": "intensidad.npy",
        "Stokes Q": "stokes_q.npy",
        "Stokes U": "stokes_u.npy",
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for i, (titulo, nombre) in enumerate(archivos.items()):
        ruta_archivo = os.path.join(ruta_res, nombre)
        if os.path.exists(ruta_archivo):
            data = np.load(ruta_archivo)

            cmap = "seismic" if "Stokes" in titulo or "RM" in titulo else "magma"
            im = axes[i].imshow(data, cmap=cmap, extent=[-768, 768, -768, 768])

            axes[i].set_title(f"Mapa de {titulo}")
            axes[i].set_xlabel("kpc")
            axes[i].set_ylabel("kpc")
            fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        else:
            axes[i].set_title(f"{titulo} (Archivo no encontrado)")

    plt.tight_layout()
    plt.savefig(os.path.join(ruta_res, "analisis_final.png"), dpi=300)
    plt.show()


if __name__ == "__main__":
    generar_plots()
