import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Agregamos la carpeta raíz para que Python encuentre el módulo 'faradaymr'
ruta_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ruta_raiz)

# Ajustamos la ruta para que Python pueda importar 'model.py' desde la carpeta examples
ruta_ejemplos = os.path.abspath(os.path.join(os.path.dirname(__file__), '../examples/filamento_whim'))
sys.path.insert(0, ruta_ejemplos)

# Ahora podemos importar tu función correctamente
from model import construir_escenario

def correr_prueba_visual():
    print("Construyendo el escenario cilíndrico...")
    # Construir un escenario con el filamento alineado al eje X (1, 0, 0)
    bx, by, bz, ne, ne_rel, r = construir_escenario(
        n_spec=3.0, 
        b0_microgauss=0.01,
        axis_direction=[1, 0, 0] # Eje del filamento apuntando en X
    )

    print("Generando las gráficas...")
    # ne es un cubo 3D. Vamos a tomar cortes a la mitad del cubo.
    mitad = ne.shape[0] // 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Corte YZ (Perpendicular al eje X del filamento)
    corte_transversal = ne[mitad, :, :]
    im1 = ax1.imshow(corte_transversal, cmap='viridis', origin='lower')
    ax1.set_title("Corte Transversal (Plano YZ)\nDebería verse como un Círculo")
    plt.colorbar(im1, ax=ax1, label='Densidad Electrónica')

    # Corte XY (Paralelo al eje X del filamento)
    corte_longitudinal = ne[:, :, mitad]
    im2 = ax2.imshow(corte_longitudinal, cmap='viridis', origin='lower')
    ax2.set_title("Corte Longitudinal (Plano XY)\nDebería verse como un Tubo recto")
    plt.colorbar(im2, ax=ax2, label='Densidad Electrónica')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    correr_prueba_visual()