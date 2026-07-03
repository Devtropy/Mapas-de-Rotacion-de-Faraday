from __future__ import annotations

import os

from .backend import to_numpy


def save_maps(directory, maps: dict):
    """Guarda cada arreglo del diccionario `maps` como `<directory>/<nombre>.npy`."""
    os.makedirs(directory, exist_ok=True)
    for nombre, arreglo in maps.items():
        if arreglo is None:
            continue
        import numpy as np

        np.save(os.path.join(directory, f"{nombre}.npy"), to_numpy(arreglo))


def load_map(directory, nombre):
    import numpy as np

    return np.load(os.path.join(directory, f"{nombre}.npy"))
