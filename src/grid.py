import numpy as np


def malla(N, dx):
    x = np.linspace(-N / 2, N / 2, N) * dx
    return np.meshgrid(x, x, x, indexing="ij")
