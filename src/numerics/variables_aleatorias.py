import numpy as np


def Fase():
    return np.random.uniform(0, 2 * np.pi)


def Rayleigh(sigma):
    return np.random.rayleigh(sigma)
