import numpy as np


def FFT3D(Campo):
    return np.fft.fftn(Campo)


def IFFT3D(Campo):
    return np.fft.ifftn(Campo).real
