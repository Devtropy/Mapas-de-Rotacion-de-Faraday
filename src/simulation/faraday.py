def RM(ne, Bz, dz):
    return 812 * (ne * Bz).sum(axis=2) * dz
