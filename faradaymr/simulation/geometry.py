def cylindrical_radius(xx, yy, zz, axis_direction, xp=None):
    """
    Calcula la distancia perpendicular de cada punto en la malla 
    al eje del filamento (simetría cilíndrica).
    
    Parámetros:
    -----------
    xx, yy, zz : array_like
        Mallas de coordenadas 3D.
    axis_direction : list o array_like
        Vector de dirección del eje del filamento (ej. [0, 0, 1]).
    xp : module, opcional
        Módulo de array (numpy o cupy). Si es None, intenta inferirlo.
    """
    if xp is None:
        try:
            import cupy as xp
        except ImportError:
            import numpy as xp

    # Apilar las coordenadas en un solo arreglo vectorial
    pos = xp.stack([xx, yy, zz], axis=-1)
    
    # Normalizar el vector director del eje
    eje = xp.asarray(axis_direction, dtype=float)
    eje = eje / xp.linalg.norm(eje)
    
    # Calcular la proyección de cada punto sobre el eje (producto punto)
    proyeccion = xp.tensordot(pos, eje, axes=([-1], [0]))
    
    # Vector perpendicular = Vector original - Vector proyectado
    perp = pos - proyeccion[..., None] * eje
    
    # La distancia cilíndrica es la norma del vector perpendicular
    return xp.linalg.norm(perp, axis=-1)