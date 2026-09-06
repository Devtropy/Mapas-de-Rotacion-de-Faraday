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

def filament_axis_from_viewing_angle(theta_rad):
    """
    Vector unitario del eje del filamento para un ángulo de vista theta_rad
    respecto a la línea de visión (que faradaymr.los asume siempre fija en
    el eje Z de la caja, axis=-1).

    En vez de rotar la línea de visión se rota el objeto: 
    se gira el eje del filamento dentro de la misma caja cúbica
    regular, dejando la geometría de integración exactamente como está.

    theta_rad = 0    -> eje = (0, 0, 1): filamento "de frente" (paralelo a la LoS).
    theta_rad = pi/2 -> eje = (1, 0, 0): filamento "de lado" (perpendicular a la LoS).

    Rotación 2D en (x, z), no 3D genérica: por simetría cilíndrica el
    "roll" del filamento no es observable.
    """
    import numpy as np
    return np.array([np.sin(theta_rad), 0.0, np.cos(theta_rad)])