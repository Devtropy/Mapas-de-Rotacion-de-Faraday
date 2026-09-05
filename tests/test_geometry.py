import numpy as np
import pytest
from faradaymr.simulation.geometry import cylindrical_radius

def test_cylindrical_radius_eje_z():
    """Prueba que el radio cilíndrico respecto al eje Z ignora la coordenada Z."""
    # Crear una malla pequeña de 3x3x3
    eje = np.array([-1.0, 0.0, 1.0])
    xx, yy, zz = np.meshgrid(eje, eje, eje, indexing="ij")
    
    # Eje del filamento apuntando en Z
    axis_direction = [0, 0, 1]
    
    # Calcular usando la nueva función
    r_calculado = cylindrical_radius(xx, yy, zz, axis_direction, xp=np)
    
    # El radio teórico para un cilindro en Z es sqrt(x^2 + y^2)
    r_teorico = np.sqrt(xx**2 + yy**2)
    
    # Afirmar que ambos arreglos son casi exactamente iguales
    np.testing.assert_allclose(r_calculado, r_teorico, rtol=1e-5)

def test_cylindrical_radius_eje_inclinado():
    """Prueba un punto específico con un eje diagonal."""
    # Un solo punto en (1, 1, 1)
    xx, yy, zz = np.array([1.0]), np.array([1.0]), np.array([1.0])
    
    # Eje diagonal en el plano XY (vector [1, 1, 0])
    axis_direction = [1, 1, 0]
    
    r_calculado = cylindrical_radius(xx, yy, zz, axis_direction, xp=np)
    
    # La distancia perpendicular de (1,1,1) a la recta x=y, z=0 es 1 (la altura en Z)
    np.testing.assert_allclose(r_calculado, [1.0], rtol=1e-5)