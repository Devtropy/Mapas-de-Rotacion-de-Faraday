import numpy as np
import pytest
from faradaymr.simulation.geometry import cylindrical_radius, filament_axis_from_viewing_angle

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

def test_filament_axis_theta_cero_es_paralelo_a_la_los():
    # theta=0: el filamento se ve "de frente", su eje coincide con el eje Z
    # de la caja, que es la línea de visión que asume `faradaymr.los`.
    eje = filament_axis_from_viewing_angle(0.0)
    np.testing.assert_allclose(eje, [0.0, 0.0, 1.0], atol=1e-12)
 
 
def test_filament_axis_theta_90_es_perpendicular_a_la_los():
    # theta=pi/2: el filamento se ve "de lado", su eje cae en el plano del
    # cielo (perpendicular al eje Z de integración).
    eje = filament_axis_from_viewing_angle(np.pi / 2)
    np.testing.assert_allclose(eje, [1.0, 0.0, 0.0], atol=1e-12)
 
 
def test_filament_axis_siempre_es_vector_unitario():
    # Para cualquier ángulo intermedio, la rotación 2D en (x,z) no cambia
    # la norma del vector: sigue siendo un eje válido para pasarle a
    # `cylindrical_radius`.
    thetas = np.linspace(0.0, np.pi / 2, 25)
    for theta in thetas:
        eje = filament_axis_from_viewing_angle(theta)
        assert np.isclose(np.linalg.norm(eje), 1.0)
 
 
def test_filament_axis_permanece_en_el_plano_xz():
    # El "roll" alrededor del propio eje del filamento no es observable por
    # su simetría cilíndrica, así que la componente Y debe ser exactamente
    # cero para cualquier ángulo -no una rotación 3D genérica.
    thetas = np.linspace(0.0, 2 * np.pi, 13)
    for theta in thetas:
        eje = filament_axis_from_viewing_angle(theta)
        assert eje[1] == 0.0