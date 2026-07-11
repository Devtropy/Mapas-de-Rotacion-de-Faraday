"""
Observables que se construyen integrando a lo largo de la línea de visión.

Un radiotelescopio no mide el campo magnético ni la densidad de electrones
directamente: mide la luz que atravesó todo el plasma en su camino hacia
nosotros, es decir, integrales a lo largo de la línea de visión (line of
sight, LOS). Este módulo reúne las tres integrales que aparecen una y otra
vez en radioastronomía de medios magnetizados, para no reescribirlas cada
vez que se simula un objeto nuevo (cúmulos, halos galácticos, restos de
supernova, etc.):

1. Medida de Rotación (RM): cuánto gira el ángulo de polarización de la luz
   por efecto Faraday al atravesar un plasma magnetizado,
       RM(x,y) = 0.812 * integral( n_e(l) * B_parallel(l) dl ),
   con n_e en cm^-3, B en microgauss, dl en pc, y RM en rad/m^2. La
   constante 0.812 no es un ajuste: sale de la física del efecto Faraday
   (carga y masa del electrón, permitividad del vacío) expresada en esas
   unidades particulares.

2. Emisividad sincrotrón j_nu: la luz que radían los electrones
   relativistas al girar alrededor de las líneas de campo magnético
   perpendiculares a la línea de visión, con dependencia en frecuencia fija
   por el índice de energía p de la población de electrones.

3. Polarización lineal (Stokes Q, U): la emisión sincrotrón nace polarizada
   en un ángulo psi_0 fijado por la dirección del campo magnético local,
   pero ese ángulo va rotando (efecto Faraday) a medida que la luz sigue
   viajando por el resto del plasma que tiene delante. Por eso Q y U no son
   una simple integral de j_nu: hay que rotar el ángulo con la RM acumulada
   *desde cada punto hasta el observador*, no con la RM total de la línea de
   visión completa.

Justificación de la implementación:
Las tres cantidades comparten la misma estructura (una suma sobre el eje de
línea de visión, multiplicada por el tamaño de celda), así que se separan
en funciones independientes y pequeñas en vez de una única función
monolítica "calcula todo". Esto permite, por ejemplo, calcular solo RM sin
necesidad de una población de electrones relativistas (útil en fuentes de
fondo polarizadas donde no hay sincrotrón local), que es un caso de uso real
en radioastronomía además del estudio del ICM.
"""

from __future__ import annotations

# Constante física: factor de conversión del efecto Faraday para
# n_e en cm^-3, B en microgauss y dl en pc -> RM en rad/m^2.
FARADAY_CONSTANT_CGS = 0.812


def rotation_measure(ne, b_parallel, dl, axis=-1, xp=None):

    """
    Calcula la Medida de Rotación (RM) mediante la suma de
    0.812 * n_e * B_parallel * dl a lo largo de la línea de visión.

    Con Parámetros:

    n_e : ndarray
    Densidad electrónica en cm^-3.
    b_parallel : ndarray
    Componente del campo magnético paralela a la línea de visión,
    en microgauss.
    dl : float
    Paso de integración en pc.
    axis : int
    Eje de la malla que representa la línea de visión.

    Otras cosas a tomar en cuebta: 

    Esta implementación asume una malla cartesiana uniforme con un paso de
    integración constante (dl) para todas las celdas.

    Por ello, es válida para observaciones con líneas de visión paralelas
    (Proyectos I y II).

    No debe utilizarse para ray tracing con un observador interior
    (Proyecto III), donde la longitud recorrida dentro de cada celda depende
    de la dirección del rayo.

    Véase: los_raytrace.py.
    """
    if xp is None:
        import numpy as xp
    # Esta implementación supone un paso de integración constante (dl)
    # a lo largo de toda la línea de visión. No es válida para ray tracing
    # con observador interior, esa funcionalidad deberá
    # implementarse en los_raytrace.py.
    return xp.sum(FARADAY_CONSTANT_CGS * ne * b_parallel * dl, axis=axis)


def rotation_measure_cumulative(ne, b_parallel, dl, axis=-1, xp=None):
    """
    RM acumulada *desde cada punto hasta el borde final* de la línea de
    visión (no desde el origen). Se necesita para la polarización: el ángulo
    observado en un punto solo rota por el plasma que la luz atraviesa
    *después* de emitirse ahí, es decir, entre ese punto y el observador.

    Se implementa invirtiendo el eje, acumulando con `cumsum`, e invirtiendo
    de vuelta: es la manera más simple de expresar "integral desde l hasta
    el final" en términos de una suma acumulada estándar, sin escribir un
    bucle explícito sobre las celdas.
    """
    if xp is None:
        import numpy as xp
    integrando = FARADAY_CONSTANT_CGS * ne * b_parallel * dl
    return xp.flip(xp.cumsum(xp.flip(integrando, axis=axis), axis=axis), axis=axis)


def inclination_angle(bx, by, bz, xp=None):
    """
    Ángulo de inclinación alpha entre el campo magnético local B y el eje
    de integración de la línea de visión (eje z de la caja; ver la
    convención de ejes documentada en `faradaymr.pipeline.ObservationPipeline`,
    donde el LoS es siempre el último eje de la caja 3D).

    Antes este ángulo no se calculaba explícitamente en ningún lado: el
    pipeline obtenía B_perp = sqrt(bx^2+by^2) directamente, lo cual es
    correcto pero es una igualdad que solo se cumple *porque* el LoS
    coincide con z, y esconde la física real, que es que la emisividad
    depende de B*sin(alpha):
        cos(alpha) = B_parallel / |B| = bz / |B|
        sin(alpha) = B_perp / |B|     = sqrt(bx^2+by^2) / |B|
    Se calcula como arctan2(B_perp, B_parallel) en vez de arccos(bz/|B|)
    por dos razones: no diverge/da NaN cuando |B| -> 0 en una celda (no hay
    que dividir por |B| para obtenerlo), y devuelve directamente el ángulo
    en el rango físico correcto [0, pi] sin pasos extra de normalización.
    """
    if xp is None:
        import numpy as xp
    b_perp = xp.sqrt(bx**2 + by**2)
    b_parallel = bz
    return xp.arctan2(b_perp, b_parallel)


def perpendicular_field_magnitude(bx, by, bz, xp=None):
    """
    B_perp = |B| * sin(alpha): la componente del campo magnético
    perpendicular a la línea de visión, que es la que efectivamente
    sostiene la emisión sincrotrón observable (los electrones relativistas
    giran alrededor de las líneas de campo, así que solo la componente
    perpendicular a la LoS contribuye a la radiación que llega al
    observador).

    Se expresa explícitamente como |B|*sin(alpha) -en vez de solo
    sqrt(bx^2+by^2), que da el mismo número pero oculta el término
    sin(alpha) de la física- para que la dependencia angular que pide la
    fórmula de emisividad (j_nu ∝ B_perp^gamma = (B*sin(alpha))^gamma)
    quede explícita en el código y no solo en la convención de ejes.
    """
    if xp is None:
        import numpy as xp
    b_mag = xp.sqrt(bx**2 + by**2 + bz**2)
    alpha = inclination_angle(bx, by, bz, xp=xp)
    return b_mag * xp.sin(alpha)


def synchrotron_emissivity(b_perp, n_rel, frequency, p_index, xp=None):
    """
    Emisividad sincrotrón j_nu ∝ n_rel * B_perp^{(p+1)/2} * nu^{-(p-1)/2},
    con B_perp = B*sin(alpha) (ver `perpendicular_field_magnitude` y
    `inclination_angle` para el cálculo explícito del ángulo de inclinación
    alpha entre el campo local y la línea de visión).

    b_perp: componente del campo magnético perpendicular a la línea de
    visión (solo esa componente produce radiación sincrotrón observable).
    n_rel: densidad de electrones relativistas. p_index: índice de la ley de
    potencia de energía de esos electrones, dN/dE ∝ E^{-p}. frequency: en
    las mismas unidades que se usó para fijar p_index (el framework no
    reescala la constante de proporcionalidad, que depende de detalles
    atómicos que no cambian el análisis relativo entre pixeles).
    """
    if xp is None:
        import numpy as xp
    return (
        n_rel
        * xp.power(b_perp, (p_index + 1) / 2)
        * xp.power(frequency, -(p_index - 1) / 2)
    )


def synchrotron_intensity(j_nu, dl, axis=-1, xp=None):
    """Intensidad total I = integral( j_nu dl ), simple suma sobre el LOS."""
    if xp is None:
        import numpy as xp
    return xp.sum(j_nu * dl, axis=axis)


def polarization_angle_intrinsic(bx, by, xp=None):
    """
    Ángulo de polarización intrínseco de la emisión sincrotrón en cada
    celda, perpendicular a la proyección del campo magnético en el plano del
    cielo (de ahí el signo relativo entre bx y by: la polarización sincrotrón
    es perpendicular a B_perp, no paralela).
    """
    if xp is None:
        import numpy as xp
    return xp.arctan2(bx, -by)


def stokes_qu(j_nu, psi_0, rm_cumulative, wavelength, p_index, dl, axis=-1, xp=None):
    """
    Mapas de Stokes Q y U integrando la emisión sincrotrón con su ángulo de
    polarización ya rotado por Faraday.

    psi_obs = psi_0 + RM_acumulada * lambda^2   (rotación de Faraday)
    Q = suma( f_p * j_nu * cos(2*psi_obs) * dl )
    U = suma( f_p * j_nu * sin(2*psi_obs) * dl )

    f_p = (p+1)/(p+7/3) es el grado de polarización intrínseco máximo de la
    emisión sincrotrón para un índice de energía p (un resultado estándar de
    la teoría de radiación sincrotrón, no un parámetro libre de esta
    simulación). El factor 2 en el ángulo aparece porque Q y U describen una
    magnitud con simetría de 180°, no de 360° (una línea, no una flecha).
    """
    if xp is None:
        import numpy as xp
    fp = (p_index + 1) / (p_index + 7 / 3)
    psi_obs = psi_0 + rm_cumulative * (wavelength**2)
    q_map = xp.sum(fp * j_nu * xp.cos(2 * psi_obs) * dl, axis=axis)
    u_map = xp.sum(fp * j_nu * xp.sin(2 * psi_obs) * dl, axis=axis)
    return q_map, u_map


def polarization_perpendicular_to_projected_field(bx, by, psi_0, xp=None, atol=1e-8):
    """
    Chequeo de consistencia física, celda por celda: el ángulo de
    polarización intrínseco psi_0 debe ser exactamente perpendicular a la
    dirección local del campo magnético proyectado en el plano del cielo
    (bx, by) -- la emisión sincrotrón nace polarizada perpendicular a las
    líneas de campo, nunca paralela a ellas.

    Se calcula el ángulo de la proyección de B, theta_B = arctan2(by, bx), y
    se compara contra psi_0 módulo pi (la polarización lineal tiene simetría
    de 180 grados: una línea, no una flecha con sentido definido, así que
    "perpendicular" solo tiene sentido módulo pi). Devuelve un arreglo
    booleano, pensado para usarse como assert en tests o como diagnóstico
    puntual, no como parte del pipeline de producción en cada corrida.
    """
    if xp is None:
        import numpy as xp
    theta_b = xp.arctan2(by, bx)
    diferencia = xp.mod(psi_0 - theta_b, xp.pi)
    return xp.isclose(diferencia, xp.pi / 2, atol=atol)
