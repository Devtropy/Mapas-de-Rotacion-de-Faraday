import numpy as np

from faradaymr import los


def test_rm_de_campo_uniforme_es_analitica():
    # para n_e y B_paralelo constantes a lo largo de
    # una línea de visión de longitud L, la integral se resuelve a mano:
    # RM = 0.812 * n_e * B_parallel * L. Es el caso más simple posible y
    # sirve para confirmar que no hay un factor de escala o un signo mal
    # puesto en la implementación.
    n_e = 1e-3
    b_par = 1.0
    dl = 1.0
    n_celdas = 100

    ne_cubo = np.full((1, 1, n_celdas), n_e)
    b_cubo = np.full((1, 1, n_celdas), b_par)

    rm = los.rotation_measure(ne_cubo, b_cubo, dl, xp=np)
    esperado = 0.812 * n_e * b_par * dl * n_celdas

    assert np.isclose(rm[0, 0], esperado)


def test_rm_cumulative_es_monotona_con_campo_positivo():
    # Con B_parallel > 0 en todas partes, la RM acumulada desde el final
    # hacia el origen solo puede crecer (nunca "perder" contribución ya
    # sumada), así que debe ser una secuencia no decreciente al recorrer la
    # línea de visión en sentido contrario al observador.
    ne_cubo = np.full((1, 1, 20), 1e-3)
    b_cubo = np.full((1, 1, 20), 1.0)

    rm_cum = los.rotation_measure_cumulative(ne_cubo, b_cubo, dl=1.0, xp=np)
    diffs = np.diff(rm_cum[0, 0])

    assert np.all(
        diffs <= 0
    )  # decrece hacia el observador (el último punto ya no acumula nada delante)
    assert rm_cum[0, 0, -1] > 0


def test_stokes_qu_sin_rotacion_recupera_angulo_intrinseco():
    # Si la longitud de onda es cero, no hay rotación de Faraday: el ángulo
    # observado debe coincidir con el ángulo intrínseco psi_0, y por lo
    # tanto Q y U deben ser exactamente los de una fuente sin Faraday.
    j_nu = np.ones((4, 4, 5))
    psi_0 = np.full((4, 4, 5), np.pi / 6)
    rm_cum = np.ones((4, 4, 5))

    q_sin_rm, u_sin_rm = los.stokes_qu(
        j_nu, psi_0, rm_cum, wavelength=0.0, p_index=3.0, dl=1.0, xp=np
    )
    q_esperado, u_esperado = los.stokes_qu(
        j_nu, psi_0, np.zeros_like(rm_cum), wavelength=1.0, p_index=3.0, dl=1.0, xp=np
    )

    assert np.allclose(q_sin_rm, q_esperado)
    assert np.allclose(u_sin_rm, u_esperado)


def test_inclination_angle_casos_limite():
    # Campo puramente paralelo a la LoS (bz>0, bx=by=0): alpha=0, no hay
    # emisión sincrotrón posible (B_perp=0). Campo puramente perpendicular
    # (bz=0): alpha=pi/2, sin(alpha)=1, toda la intensidad de campo
    # contribuye a B_perp.
    bx = np.array([0.0, 3.0, 1.0])
    by = np.array([0.0, 0.0, 0.0])
    bz = np.array([5.0, 0.0, 1.0])

    alpha = los.inclination_angle(bx, by, bz, xp=np)

    assert np.isclose(alpha[0], 0.0)
    assert np.isclose(alpha[1], np.pi / 2)
    assert np.isclose(alpha[2], np.pi / 4)


def test_perpendicular_field_magnitude_coincide_con_pitagoras():
    # B_perp = |B|*sin(alpha) tiene que dar exactamente lo mismo que
    # sqrt(bx^2+by^2) (la fórmula que usaba el pipeline antes de hacer
    # explícito el ángulo alpha): son la misma cantidad física, solo que
    # ahora se llega a ella pasando por sin(alpha) en vez de Pitágoras
    # directo.
    rng = np.random.RandomState(0)
    bx, by, bz = rng.normal(size=(3, 10, 10, 10))

    b_perp_explicito = los.perpendicular_field_magnitude(bx, by, bz, xp=np)
    b_perp_pitagoras = np.sqrt(bx**2 + by**2)

    assert np.allclose(b_perp_explicito, b_perp_pitagoras)


def test_polarizacion_es_perpendicular_al_campo_proyectado():
    # Validación física del objetivo: la polarización lineal intrínseca
    # (perpendicular por construcción a B_perp, ver
    # `los.polarization_angle_intrinsic`) debe ser consistente con la
    # dirección local del campo magnético proyectado en el plano del cielo,
    # para cualquier realización aleatoria del campo, no solo casos
    # particulares.
    rng = np.random.RandomState(1)
    bx, by = rng.normal(size=(2, 8, 8, 8))

    psi_0 = los.polarization_angle_intrinsic(bx, by, xp=np)
    consistente = los.polarization_perpendicular_to_projected_field(
        bx, by, psi_0, xp=np
    )

    assert np.all(consistente)
