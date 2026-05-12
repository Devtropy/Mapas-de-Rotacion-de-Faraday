import numpy as np


# Algoritmo 3: Cálculo de Polarización y Despolarización
def polarizacion_stokes(B, n_e, j_nu, I_nu, dz, f_p, longitud_onda):

    # Diccionario de variables:
    # B: array 4D (3,N,N,N) del camppio magnético final donde
    #   B[0]=B_x, B[1]=B_y, B[2]=B_z
    # ne: Array 3D de la densidad de electrones térmicos
    # j_nu: Array 3D de la emisividad sincrotrón
    # I_nu: Array 2D de la intensidad sincrotrónica
    # dz : tamaño de celda en la línea de visión
    # f_p : grado de polarización intrínseco
    # longitud de onda : longitud de oonda de observación

    # Etiquetamos las componentes de B
    Bx, By, Bz = B[0], B[1], B[2]

    # 1. Angulo de posicion intrınseco (perpendicular a B en el plano xy)
    Psi_0 = np.arctan2(By, Bx) + np.pi / 2.0

    # 2. Rotación de Faraday acumulada desde la celda hasta el observador
    dRM = 812.0 * n_e * Bz * dz
    RM_z_a_obs = np.flip(np.cumsum(np.flip(dRM, axis=2), axis=2), axis=2)
    Psi_obs = Psi_0 + (RM_z_a_obs * longitud_onda**2)

    # 3. Sumar contribuciones vectoriales de polarización lineal
    dQ = (f_p * j_nu) * np.cos(2.0 * Psi_obs) * dz
    dU = (f_p * j_nu) * np.sin(2.0 * Psi_obs) * dz

    # Sumamos las contribuciones en z
    Q_tot = np.sum(dQ, axis=2)
    U_tot = np.sum(dU, axis=2)

    # 4. Calcular observables finales
    # Evitamos divisiones por cero
    I_nu_segura = np.where(I_nu == 0, 1e-10, I_nu)

    P = np.sqrt(Q_tot**2 + U_tot**2) / I_nu_segura
    Psi_final = 0.5 * np.arctan2(U_tot, Q_tot)

    # En caso de que I_nu = 0
    P[I_nu == 0] = 0.0
    Psi_final[I_nu == 0] = 0.0

    return Q_tot, U_tot, P, Psi_final
