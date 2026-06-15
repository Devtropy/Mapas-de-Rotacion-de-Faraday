import astropy.units as u

# para el observatorio
BEAM_FWHM = 50.0 * u.kpc
# Parametros para la malla
N_BASE = 128  # Adimensional
N_REFINADO = 256  # Adimensional
DX_BASE = 10.0 * u.kpc  # tamaño de pixel
DX_REFINADO = 3.0 * u.kpc  # tamaño de pixel
RADIO_REFINAMIENTO = 200.0 * u.kpc  # radio

# Parametros del medio intrecúmulo (ICM)
N0 = 1e-3 * (u.cm**-3)  # Densidad central de electrones
RC = 400.0 * u.kpc  # Radio del core
BETA = 0.6  # Parametro beta (adimensional)
B0 = 1.0 * u.microgauss  # Intensidad magnetica
MU = 0.5  # Fraccion molecular (adimensional)

# Parametros de turbulencia
LAMBDA_MIN = 6.0 * u.kpc  # Escala mínima de disipación
LAMBDA_MAX = 768.0 * u.kpc  # Escala máxima de inyección
N_SPEC = 3.0  # Índice del espectro magnético
P_SPEC = 3.0  # Índice de energía de electrones relativistas

# Parametros de observación
NU = 1.4e9 * u.Hz  # Frecuencia de observación
C = 3e8 * (u.m / u.s)  # Velocidad de la luz
LAMBDA_ONDA = C / NU  # Longitud de onda

BEAM_ARCSEC = 15.0 * u.arcsec
D_A_MPC = 100.0 * u.Mpc
NOISE_RMS = 1e-6
