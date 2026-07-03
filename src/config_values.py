import config as cfg_units
import astropy.units as u

# Variables adimensionales
N_BASE = cfg_units.N_BASE
N_REFINADO = cfg_units.N_REFINADO
BETA = cfg_units.BETA
MU = cfg_units.MU
N_SPEC = cfg_units.N_SPEC
P_SPEC = cfg_units.P_SPEC

# Extracción de valores en unidades necesarias para la matmática
DX_BASE_KPC = cfg_units.DX_BASE.to(u.kpc).value
DX_REFINADO_KPC = cfg_units.DX_REFINADO.to(u.kpc).value
RADIO_REFINAMIENTO_KPC = cfg_units.RADIO_REFINAMIENTO.to(u.kpc).value

N0_CM3 = cfg_units.N0.to(u.cm**-3).value
RC_KPC = cfg_units.RC.to(u.kpc).value
B0_MG = cfg_units.B0.to(u.microgauss).value

LAMBDA_MIN_KPC = cfg_units.LAMBDA_MIN.to(u.kpc).value
LAMBDA_MAX_KPC = cfg_units.LAMBDA_MAX.to(u.kpc).value
BEAM_FWHM_KPC = cfg_units.BEAM_FWHM.to(u.kpc).value
DESV_EST_RUIDO = cfg_units.DESV_EST_RUIDO
AGREGAR_RUIDO = cfg_units.AGREGAR_RUIDO
NU_HZ = cfg_units.NU.to(u.Hz).value
LAMBDA_ONDA_M = cfg_units.LAMBDA_ONDA.to(u.m).value

# dx en pc para el cálculo de RM
DX_BASE_PC = cfg_units.DX_BASE.to(u.pc).value
