import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter


def arcsec_to_sigma_pixels(beam_arcsec, d_a_mpc, dx_kpc):
    theta_rad = beam_arcsec * cp.pi / 648000.0
    fwhm_kpc = (d_a_mpc * 1000.0) * theta_rad
    sigma_kpc = fwhm_kpc / (2.0 * cp.sqrt(2.0 * cp.log(2.0)))
    return float(sigma_kpc / dx_kpc)


def aplicar_ruido_gaussiano(mapa, rms_noise):
    ruido = cp.random.normal(0.0, rms_noise, mapa.shape)
    return mapa + ruido


def aplicar_beam_stokes(i_map, q_map, u_map, beam_arcsec, d_a_mpc, dx_kpc, rms_noise):

    sigma_pix = arcsec_to_sigma_pixels(beam_arcsec, d_a_mpc, dx_kpc)

    i_beam = gaussian_filter(i_map, sigma=sigma_pix, mode="nearest")
    q_beam = gaussian_filter(q_map, sigma=sigma_pix, mode="nearest")
    u_beam = gaussian_filter(u_map, sigma=sigma_pix, mode="nearest")

    i_beam = aplicar_ruido_gaussiano(i_beam, rms_noise)
    q_beam = aplicar_ruido_gaussiano(q_beam, rms_noise)
    u_beam = aplicar_ruido_gaussiano(u_beam, rms_noise)

    return i_beam, q_beam, u_beam


# con aplicar_beam_stokes ya podemos calcular esto de aca
def calcular_fraccion_polarizacion(i_beam, q_beam, u_beam):

    p_beam = cp.sqrt(q_beam**2 + u_beam**2)
    frac_pol = p_beam / (i_beam + 1e-12)

    return p_beam, frac_pol


def calcular_depolarizacion(i_map, q_map, u_map, i_beam, q_beam, u_beam):

    p0 = cp.sqrt(q_map**2 + u_map**2)
    fp0 = p0 / (i_map + 1e-12)

    p1 = cp.sqrt(q_beam**2 + u_beam**2)
    fp1 = p1 / (i_beam + 1e-12)

    dp = fp1 / (fp0 + 1e-12)

    return dp


def aplicar_observacion( rm_map, i_map, q_map, u_map, beam_arcsec, d_a_mpc, dx_kpc, rms_noise):
    
    sigma_pix = arcsec_to_sigma_pixels(beam_arcsec, d_a_mpc, dx_kpc)

    rm_beam = gaussian_filter(rm_map, sigma=sigma_pix, mode="nearest")

    i_beam, q_beam, u_beam = aplicar_beam_stokes(
    i_map, q_map, u_map, beam_arcsec, d_a_mpc, dx_kpc, rms_noise
    )

    p_beam, frac_pol = calcular_fraccion_polarizacion(i_beam, q_beam, u_beam)
    
    dp = calcular_depolarizacion(i_map, q_map, u_map, i_beam, q_beam, u_beam)

    return rm_beam, i_beam, q_beam, u_beam, p_beam, frac_pol, dp
