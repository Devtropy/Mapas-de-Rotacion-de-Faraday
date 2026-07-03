import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter
import config_values as cfg


def fwhm_to_sigma(fwhm):

    return fwhm / (2.0 * cp.sqrt(2.0 * cp.log(2.0)))


def sigma_pixels(fwhm_kpc, pixel_size_kpc):

    sigma_kpc = fwhm_to_sigma(fwhm_kpc)
    
    return sigma_kpc / pixel_size_kpc


def aplicar_beam(mapa, fwhm_kpc, pixel_size_kpc):

    sigma_pix = float(sigma_pixels(fwhm_kpc,pixel_size_kpc))

    return gaussian_filter(mapa,sigma=sigma_pix,mode="nearest")


def aplicar_beam_stokes(i_map, q_map, u_map, fwhm_kpc, pixel_size_kpc):

    sigma_pix = float(sigma_pixels(fwhm_kpc, pixel_size_kpc))
    i_beam = gaussian_filter(i_map, sigma=sigma_pix, mode="nearest")
    q_beam = gaussian_filter(q_map, sigma=sigma_pix, mode="nearest")
    u_beam = gaussian_filter(u_map, sigma=sigma_pix, mode="nearest")

    return i_beam, q_beam, u_beam

#con aplicar_beam_stokes ya podemos calcular esto de aca
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


def aplicar_observacion( rm_map, i_map, q_map, u_map, fwhm_kpc, pixel_size_kpc):
   
    rm_beam = aplicar_beam(rm_map, fwhm_kpc, pixel_size_kpc)
    i_beam, q_beam, u_beam = aplicar_beam_stokes( i_map, q_map, u_map, fwhm_kpc, pixel_size_kpc)

    if cfg.AGREGAR_RUIDO:

        rm_beam = agregar_ruido_gaussiano(rm_beam, cfg.DESV_EST_RUIDO)
        i_beam = agregar_ruido_gaussiano( i_beam, cfg.DESV_EST_RUIDO)
        q_beam = agregar_ruido_gaussiano( q_beam, cfg.DESV_EST_RUIDO)
        u_beam = agregar_ruido_gaussiano( u_beam, cfg.DESV_EST_RUIDO)

    p_beam, frac_pol = calcular_fraccion_polarizacion( i_beam, q_beam, u_beam)
    dp = calcular_depolarizacion( i_map, q_map, u_map, i_beam, q_beam, u_beam)

    return (rm_beam, i_beam, q_beam, u_beam, p_beam, frac_pol, dp)


def agregar_ruido_gaussiano(mapa, sigma):

    ruido = cp.random.normal(loc=0.0, scale=sigma, size=mapa.shape)

    return mapa + ruido
    