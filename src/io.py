"""
Exportación de resultados a disco.

Un mapa 2D (RM, intensidad, Stokes Q/U, ...) recién calculado en memoria no
sirve de mucho una vez que termina el proceso de Python: para compararlo con
observaciones reales, mostrarlo en un paper, o simplemente volver a mirarlo
la semana que viene, tiene que quedar en un archivo. Este módulo ofrece tres
formatos, cada uno para un uso distinto:

- `.npy` (`save_maps`/`load_map`): el más simple, un arreglo suelto por
  archivo; sirve para guardar rápido los resultados de una corrida y
  volver a cargarlos en Python, pero no lleva metadatos ni es un estándar
  fuera de numpy.
- FITS (`save_fits`/`load_fits`): el formato estándar de la astronomía
  observacional. Se usa cuando el mapa se va a comparar con datos reales de
  un radiotelescopio o se va a abrir con herramientas del oficio (DS9,
  CASA, astropy): esas herramientas esperan FITS, no `.npy`.
- HDF5 (`save_hdf5`/`load_hdf5`): pensado para el estudio paramétrico, no
  para una corrida suelta. En vez de un archivo por mapa por corrida (lo
  que con muchas combinaciones de n_spec/B0 se vuelve rápidamente miles de
  archivos sueltos difíciles de organizar), cada corrida se guarda como un
  grupo dentro de un único archivo `.h5`, con sus parámetros físicos
  colgados como atributos del grupo -así el archivo completo del estudio
  paramétrico es autocontenido y queda claro qué corrida es cuál sin tener
  que mantener una convención de nombres de archivo aparte.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from .backend import to_numpy

_logger = logging.getLogger(__name__)


def save_maps(directory, maps: dict):
    """Guarda cada arreglo del diccionario `maps` como `<directory>/<nombre>.npy`."""
    os.makedirs(directory, exist_ok=True)
    for nombre, arreglo in maps.items():
        if arreglo is None:
            continue
        import numpy as np

        np.save(os.path.join(directory, f"{nombre}.npy"), to_numpy(arreglo))
    _logger.info("Mapas guardados en %s (.npy, %d arreglos).", directory, len(maps))


def load_map(directory, nombre):
    import numpy as np

    return np.load(os.path.join(directory, f"{nombre}.npy"))


def save_fits(
    path,
    data_map,
    pixel_size: Optional[float] = None,
    pixel_unit: str = "kpc",
    bunit: Optional[str] = None,
    header_extra: Optional[dict] = None,
):
    """
    Guarda un mapa 2D como FITS con metadatos WCS (World Coordinate System).

    El WCS aquí no describe una posición real en el cielo (RA/Dec): esta es
    una simulación, no una observación apuntada a un objeto real, así que
    no hay una posición celeste que asignarle. Lo que sí tiene sentido
    físico, y es lo que de verdad usa alguien que abra este FITS después,
    es la escala espacial: cuántos kpc (o la unidad que sea) mide cada
    píxel. Por eso se arma un WCS "LINEAR" centrado en el medio del mapa
    con `CDELT`=`pixel_size`, en vez de forzar un sistema de coordenadas
    celeste ficticio que daría una falsa sensación de que el mapa está
    apuntado a algún lugar real del cielo.

    data_map: arreglo 2D (numpy o cupy; se trae a CPU antes de guardar).
    pixel_size: tamaño físico de una celda de la malla; si es None, el
        FITS se guarda sin WCS (solo el arreglo).
    pixel_unit: unidad de `pixel_size` (para el header, informativo).
    bunit: unidad física del mapa en sí (p.ej. "rad/m2" para RM), guardada
        en la palabra clave estándar de FITS `BUNIT`.
    header_extra: pares clave-valor adicionales para el header (p.ej.
        {"NSPEC": 2.5, "B0_UG": 1.0}), para dejar registrados en el propio
        archivo los parámetros físicos que produjeron este mapa -así el
        FITS es autocontenido y no depende de recordar el nombre del
        archivo o una carpeta externa para saber de qué corrida salió.
    """
    from astropy.io import fits
    from astropy.wcs import WCS

    array = to_numpy(data_map)

    header = None
    if pixel_size is not None:
        ny, nx = array.shape
        wcs = WCS(naxis=2)
        # CRPIX en convención FITS (1-indexado); se centra en la celda
        # media del mapa porque no hay ningún otro punto de referencia
        # físico (no hay un centro celeste real que anclar).
        wcs.wcs.crpix = [nx / 2 + 0.5, ny / 2 + 0.5]
        wcs.wcs.cdelt = [pixel_size, pixel_size]
        wcs.wcs.crval = [0.0, 0.0]
        wcs.wcs.ctype = ["LINEAR", "LINEAR"]
        wcs.wcs.cunit = [pixel_unit, pixel_unit]
        header = wcs.to_header()

    hdu = fits.PrimaryHDU(data=array, header=header)
    if bunit is not None:
        hdu.header["BUNIT"] = bunit
    if header_extra:
        for clave, valor in header_extra.items():
            hdu.header[clave] = valor

    directorio = os.path.dirname(os.path.abspath(path))
    if directorio:
        os.makedirs(directorio, exist_ok=True)
    hdu.writeto(path, overwrite=True)
    _logger.info("Mapa guardado en %s (FITS, WCS=%s).", path, pixel_size is not None)


def load_fits(path):
    """
    Carga un FITS guardado con `save_fits`.

    Devuelve (arreglo, header): el header se regresa completo (no solo el
    WCS) porque puede traer parámetros físicos guardados en
    `header_extra` que hacen falta para interpretar el mapa, no solo su
    escala espacial.
    """
    from astropy.io import fits

    with fits.open(path) as hdul:
        return hdul[0].data, hdul[0].header


def save_hdf5(
    path, maps: dict, attrs: Optional[dict] = None, group: Optional[str] = None
):
    """
    Guarda `maps` en un archivo HDF5, opcionalmente como un grupo con
    nombre (`group`) dentro de un archivo compartido por todo el estudio
    paramétrico.

    Se abre en modo "a" (append) en vez de sobreescribir el archivo
    completo: la idea de usar HDF5 aquí es justamente poder ir acumulando,
    corrida tras corrida de un barrido paramétrico, todos los resultados
    en un único archivo, sin tener que decidir de antemano cuántas
    corridas van a caber ni volver a escribir las anteriores cada vez.
    Si `group` ya existe se reemplaza entero (no se mezclan datasets de
    una corrida vieja con la nueva bajo el mismo nombre de grupo).

    attrs: parámetros físicos de la corrida (n_spec, B0, semilla, ...), se
    guardan como atributos HDF5 del grupo -no como un dataset más- porque
    son metadatos escalares de la corrida, no otro mapa 2D.

    Se usa compresión gzip en cada dataset: para un estudio con muchas
    corridas de alta resolución el archivo agregado puede crecer mucho, y
    los mapas de RM/intensidad son razonablemente compresibles (no son
    ruido puro), así que la compresión es prácticamente gratis en tiempo
    de cómputo comparado con el costo de generar la corrida.
    """
    import h5py

    directorio = os.path.dirname(os.path.abspath(path))
    if directorio:
        os.makedirs(directorio, exist_ok=True)

    with h5py.File(path, "a") as archivo:
        if group is not None:
            if group in archivo:
                del archivo[group]
            destino = archivo.create_group(group)
        else:
            destino = archivo

        n_guardados = 0
        for nombre, arreglo in maps.items():
            if arreglo is None:
                continue
            destino.create_dataset(nombre, data=to_numpy(arreglo), compression="gzip")
            n_guardados += 1

        if attrs:
            for clave, valor in attrs.items():
                destino.attrs[clave] = valor

    _logger.info(
        "Corrida guardada en %s (HDF5, grupo=%s, %d mapas).",
        path,
        group or "/",
        n_guardados,
    )


def load_hdf5(path, group: Optional[str] = None) -> dict:
    """
    Carga de vuelta a memoria (como numpy) todos los datasets de `path`
    (o de un `group` puntual dentro de un archivo con muchas corridas).

    Devuelve un diccionario {nombre_del_mapa: arreglo}; los atributos
    físicos guardados con `attrs` en `save_hdf5` se devuelven aparte, en
    la clave especial "_attrs", para no confundirlos con un mapa más.
    """
    import h5py
    import numpy as np

    with h5py.File(path, "r") as archivo:
        origen = archivo[group] if group is not None else archivo
        resultado = {nombre: np.asarray(origen[nombre]) for nombre in origen.keys()}
        resultado["_attrs"] = dict(origen.attrs)

    return resultado
