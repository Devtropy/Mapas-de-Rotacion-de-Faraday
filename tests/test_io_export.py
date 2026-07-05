import numpy as np
import pytest

from faradaymr.io import load_fits, load_hdf5, save_fits, save_hdf5
from faradaymr.plotting import Plotter

astropy = pytest.importorskip("astropy")
h5py = pytest.importorskip("h5py")


def test_fits_guarda_y_recupera_el_mapa_con_su_escala_espacial(tmp_path):
    # Lo que de verdad le importa a quien reabra este FITS con DS9 o
    # astropy es poder saber cuántos kpc mide cada píxel sin tener que
    # recordar el valor de memoria; por eso se verifica el WCS, no solo
    # que el arreglo numérico sobreviva al viaje a disco.
    rm_mapa = np.arange(16.0).reshape(4, 4)
    ruta = tmp_path / "rm_mapa.fits"

    save_fits(str(ruta), rm_mapa, pixel_size=0.5, pixel_unit="kpc", bunit="rad/m2")
    arreglo, header = load_fits(str(ruta))

    assert np.allclose(arreglo, rm_mapa)
    assert header["BUNIT"] == "rad/m2"
    assert np.isclose(header["CDELT1"], 0.5)
    assert header["CUNIT1"].strip() == "kpc"


def test_fits_sin_pixel_size_no_agrega_wcs(tmp_path):
    # Sin una escala espacial no hay nada físico que poner en el WCS; el
    # mapa debe poder guardarse igual, solo que sin esas palabras clave.
    mapa = np.ones((3, 3))
    ruta = tmp_path / "sin_wcs.fits"

    save_fits(str(ruta), mapa)
    arreglo, header = load_fits(str(ruta))

    assert np.allclose(arreglo, mapa)
    assert "CDELT1" not in header


def test_hdf5_agrupa_varias_corridas_de_un_estudio_parametrico(tmp_path):
    # El punto de usar HDF5 en vez de un .npy por mapa por corrida es
    # justamente poder juntar muchas corridas del barrido paramétrico en
    # un solo archivo, cada una identificable por sus propios parámetros
    # físicos (aquí, n_spec y B0) guardados como atributos del grupo.
    ruta = tmp_path / "estudio.h5"

    save_hdf5(
        str(ruta),
        {"rm_mapa": np.full((4, 4), 1.0)},
        attrs={"n_spec": 2.0, "b0_microgauss": 1.0},
        group="corrida_0001",
    )
    save_hdf5(
        str(ruta),
        {"rm_mapa": np.full((4, 4), 2.0)},
        attrs={"n_spec": 3.0, "b0_microgauss": 1.0},
        group="corrida_0002",
    )

    corrida_1 = load_hdf5(str(ruta), group="corrida_0001")
    corrida_2 = load_hdf5(str(ruta), group="corrida_0002")

    assert np.allclose(corrida_1["rm_mapa"], 1.0)
    assert np.allclose(corrida_2["rm_mapa"], 2.0)
    assert corrida_1["_attrs"]["n_spec"] == 2.0
    assert corrida_2["_attrs"]["n_spec"] == 3.0


def test_hdf5_reescribe_un_grupo_existente_sin_mezclar_datasets(tmp_path):
    # Si una corrida se repite (p.ej. se corrigió un parámetro y se volvió
    # a correr con el mismo id de grupo), no debe quedar un dataset viejo
    # conviviendo con los nuevos bajo el mismo grupo.
    ruta = tmp_path / "estudio.h5"

    save_hdf5(str(ruta), {"i_map": np.zeros((2, 2))}, group="corrida")
    save_hdf5(str(ruta), {"rm_mapa": np.ones((2, 2))}, group="corrida")

    datos = load_hdf5(str(ruta), group="corrida")
    assert "i_map" not in datos
    assert np.allclose(datos["rm_mapa"], 1.0)


def test_plotter_aplica_y_restaura_rcparams():
    import matplotlib as mpl

    tamano_original = mpl.rcParams["font.size"]

    with Plotter("apj") as plotter:
        assert plotter.estilo == "apj"
        assert mpl.rcParams["font.size"] == 10

    assert mpl.rcParams["font.size"] == tamano_original


def test_plotter_estilo_desconocido_lanza_error():
    with pytest.raises(ValueError):
        Plotter("nature")
