"""
Estilos de figura consistentes con las normas de una revista.

Cada revista de astronomía (ApJ, A&A) impone un ancho de columna, un tamaño
de fuente mínimo legible y ciertas convenciones de estilo (marcas hacia
adentro, ejes con marca arriba y a la derecha además de abajo/izquierda,
etc.). Rehacer esas decisiones a mano en cada script de graficado del
estudio paramétrico -y mantenerlas iguales entre todos los scripts- es
trabajo repetido y una fuente de inconsistencias entre figuras del mismo
paper. `Plotter` centraliza esas decisiones una sola vez y las aplica de
forma global a `matplotlib.rcParams`, para que cualquier figura hecha
después de `aplicar()` -sin importar en qué módulo se dibuje- salga con el
aspecto correcto sin que ese módulo tenga que saber nada de la revista.

Los anchos de columna (3.5" para ApJ, 88 mm para A&A) y tamaños de fuente
son los valores estándar publicados en las guías de autor de ambas
revistas, no un criterio estético arbitrario de esta implementación.
"""

from __future__ import annotations

import logging
from typing import Optional

_logger = logging.getLogger(__name__)

_PULGADAS_POR_MM = 1.0 / 25.4

_ESTILOS = {
    # ApJ: guía de autor de AAS Journals, columna simple ~3.5 in.
    "apj": {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": False,
        "figure.figsize": (3.5, 3.0),
        "savefig.dpi": 300,
    },
    # A&A: guía de autor de Astronomy & Astrophysics, columna simple 88 mm.
    "aanda": {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": False,
        "figure.figsize": (88 * _PULGADAS_POR_MM, 75 * _PULGADAS_POR_MM),
        "savefig.dpi": 300,
    },
}


class Plotter:
    """
    Aplica (y restaura) un estilo de publicación de forma global.

    Se implementa como una clase con estado -y no como una función suelta
    que solo aplica el estilo- porque hace falta poder *volver atrás*: un
    estudio paramétrico puede necesitar figuras de exploración rápida con
    el estilo por defecto de matplotlib y, aparte, un puñado de figuras
    finales con el estilo de la revista; sin un `restaurar()` explícito,
    el estilo de la revista se quedaría pegado a todas las figuras
    siguientes de la sesión, incluidas las que no son para el paper.

    Uso directo:
        plotter = Plotter("apj")
        plotter.aplicar()
        ... graficar ...
        plotter.restaurar()

    O como manejador de contexto, que aplica y restaura automáticamente:
        with Plotter("aanda"):
            ... graficar ...
    """

    def __init__(self, estilo: str = "apj"):
        estilo = estilo.lower()
        if estilo not in _ESTILOS:
            raise ValueError(
                f"Estilo de publicación desconocido: {estilo!r}. "
                f"Opciones disponibles: {sorted(_ESTILOS)}."
            )
        self.estilo = estilo
        self._rcparams_previos: Optional[dict] = None

    def aplicar(self) -> None:
        """Aplica el estilo globalmente a `matplotlib.rcParams`."""
        import matplotlib as mpl

        # Se guardan solo las claves que este estilo va a tocar (no todo
        # rcParams), porque es lo mínimo necesario para poder revertir
        # exactamente lo que `aplicar` cambió, sin arrastrar de vuelta el
        # resto de la configuración de matplotlib del usuario.
        self._rcparams_previos = {
            clave: mpl.rcParams[clave] for clave in _ESTILOS[self.estilo]
        }
        mpl.rcParams.update(_ESTILOS[self.estilo])
        _logger.info("Estilo de publicación '%s' aplicado globalmente.", self.estilo)

    def restaurar(self) -> None:
        """Deshace `aplicar()`, devolviendo esas claves de rcParams a como
        estaban antes."""
        if self._rcparams_previos is None:
            return
        import matplotlib as mpl

        mpl.rcParams.update(self._rcparams_previos)
        _logger.info("Estilo de publicación '%s' restaurado.", self.estilo)
        self._rcparams_previos = None

    def __enter__(self) -> "Plotter":
        self.aplicar()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.restaurar()
