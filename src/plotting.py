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
