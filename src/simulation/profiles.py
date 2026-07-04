"""
Perfiles radiales de densidad de electrones térmicos.

El plasma que llena un cúmulo de galaxias (o cualquier halo caliente) no
tiene densidad uniforme: cae con el radio. Distintas familias de objetos, o
distintos niveles de detalle del mismo objeto, se describen mejor con
distintas parametrizaciones de esa caída:

- Modelo beta: la parametrización estándar en rayos X para un halo de gas
  en equilibrio hidrostático con un potencial isotérmico,
      n_e(r) = n0 * (1 + (r/r_core)^2)^(-3*beta/2).
- Doble beta: dos componentes beta sumadas. Se usa cuando el cúmulo tiene
  un "cool core" -un exceso de densidad en el centro, más concentrado que
  el halo general- que una sola componente no puede reproducir a la vez en
  el centro y en las afueras.
- NFW: perfil de Navarro-Frenk-White, n_e(r) ∝ [(r/rs)(1+r/rs)^2]^{-1}.
  Nació para materia oscura, pero la misma forma funcional aproxima bien la
  caída del gas a grandes radios en halos más masivos, donde el potencial
  ya no es isotérmico.
- Tabulado: cuando la densidad no viene de una fórmula cerrada sino de una
  simulación hidrodinámica externa (una tabla (r, n_e) promediada en
  cascarones esféricos), no hay parametrización que ajustar: se interpola.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np


class DensityProfile(ABC):
    """
    Interfaz común de todo modelo de densidad electrónica n_e(r).

    Cualquier perfil (beta, doble beta, NFW, tabulado, o uno nuevo que se
    agregue después) solo necesita implementar `density(r, xp=None)`; el
    resto del framework -armado del escenario, integración de línea de
    visión- depende únicamente de esta interfaz, no de qué fórmula hay
    detrás. Eso es lo que permite intercambiar modelos sin tocar la lógica
    de integración: cualquier objeto que la respete sirve como reemplazo.
    """

    @abstractmethod
    def density(self, r, xp=None):
        """
        Evalúa n_e(r) sobre una malla de radios `r`.

        r debe estar en las mismas unidades de longitud que use el perfil
        (el framework no impone un sistema de unidades particular). `xp` es
        el módulo de arreglos a usar (numpy o cupy, ver `faradaymr.backend`);
        se sigue la misma convención que el resto del framework para poder
        evaluar el perfil directamente sobre una malla en GPU sin copiarla a
        CPU primero.
        """
        raise NotImplementedError

    def __call__(self, r, xp=None):
        """Azúcar sintáctica: `perfil(r)` es lo mismo que `perfil.density(r)`."""
        return self.density(r, xp=xp)


@dataclass
class BetaModel(DensityProfile):
    """
    Perfil beta de densidad electrónica,

        n_e(r) = n0 * (1 + (r/r_core)^2)^(-3*beta/2)

    donde n0 es la densidad central, r_core el radio de núcleo y beta el
    índice que controla qué tan rápido cae la densidad lejos del centro. No
    es una elección arbitraria: es el ajuste típico a perfiles de brillo
    superficial de rayos X observados en cúmulos reales.
    """

    n0: float
    r_core: float
    beta: float

    def density(self, r, xp=None):
        if xp is None:
            import numpy as xp
        return self.n0 * (1.0 + (r / self.r_core) ** 2) ** (-1.5 * self.beta)


@dataclass
class DoubleBetaModel(DensityProfile):
    """
    Suma de dos componentes beta independientes,

        n_e(r) = n0_1*(1+(r/rc_1)^2)^(-3*beta_1/2)
                 + n0_2*(1+(r/rc_2)^2)^(-3*beta_2/2)

    Perspectiva de físico: se usa cuando un solo perfil beta no puede
    reproducir a la vez el exceso de densidad central ("cool core") y la
    caída suave a gran radio de un cúmulo. La componente 1 (r_core chico)
    domina cerca del centro; la componente 2 (r_core grande) domina en las
    afueras. Es una suma, no un producto ni un empalme por tramos, porque
    ambas fases de gas coexisten en el mismo volumen.
    """

    n0_1: float
    r_core_1: float
    beta_1: float
    n0_2: float
    r_core_2: float
    beta_2: float

    def density(self, r, xp=None):
        if xp is None:
            import numpy as xp
        componente_1 = self.n0_1 * (1.0 + (r / self.r_core_1) ** 2) ** (
            -1.5 * self.beta_1
        )
        componente_2 = self.n0_2 * (1.0 + (r / self.r_core_2) ** 2) ** (
            -1.5 * self.beta_2
        )
        return componente_1 + componente_2


@dataclass
class NFWModel(DensityProfile):
    """
    Perfil de Navarro-Frenk-White aplicado a la densidad de electrones,

        n_e(r) = n0 / [ (r/r_s) * (1 + r/r_s)^2 ]

    Perspectiva de físico: el NFW es, originalmente, el perfil de densidad
    de materia oscura que sale de simulaciones de N-cuerpos. Se incluye
    aquí porque la misma forma funcional -caída ~r^-1 en el centro y ~r^-3
    lejos de él- también aproxima bien la densidad de gas en halos masivos
    donde el potencial gravitatorio ya no es isotérmico (a diferencia del
    caso beta, que asume justamente eso).

    r_s es el radio de escala (donde la pendiente logarítmica del perfil
    vale -2). El perfil diverge formalmente en r=0 (~1/r); para evitar un
    0/0 al evaluar justo en el centro de la malla, se satura r a un valor
    mínimo positivo antes de dividir, en vez de dejar pasar un NaN.
    """

    n0: float
    r_s: float

    def density(self, r, xp=None):
        if xp is None:
            import numpy as xp
        x = r / self.r_s
        # Evita la división por cero en r=0 (el perfil NFW diverge ahí
        # formalmente); saturar a un x mínimo positivo es más seguro que
        # dejar pasar un NaN al resto del pipeline.
        x_seguro = xp.where(x <= 0, 1e-12, x)
        return self.n0 / (x_seguro * (1.0 + x_seguro) ** 2)


@dataclass
class TabulatedProfile(DensityProfile):
    """
    Perfil de densidad tomado de una tabla (r, n_e) externa, típicamente el
    promedio en cascarones esféricos de una simulación hidrodinámica (SPH,
    AMR, moving-mesh) en vez de un ajuste analítico.

    Perspectiva de físico: los perfiles beta/doble-beta/NFW son ajustes
    suaves e idealizados; una simulación hidrodinámica captura estructura
    que esas fórmulas no tienen (grumos, choques, un cool core con forma
    arbitraria). Para usar ese resultado en este framework no hace falta
    ajustarle una fórmula: alcanza con interpolar la tabla directamente.

    Justificación de la implementación:
    La interpolación se hace en espacio log-log (log10(r) vs log10(n_e)) en
    vez de lineal, porque n_e(r) típicamente cae varias décadas a lo largo
    de la tabla; interpolar linealmente en espacio real subestima groseramente
    la densidad entre puntos de tabla espaciados en log (que es como se
    suelen tabular estos perfiles). En log-log, un tramo de tabla que sigue
    una ley de potencia -el caso típico- se interpola exactamente.

    Parámetros
    ----------
    r_table, n_e_table : arreglos 1D de igual longitud, r y n_e > 0 (los
        exige la interpolación log-log; no tiene sentido físico una
        densidad negativa o nula en una tabla de este tipo).
    extrapolate : si False (por defecto), fuera del rango de la tabla se
        satura al valor del borde más cercano (política conservadora: no
        inventa una tendencia fuera de donde la simulación tiene datos). Si
        True, extrapola linealmente en espacio log-log, es decir, continúa
        la ley de potencia local del borde de la tabla.
    """

    r_table: Sequence[float]
    n_e_table: Sequence[float]
    extrapolate: bool = False

    def __post_init__(self):
        r_arr = np.asarray(self.r_table, dtype=float)
        n_arr = np.asarray(self.n_e_table, dtype=float)
        if r_arr.shape != n_arr.shape or r_arr.ndim != 1:
            raise ValueError(
                "r_table y n_e_table deben ser arreglos 1D de la misma longitud."
            )
        if np.any(r_arr <= 0) or np.any(n_arr <= 0):
            raise ValueError(
                "TabulatedProfile requiere r_table y n_e_table estrictamente "
                "positivos: la interpolación se hace en espacio log-log."
            )
        orden = np.argsort(r_arr)
        self._r_table = r_arr[orden]
        self._n_e_table = n_arr[orden]
        self._log_r = np.log10(self._r_table)
        self._log_n = np.log10(self._n_e_table)

    @classmethod
    def from_txt(cls, path, extrapolate: bool = False, **kwargs):
        """
        Construye el perfil desde un archivo de texto de dos columnas
        (r, n_e), el formato más común en que se exporta un perfil radial
        promediado de una simulación hidrodinámica externa. `kwargs` se pasa
        a `numpy.loadtxt` (por ejemplo `delimiter=","` para CSV).
        """
        tabla = np.loadtxt(path, **kwargs)
        return cls(r_table=tabla[:, 0], n_e_table=tabla[:, 1], extrapolate=extrapolate)

    def density(self, r, xp=None):
        if xp is None:
            import numpy as xp
        from scipy.interpolate import interp1d

        from ..backend import to_numpy

        # La interpolación (scipy) solo entiende arreglos de CPU, así que se
        # trae `r` a numpy si viniera de un backend GPU y se devuelve el
        # resultado ya reconvertido a `xp` al final: la interfaz sigue
        # siendo transparente al backend, aunque el cómputo interno no lo
        # sea (scipy no tiene equivalente en cupy para esta interpolación).
        r_cpu = np.clip(to_numpy(r), 1e-30, None)  # evita log10(0) en r=0
        log_r_consulta = np.log10(r_cpu)

        valor_borde = (self._log_n[0], self._log_n[-1])
        interpolador = interp1d(
            self._log_r,
            self._log_n,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate" if self.extrapolate else valor_borde,
        )
        n_e_cpu = 10.0 ** interpolador(log_r_consulta)
        return xp.asarray(n_e_cpu)


def beta_model(r, n0, r_core, beta):
    """
    Envoltorio retrocompatible de la antigua función `beta_model`.

    Preferir `BetaModel(n0, r_core, beta).density(r)` (o simplemente
    `BetaModel(n0, r_core, beta)(r)`) en código nuevo: esta función se deja
    solo para no romper a quien ya la importaba directamente.
    """
    return BetaModel(n0=n0, r_core=r_core, beta=beta).density(r)
