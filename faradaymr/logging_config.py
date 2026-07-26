"""
Sistema de logging jerárquico de faradaymr.

Por qué logging y no print(): un estudio paramétrico real no es una corrida
suelta en la laptop, sino muchas corridas en un clúster, cada una con varios
pasos costosos (generar la malla turbulenta, integrar la línea de visión,
aplicar el instrumento). Cuando algo tarda más de lo esperado o falla a la
mitad, `print()` no deja ningún rastro permanente y no distingue "esto es el
avance normal de la corrida" de "esto es una advertencia física" (por
ejemplo, una malla cuya divergencia numérica no es despreciable). Un log con
niveles de severidad y marca de tiempo, guardado en disco, sí permite
reconstruir después qué pasó en una corrida que ya terminó.

La jerarquía no se inventa aparte: Python ya arma el árbol de loggers a
partir del nombre de módulo (`faradaymr.fields.gaussian_random_field` es
hijo de `faradaymr.fields`, que es hijo de `faradaymr`), así que basta con
que cada módulo pida su logger con `logging.getLogger(__name__)` y con
configurar una sola vez el logger raíz "faradaymr" aquí. Subir el nivel de
"faradaymr.fields" a DEBUG, por ejemplo, no afecta al resto del framework.
"""

from __future__ import annotations

import functools
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Optional

NOMBRE_LOGGER_RAIZ = "faradaymr"


def generar_id_simulacion() -> str:
    """
    Identificador único de una corrida: fecha/hora (para poder ordenar
    cronológicamente los logs de un estudio paramétrico sin abrir cada
    archivo) más un sufijo hexadecimal corto (para que dos corridas
    lanzadas el mismo segundo -algo normal al mandar varios trabajos en
    paralelo a un clúster- no terminen pisando el mismo archivo de log).
    """
    marca_tiempo = datetime.now().strftime("%Y%m%d_%H%M%S")
    sufijo = uuid.uuid4().hex[:8]
    return f"{marca_tiempo}_{sufijo}"


def configurar_logging(
    nivel: int = logging.INFO,
    directorio_logs: Optional[str] = None,
    id_simulacion: Optional[str] = None,
) -> logging.Logger:
    """
    Configura el logger raíz "faradaymr" para toda una corrida (llamar una
    sola vez, al arrancar el script, p.ej. `examples/.../run.py`).

    nivel: DEBUG/INFO/WARNING/ERROR de `logging`. INFO por defecto: se ve
        el avance físico de la corrida (qué paso empezó, cuánto tardó)
        sin el detalle interno de cada función auxiliar; DEBUG es para
        cuando de verdad hay que depurar.
    directorio_logs: si se da, además de imprimir en pantalla se escribe
        un archivo `<id_simulacion>.log` ahí, para poder revisar después
        una corrida larga que ya terminó (o que se cayó a la mitad). Si es
        None, solo se imprime en pantalla -conveniente en pruebas rápidas
        donde no interesa ensuciar el disco con logs.
    id_simulacion: para vincular el archivo de log con una corrida
        concreta del estudio paramétrico. Si no se da y se pidió
        `directorio_logs`, se genera uno con `generar_id_simulacion`.

    Devuelve el logger raíz ya configurado (con un atributo extra
    `id_simulacion` colgado encima, por si el llamador lo necesita para,
    por ejemplo, nombrar también los archivos de resultados).
    """
    logger = logging.getLogger(NOMBRE_LOGGER_RAIZ)
    logger.setLevel(nivel)
    logger.propagate = False

    # Si `configurar_logging` se llama más de una vez en el mismo proceso
    # (p.ej. una corrida dentro de otra, o varias corridas de un estudio
    # paramétrico ejecutadas desde el mismo script) hay que limpiar los
    # handlers previos primero: si no, cada llamada agrega un handler más
    # y cada línea de log terminaría apareciendo duplicada N veces.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formato = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    consola = logging.StreamHandler()
    consola.setFormatter(formato)
    logger.addHandler(consola)

    if directorio_logs is not None:
        os.makedirs(directorio_logs, exist_ok=True)
        if id_simulacion is None:
            id_simulacion = generar_id_simulacion()
        ruta_log = os.path.join(directorio_logs, f"{id_simulacion}.log")
        archivo = logging.FileHandler(ruta_log, encoding="utf-8")
        archivo.setFormatter(formato)
        logger.addHandler(archivo)
        logger.info("Inicia la simulación %s (log en %s)", id_simulacion, ruta_log)

    logger.id_simulacion = id_simulacion
    return logger


def _sincronizar_gpu() -> None:
    """
    Si cupy está disponible, espera a que termine todo lo encolado en el
    device antes de seguir.

    cupy lanza las operaciones de forma asíncrona: el control vuelve a
    Python antes de que la GPU haya terminado de calcular de verdad. Sin
    este `synchronize()`, cronometrar un kernel con `time.perf_counter()`
    mediría solo el tiempo de *encolar* la operación, no el de ejecutarla
    -un número siempre artificialmente chico y engañoso para decidir dónde
    optimizar. En un entorno sin GPU (numpy puro) esta función no hace
    nada, porque no hay ninguna cola asíncrona que esperar.
    """
    try:
        import cupy as cp

        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass


def medir_tiempo_kernel(func=None, *, logger: Optional[logging.Logger] = None):
    """
    Decorador para medir cuánto tarda en la práctica un kernel de cómputo
    pesado (la FFT de la malla turbulenta, la integración de la línea de
    visión y la respuesta instrumental, etc.).

    Por qué medir esto en vez de solo "verlo correr": en un estudio
    paramétrico con muchas corridas, el tiempo de GPU casi siempre se
    concentra en dos o tres pasos, y sin un número concreto es fácil
    terminar optimizando la parte equivocada. Se reporta tiempo de reloj
    (wall-clock, con `time.perf_counter`) y no tiempo de CPU, porque para
    un kernel de GPU lo que importa es cuánto se espera de verdad,
    incluida la transferencia de datos entre CPU y GPU.

    Se puede usar tal cual (`@medir_tiempo_kernel`) o pasándole un logger
    específico (`@medir_tiempo_kernel(logger=mi_logger)`); si no se pasa
    ninguno, usa el logger del módulo donde vive la función decorada,
    respetando la jerarquía normal de logging.
    """

    def decorador(f):
        nombre_kernel = f.__qualname__

        @functools.wraps(f)
        def envoltura(*args, **kwargs):
            log = logger or logging.getLogger(f.__module__)
            _sincronizar_gpu()
            inicio = time.perf_counter()
            resultado = f(*args, **kwargs)
            _sincronizar_gpu()
            duracion = time.perf_counter() - inicio
            log.info("Kernel '%s' terminó en %.4f s", nombre_kernel, duracion)
            return resultado

        return envoltura

    if func is not None:
        return decorador(func)
    return decorador
