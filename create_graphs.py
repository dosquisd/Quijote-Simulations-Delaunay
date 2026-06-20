import concurrent.futures
import ctypes
import logging
import os
import signal
import sys
import time
import traceback
from typing import Literal

import networkx as nx
import numpy as np
import readfof  # Este modulo es de Pylians
from psutil import cpu_count
from scipy.spatial import Delaunay

# --- Evitar procesos huérfanos si el padre muere abruptamente (p. ej. SIGKILL
# del OOM killer) ---------------------------------------------------------
# PR_SET_PDEATHSIG es una syscall de Linux: le dice al kernel "si mi proceso
# padre muere, mándame esta señal automáticamente". A diferencia de cualquier
# try/except o atexit en Python, esto lo hace el kernel, así que funciona
# incluso si el padre recibe SIGKILL y no llega a ejecutar ningún código de
# limpieza. Solo funciona en Linux; en otros sistemas se ignora.
PR_SET_PDEATHSIG = 1


def _worker_initializer() -> None:
    """Se ejecuta una vez en cada proceso worker, apenas se crea, antes de
    procesar cualquier tarea. Configura el 'death signal' y deja que cada
    worker reciba SIGTERM directamente (en vez de ignorar SIGINT, que es el
    comportamiento por defecto de multiprocessing y puede complicar el
    cleanup manual)."""
    if sys.platform.startswith("linux"):
        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            # Si el padre (este proceso worker) muere, el kernel nos manda
            # SIGTERM automáticamente, sin que el padre tenga que hacer nada.
            libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
        except Exception:
            # Si por algún motivo no está disponible (p. ej. contenedor con
            # restricciones), seguimos sin esta protección extra en vez de
            # crashear el worker por algo no crítico.
            pass

    # Restaurar el manejo por defecto de SIGINT en el worker. Por defecto,
    # multiprocessing hace que los workers ignoren SIGINT para que un
    # Ctrl+C lo maneje solo el proceso padre; pero si el padre muere sin
    # poder notificar a los hijos, queremos que los workers SÍ puedan ser
    # terminados directamente con Ctrl+C o SIGTERM si hace falta.
    signal.signal(signal.SIGINT, signal.SIG_DFL)


fof_root: str = "./quijote/FoF"
graphs_root: str = "./quijote/grafos"
snapnums: list[int] = [0, 1, 2, 3]
z_dict: dict[int, float] = {4: 0.0, 3: 0.5, 2: 1.0, 1: 2.0, 0: 3.0}

# Número mínimo de halos para poder construir un Delaunay 3D razonable.
# Qhull necesita al menos 4 puntos no coplanares para el primer simplex;
# pedimos un poco más de margen para evitar casos degenerados.
MIN_HALOS_FOR_DELAUNAY = 5

# Permite controlar el número de workers por variable de entorno, por si
# "núcleos físicos" sigue siendo demasiado para la RAM disponible.
# Ejemplo: MAX_WORKERS=4 uv run create_graphs.py
MAX_WORKERS: int = int(os.environ.get("MAX_WORKERS", cpu_count(logical=False) or 1))

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def distance(point1: list, point2: list) -> float:
    """Calcula la distancia entre dos puntos de la misma dimensión"""
    assert len(point1) == len(point2), "El tamaño debe ser el mismo, gilipollas..."
    return np.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))


def tolist(G: nx.Graph, node, prefix: Literal["", "v"] = "") -> list:
    """En el grafo G (son los que se están guardando como archivos) hay atributos que
    son originalmente vectores (caso de la velocidad y la posición de los halos), y
    para esos casos, los atributos para cada uno de los nodos fue representado de la
    siguiente manera:

    Para el caso de la posición, los atributos se descomponieron en `x`, `y` y `z`.
    Para el caso de la velocidad, los atributos se descomponieron en `vx`, `vy` y `vz`.

    Entonces, lo que realiza la función es que dado un nodo (`node`) de uno de esos grafos (`G`)
    que estamos guardando en archivos, crea su respectiva lista de las posiciones (`prefix=''`) o
    las velocidades (`prefix='v`).

    Hasta el momento, `prefix` únicamente puede tomar esos valores, puesto que esos son los únicos
    que son guardados en listas"""
    return [G.nodes[node][f"{prefix}{i}"] for i in ("x", "y", "z")]


assert os.path.exists(fof_root)
assert os.path.exists(graphs_root)


class SkippedGraph(Exception):
    """Se levanta cuando un catálogo no tiene suficientes halos para construir
    un grafo de Delaunay. No es un error grave, solo indica que se debe saltar
    ese archivo sin tratarlo como una falla real."""


def save_graph(FoF, snapnum: int, save_path: str) -> None:
    logger.debug(f"{save_path} - Procesando grafo...")

    redshift = z_dict[snapnum]

    pos_h = FoF.GroupPos / 1e3  # Posición de los halos en Mpc/h
    mass = FoF.GroupMass * 1e10  # Masa de los halos en Msun/h
    vel_h = FoF.GroupVel * (1.0 + redshift)  # Velocidad particulas de los halos km/s
    Npart = FoF.GroupLen  # Número de partículas CDM en los halos

    n_halos = len(pos_h)
    if n_halos < MIN_HALOS_FOR_DELAUNAY:
        # Esto reemplaza al crash de qhull ("No points given" / "not enough points").
        # No es una falla real del pipeline, simplemente no hay suficiente data.
        raise SkippedGraph(
            f"Solo {n_halos} halo(s) encontrados (mínimo {MIN_HALOS_FOR_DELAUNAY}), "
            f"no se puede construir Delaunay."
        )

    t0 = time.perf_counter()
    delaunay3d = Delaunay(pos_h)
    simplices3d = delaunay3d.simplices
    t1 = time.perf_counter() - t0
    logger.debug(f"{save_path} - Tiempo para calcular Delaunay: {t1:.4f} s")

    G = nx.Graph()

    t0 = time.perf_counter()
    for i, point in enumerate(pos_h):
        pos, vel = point, vel_h[i]

        G.add_node(
            i,
            x=pos[0],
            y=pos[1],
            z=pos[2],
            vx=vel[0],
            vy=vel[1],
            vz=vel[2],
            mass=mass[i],
            npart=Npart[i],
        )
    t2 = time.perf_counter() - t0
    logger.debug(f"{save_path} - Tiempo para agregar nodos: {t2:.4f} s")

    t0 = time.perf_counter()
    for path in simplices3d.tolist():  # type: ignore
        path.append(path[0])
        nx.add_path(G, path)
    t3 = time.perf_counter() - t0
    logger.debug(f"{save_path} - Tiempo para agregar aristas: {t3:.4f} s")

    t0 = time.perf_counter()
    distances = {
        edge: distance(tolist(G, edge[0]), tolist(G, edge[1])) for edge in G.edges
    }
    nx.set_edge_attributes(G, values=distances, name="distance")
    t4 = time.perf_counter() - t0
    logger.debug(f"{save_path} - Tiempo para calcular distancias: {t4:.4f} s")

    logger.info(f"Processing file: {save_path} -- Tiempo total: {t1 + t2 + t3 + t4} s")

    logger.debug(f"{save_path} - Guardando grafo en formato GraphML...")
    nx.write_graphml(G, save_path)

    # Liberar referencias explícitamente ayuda un poco a que el GC libere
    # memoria antes de que el worker tome la siguiente tarea.
    del G, delaunay3d, simplices3d, pos_h, vel_h, mass, Npart


def process_one(args: tuple[str, str, int, str]) -> tuple[str, bool, str]:
    """Función que corre DENTRO de cada worker. Recibe solo strings/ints (nada
    pesado), construye el catálogo FoF localmente, y maneja sus propios errores
    para que una excepción individual no tumbe el ProcessPoolExecutor.

    Retorna (save_path, ok, mensaje) en vez de levantar, así el proceso padre
    siempre puede leer el resultado sin preocuparse de excepciones serializables.
    """
    simu, snapdir_relative, snapnum, save_path = args
    snapdir = f"{fof_root}/{simu}/{snapdir_relative}"

    try:
        FoF = readfof.FoF_catalog(snapdir, snapnum=snapnum, read_IDs=False)
    except FileNotFoundError as e:
        return save_path, False, f"Catálogo no encontrado: {e}"
    except Exception as e:
        return (
            save_path,
            False,
            f"Error cargando catálogo FoF: {e}\n{traceback.format_exc()}",
        )

    try:
        save_graph(FoF, snapnum, save_path)
        return save_path, True, "ok"
    except SkippedGraph as e:
        return save_path, False, f"SKIPPED: {e}"
    except Exception as e:
        # Cubre errores de Delaunay/qhull, scipy, networkx, IO al escribir, etc.
        # OJO: esto NO cubre un SIGKILL por OOM, eso mata el proceso entero y
        # se maneja afuera, en el loop principal, detectando BrokenProcessPool.
        return (
            save_path,
            False,
            f"Error procesando grafo: {e}\n{traceback.format_exc()}",
        )
    finally:
        del FoF


def build_combinations() -> list[tuple[str, str, int, str]]:
    combinations = []
    for simu in os.listdir(fof_root):
        fof_tmp_path = f"{fof_root}/{simu}"
        if os.path.isfile(fof_tmp_path):
            continue

        graph_tmp_path = f"{graphs_root}/{simu}"
        if not os.path.exists(graph_tmp_path):
            os.mkdir(graph_tmp_path)

        for i in os.listdir(fof_tmp_path):
            graph_tmp_path_i = f"{graph_tmp_path}/{i}"
            if not os.path.exists(graph_tmp_path_i):
                os.mkdir(graph_tmp_path_i)

            for snapnum in snapnums:
                if simu == "fiducial_HR" and snapnum > 3:
                    continue

                graph_path = f"{graph_tmp_path_i}/{simu}_{snapnum:03}.graphml.xml"
                if os.path.exists(graph_path):
                    continue

                combinations.append((simu, i, snapnum, graph_path))

    return combinations


def run_pool(combinations: list[tuple[str, str, int, str]]) -> None:
    """Procesa la lista de combinaciones, recreando el ProcessPoolExecutor si
    detecta que se rompió (típicamente por un worker muerto por SIGKILL/OOM),
    para no perder el resto del trabajo pendiente."""

    global MAX_WORKERS

    pending = list(combinations)
    n_ok = 0
    n_failed = 0
    n_skipped = 0
    failed_log: list[str] = []

    while pending:
        logger.info(
            f"Lanzando pool con {MAX_WORKERS} workers para {len(pending)} tareas restantes..."
        )
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=MAX_WORKERS, initializer=_worker_initializer
        )
        future_to_args = {executor.submit(process_one, args): args for args in pending}
        # Vaciamos pending; lo que no se complete por un crash del pool se
        # vuelve a calcular comparando contra lo que sí se resolvió.
        still_pending = list(pending)

        try:
            for future in concurrent.futures.as_completed(future_to_args):
                args = future_to_args[future]
                try:
                    save_path, ok, msg = future.result()
                except concurrent.futures.process.BrokenProcessPool:
                    # Un worker murió (muy probablemente SIGKILL por OOM) mientras
                    # procesaba ESTA tarea en particular. La marcamos como fallida
                    # y dejamos que el resto del pool termine si puede; si el pool
                    # entero está roto, el except externo se encarga de relanzar.
                    logger.error(
                        f"BrokenProcessPool al procesar {args[3]} "
                        f"(probable OOM kill / SIGKILL). Se reintentará en otra ronda "
                        f"con menos paralelismo si vuelve a fallar."
                    )
                    n_failed += 1
                    failed_log.append(f"{args[3]}: BrokenProcessPool (posible OOM)")
                    raise
                except Exception as e:
                    logger.error(f"Excepción inesperada procesando {args[3]}: {e}")
                    n_failed += 1
                    failed_log.append(f"{args[3]}: {e}")
                    still_pending.remove(args)
                    continue

                still_pending.remove(args)
                if ok:
                    n_ok += 1
                elif msg.startswith("SKIPPED"):
                    n_skipped += 1
                    logger.warning(f"{save_path} - {msg}")
                else:
                    n_failed += 1
                    failed_log.append(f"{save_path}: {msg}")
                    logger.error(f"{save_path} - {msg}")

        except concurrent.futures.process.BrokenProcessPool:
            # El pool se rompió. Apagamos lo que quede y reintentamos las
            # tareas que NO alcanzaron a completarse, con menos workers para
            # bajar la presión de memoria.
            executor.shutdown(wait=False, cancel_futures=True)
            pending = still_pending
            if pending:
                new_workers = max(1, MAX_WORKERS - 1)
                logger.warning(
                    f"Pool roto. Quedan {len(pending)} tareas pendientes. "
                    f"Reduciendo workers de {MAX_WORKERS} a {new_workers} y reintentando."
                )
                MAX_WORKERS = new_workers
            continue
        else:
            executor.shutdown(wait=True)
            pending = []

    logger.info(
        f"Resumen final: {n_ok} ok, {n_skipped} saltados (sin suficientes halos), "
        f"{n_failed} fallidos."
    )
    if failed_log:
        logger.info("Detalle de fallos:")
        for line in failed_log:
            logger.info(f"  - {line}")


if __name__ == "__main__":
    combinations = build_combinations()
    logger.info(f"Total combinations to process: {len(combinations)}")
    logger.debug(f"Usando MAX_WORKERS={MAX_WORKERS}")

    run_pool(combinations)
