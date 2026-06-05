import argparse
import concurrent.futures
import json
import logging
import multiprocessing as mp
import os
import time
from multiprocessing.managers import DictProxy
from pathlib import Path
from typing import Callable, List, Literal

import igraph as ig
import networkx as nx
import nolds
import numpy as np
import scipy.sparse as sp
from scipy.stats import entropy

Snapnums = Literal["000", "001", "002", "003", "004"]

root: str = "./quijote"
graphs_root: str = f"{root}/grafos"
metrics_dir: str = f"{root}/metrics"

assert os.path.exists(graphs_root)

# Inicializar logger (será reconfigurado en __main__)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Evitar "No handlers could be found" warnings


def setup_logging(log_file: str = "metrics.log") -> logging.Logger:
    """
    Configura logging dual: consola (INFO) + archivo (DEBUG).

    Args:
        log_file: Nombre del archivo de log

    Returns:
        Logger configurado
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Handler para consola (INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)

    # Handler para archivo (DEBUG)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def validate_metrics_dict(metrics_dict: dict, lenient: bool = True) -> bool:
    """
    Valida que un diccionario de métricas tenga la estructura esperada.

    Args:
        metrics_dict: Diccionario a validar
        lenient: Si True, acepta subset de métricas (OK si falta fractal_dimension)
                Si False, requiere todas las métricas

    Returns:
        True si es válido, False en caso contrario
    """
    expected_keys = {
        "global_efficiency",
        "local_efficiencies",
        "avg_local_efficiency",
        "entropy",
        "hurst",
        "closeness",
        "betweenness",
        "eigenvector",
        "convergence",
        "laplacian",
    }

    if not isinstance(metrics_dict, dict):
        return False

    found_keys = set(metrics_dict.keys())

    if lenient:
        # Al menos algunas métricas deben existir
        return len(found_keys.intersection(expected_keys)) > 0
    else:
        # Todas las métricas deben existir
        return expected_keys.issubset(found_keys)


def load_temp_metrics(simu_name: str, snapnum: Snapnums) -> dict:
    """
    Carga métricas temporales existentes si existen y están bien formadas.

    Args:
        simu_name: Nombre de la simulación
        snapnum: Snapshot a cargar

    Returns:
        Dict con métricas existentes o dict vacío si no existe/está malformado
    """
    temp_file = Path(metrics_dir) / f"{simu_name}_{snapnum}.json"

    if not temp_file.exists():
        return {}

    try:
        with open(temp_file, "r") as f:
            data = json.load(f)
        logger.debug(f"Cargadas métricas temporales desde {temp_file}")
        return data
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"No se pudo cargar {temp_file}: {repr(e)}. Se sobreescribirá.")
        return {}


def save_temp_metrics(metrics_dict: dict, simu_name: str, snapnum: Snapnums) -> None:
    """
    Guarda métricas temporales con merge inteligente (no sobreescribir existentes).

    Args:
        metrics_dict: Diccionario de métricas a guardar
        simu_name: Nombre de la simulación
        snapnum: Snapshot
    """
    temp_file = Path(metrics_dir) / f"{simu_name}_{snapnum}.json"

    # Si existe, hacer merge inteligente a nivel métrica
    if temp_file.exists():
        try:
            with open(temp_file, "r") as f:
                existing = json.load(f)

            # Contar nuevas métricas agregadas
            new_metrics = 0
            for realization, metrics in metrics_dict.items():
                if realization not in existing:
                    existing[realization] = {}

                for metric_key, metric_value in metrics.items():
                    if metric_key not in existing[realization]:
                        existing[realization][metric_key] = metric_value
                        new_metrics += 1

            metrics_dict = existing
            logger.info(
                f"Agregadas {new_metrics} nuevas métricas a {simu_name}_{snapnum}"
            )
        except json.JSONDecodeError:
            logger.warning(f"JSON malformado en {temp_file}, sobreescribiendo...")
    else:
        logger.info(f"Creado temporal {simu_name}_{snapnum}")

    # Guardar merged o nuevo
    with open(temp_file, "w") as f:
        json.dump(
            metrics_dict,
            f,
            default=lambda x: x.tolist() if hasattr(x, "tolist") else x,
            indent=4,
        )
    logger.debug(f"Guardado: {temp_file}")


def _graph_metric_func(
    func: Callable[[], List[float]],
    shared_dict: DictProxy,
    key_name: str,
    graph_path: str = "",
) -> None:
    logger.debug(f"{graph_path} -- Calculando {key_name}...")
    t0 = time.perf_counter()
    try:
        result = func()
        shared_dict[key_name] = result
    except Exception as e:
        logger.error(f"{graph_path} -- {key_name} -- {repr(e)}", exc_info=True)

    logger.debug(
        f"{graph_path} -- {key_name} calculada en {time.perf_counter() - t0:.2f} segundos"
    )


def calculate_metrics(
    g: ig.Graph,
    save_dict: dict,
    laplacian_path: str,
    cutoff: int | None = None,
    graph_path: str = "",
) -> None:
    """
    Aquí se calculan literalmente todas las métricas que se necesitan para el grafo `g`,
    y son guardadas en un diccionario (en `save_dict`) todos los valores correspondientes
    de cada métrica.

    Args:
        g (igraph.Graph): Grafo no dirigido el cual se le sacarán las métricas, es importante anotar
            que las aristas deben tener un peso el cual es llamado `distance`.
        save_dict (dict): Diccionario en el cual se guardarán el resultado de las métricas.
        laplacian_path (str): Nombre del archivo (junto a su ubicación) donde se guardará el
            resultado de la matriz laplaciana. La extensión de este tipo de archivo debe ser .npz.
        cutoff (int | None): Este parametro es esencial para especificar qué tan aproximado se
            quiere que las métricas de centralidad de betweenness y closeness sean. Por defecto,
            `cutoff=None`, lo que significa que no se hace ninguna aproximación, es decir, que se
            retorna el valor real.
        graph_path (str): Este parámetro solo es usado para imprimir
    Returns:
        None: La función no retorna nada, únicamente devuelve el diccionario completo con las
            métricas que se calcularon y con la matriz laplaciana guardada en la dirección que se
            especificó previamente.
    """

    def global_efficiency(g: ig.Graph) -> float:
        shortest_paths = g.distances()
        efficiency = 0.0
        n = len(g.vs)

        for i in range(n):
            for j in range(i + 1, n):
                if shortest_paths[i][j] > 0:
                    efficiency += 1 / shortest_paths[i][j]

        efficiency /= n * (n - 1) / 2

        return efficiency

    def local_efficiency(g: ig.Graph, v: ig.Vertex) -> float:
        neighbors = g.neighbors(v)
        subgraph = g.subgraph(neighbors)

        if len(neighbors) <= 1:
            return 0

        return global_efficiency(subgraph)

    # Métricas de eficiencia
    if (
        "local_efficiencies" not in save_dict
        or len(save_dict["local_efficiencies"]) > 0
    ):
        logger.debug(f"{graph_path} -- Calculando local_efficiency...")
        t0 = time.perf_counter()
        try:
            # Paralelizar cálculo de local_efficiencies con threads (liberan GIL)
            vertices = list(g.vs)
            n_vertices = len(vertices)

            # Decidir si vale la pena paralelizar (> 1000 vértices)
            if n_vertices > 1000:
                n_threads = max(2, mp.cpu_count() // 4)  # Usar ~25% de cores
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=n_threads
                ) as thread_executor:
                    local_efficiencies = list(
                        thread_executor.map(
                            lambda v: local_efficiency(g, v.index), vertices
                        )
                    )
            else:
                # Para grafos pequeños, calcular secuencialmente es más eficiente
                local_efficiencies = [local_efficiency(g, v.index) for v in vertices]

            avg_local_efficiency = sum(local_efficiencies) / len(local_efficiencies)

            save_dict["local_efficiencies"] = local_efficiencies
            save_dict["avg_local_efficiency"] = avg_local_efficiency
        except Exception as e:
            logger.error(f"{graph_path} -- local_efficiencies -- {repr(e)}")
        logger.debug(
            f"{graph_path} -- local_efficiency calculada en {time.perf_counter() - t0:.2f} segundos"
        )

    # Calcula la centralidad de grado y normalízala
    degree = g.degree()
    degree_distribution = np.array(degree) / sum(degree)

    # Entropia
    if "entropy" not in save_dict:
        logger.debug(f"{graph_path} -- Calculando entropy...")
        t0 = time.perf_counter()
        graph_entropy = entropy(degree_distribution)
        save_dict["entropy"] = graph_entropy
        logger.debug(
            f"{graph_path} -- entropy calculada en {time.perf_counter() - t0:.2f} segundos"
        )

    # Dimensión fractal (box counting)
    if "fractal_dimension" not in save_dict:
        # Falta buscar la función todavía, pero se aplicaría
        # sobre `degree_distribution`
        pass

    # Hurst
    if "hurst" not in save_dict:
        logger.debug(f"{graph_path} -- Calculando hurst...")
        t0 = time.perf_counter()
        hurst = nolds.hurst_rs(degree_distribution, fit="poly")
        save_dict["hurst"] = hurst
        logger.debug(
            f"{graph_path} -- hurst calculada en {time.perf_counter() - t0:.2f} segundos"
        )

    if "eigenvector" not in save_dict or not len(save_dict["eigenvector"]):
        logger.debug(f"{graph_path} -- Calculando eigenvector...")
        t0 = time.perf_counter()
        eigenvector = g.eigenvector_centrality(
            directed=False, weights="distance", scale=False
        )
        save_dict["eigenvector"] = eigenvector
        logger.debug(
            f"{graph_path} -- eigenvector calculada en {time.perf_counter() - t0:.2f} segundos"
        )

    # Matriz laplaciana (La matriz laplaciana no se demora nada)
    logger.debug(f"{graph_path} -- Calculando laplacian...")
    t0 = time.perf_counter()
    G = g.to_networkx()
    laplacian = nx.laplacian_matrix(G, weight="distance")

    # Guardar matriz solo si es un archivo .npz
    if os.path.splitext(laplacian_path)[1] != ".npz":
        laplacian_path = laplacian_path + ".npz"

    sp.save_npz(laplacian_path, laplacian)
    save_dict["laplacian"] = laplacian_path
    logger.debug(
        f"{graph_path} -- laplacian calculada y guardada en {time.perf_counter() - t0:.2f} segundos"
    )

    with mp.Manager() as manager:
        centrality_dict = manager.dict()
        processes = []

        if "global_efficiency" not in save_dict:
            # logger.debug(f"{graph_path} -- Calculando global_efficiency...")
            # t0 = time.perf_counter()
            # try:
            #     efficiency = global_efficiency(g)
            #     save_dict["global_efficiency"] = efficiency
            # except Exception as e:
            #     logger.error(f"{graph_path} -- global_efficiency -- {repr(e)}")
            # logger.debug(
            #     f"{graph_path} -- global_efficiency calculada en {time.perf_counter() - t0:.2f} segundos"
            # )
            process = mp.Process(
                target=_graph_metric_func,
                args=(
                    lambda: global_efficiency(g),
                    centrality_dict,
                    "global_efficiency",
                    graph_path,
                ),
            )
            process.start()
            processes.append(process)

        # Centralidad
        if "closeness" not in save_dict or not len(save_dict["closeness"]):
            # logger.debug(f"{graph_path} -- Calculando closeness...")
            # t0 = time.perf_counter()
            # closeness = g.closeness(weights="distance", normalized=True, cutoff=cutoff)
            # save_dict["closeness"] = closeness
            # logger.debug(
            #     f"{graph_path} -- closeness calculada en {time.perf_counter() - t0:.2f} segundos"
            # )
            process = mp.Process(
                target=_graph_metric_func,
                args=(
                    lambda: g.closeness(
                        weights="distance", normalized=True, cutoff=cutoff
                    ),
                    centrality_dict,
                    "closeness",
                    graph_path,
                ),
            )
            process.start()
            processes.append(process)

        if "betweenness" not in save_dict or not len(save_dict["betweenness"]):
            # logger.debug(f"{graph_path} -- Calculando betweenness...")
            # t0 = time.perf_counter()
            # betweenness = g.betweenness(directed=False, weights="distance", cutoff=cutoff)
            # save_dict["betweenness"] = betweenness
            # logger.debug(
            #     f"{graph_path} -- betweenness calculada en {time.perf_counter() - t0:.2f} segundos"
            # )
            process = mp.Process(
                target=_graph_metric_func,
                args=(
                    lambda: g.betweenness(
                        directed=False, weights="distance", cutoff=cutoff
                    ),
                    centrality_dict,
                    "betweenness",
                    graph_path,
                ),
            )
            process.start()
            processes.append(process)

        if "convergence" not in save_dict or not len(save_dict["convergence"]):
            # logger.debug(f"{graph_path} -- Calculando convergence...")
            # t0 = time.perf_counter()
            # convergence = g.convergence_degree()
            # save_dict["convergence"] = convergence
            # logger.debug(
            #     f"{graph_path} -- convergence calculada en {time.perf_counter() - t0:.2f} segundos"
            # )
            process = mp.Process(
                target=_graph_metric_func,
                args=(
                    lambda: g.convergence_degree(),
                    centrality_dict,
                    "convergence",
                    graph_path,
                ),
            )
            process.start()
            processes.append(process)

        # Esperar a que terminen los procesos de centralidad
        for process in processes:
            process.join()

        save_dict.update(centrality_dict)


def save_metrics(metrics: dict, graph_path: str, laplacian_path: str) -> None:
    """
    Calcula y guarda métricas para un grafo.

    Args:
        metrics (dict): Un diccionario que contiene las métricas a calcular.
        graph_path (str): La ruta del archivo GraphML que representa el grafo.
        laplacian_path (str): La ruta del archivo donde se guardará la matriz laplaciana del grafo.

    Returns:
        None

    Prints:
        Información sobre el grafo (número de vértices y aristas) y el tiempo tomado
        para calcular las métricas.
    """
    g = ig.Graph.Read_GraphML(graph_path)
    logger.info(f"{graph_path} -- Vertices {len(g.vs):,} -- Aristas {len(g.es):,}")

    t0 = time.perf_counter()
    calculate_metrics(g, metrics, laplacian_path, graph_path=graph_path)
    tf = time.perf_counter() - t0

    logger.info(f"{graph_path} -- Tiempo: {tf} segundos")


def is_snapnum(path: Path, snapnum: Snapnums) -> bool:
    """
    Verifica si el archivo especificado por la ruta `path` corresponde a un número de snapshot
    específico (`snapnum`).

    Args:
        path (Path): La ruta del archivo que se desea verificar.
        snapnum (Snapnums): El número de snapshot que se desea comparar.

    Returns:
        bool: `True` si el archivo corresponde al número de snapshot especificado,
        de lo contrario, `False`.
    """
    name_without_ext: str = path.name.split(".")[0]
    return name_without_ext.split("_")[-1] == snapnum


def is_random_simu(simu_path: Path) -> bool:
    """
    Determina si una simulación es aleatoria o no.

    La función verifica si la simulación especificada por el argumento `simu_path`
    pertenece a una simulación aleatoria. Esto se realiza comprobando el nombre
    de la carpeta padre del archivo o directorio proporcionado. Si el nombre de
    la carpeta padre es un número, se considera que no pertenece a una simulación
    aleatoria. En caso contrario, se clasifica como aleatoria.

    Parámetros:
        simu_path (Path): Ruta al archivo o directorio de la simulación.

    Devuelve:
        bool: `True` si la simulación es aleatoria, `False` en caso contrario.
    """
    return not simu_path.parent.name.isdigit()


def parse_simus(value: str) -> list[str]:
    """
    Parses a comma-separated string of simulation names, validates them against
    a predefined list of valid choices, and returns a list of valid simulation names.
    Args:
        value (str): A comma-separated string of simulation names.
    Returns:
        list[str]: A list of valid simulation names.
    Raises:
        argparse.ArgumentTypeError: If any of the provided simulation names are invalid.
    Notes:
        - An empty string will result in an empty list being returned.
        - The global variable `SIMUS_CHOICES` must be defined and contain the valid
          simulation names.
    """
    global SIMUS_CHOICES

    if value == "":
        return []

    # Separar las comas y quitar espacios en blanco
    selected_simus = list(map(lambda s: s.strip(), value.split(",")))

    # Verificar que todas las simulaciones sean correctas
    invalid_simus = [s for s in selected_simus if s not in SIMUS_CHOICES]
    if invalid_simus:
        raise argparse.ArgumentTypeError(
            f"Invalid simulation(s): {', '.join(invalid_simus)}. "
            f"Valid choices are: {', '.join(SIMUS_CHOICES)}"
        )

    return selected_simus


def _process_task(task: tuple) -> dict:
    """
    Función wrapper para ejecutar main() en ProcessPoolExecutor.
    Convierte tupla (simu_name, snapnum, graph_dir) a argumentos.
    Esta función es necesaria porque lambdas no se pueden serializar.

    Args:
        task: Tupla de (simu_name, snapnum, graph_dir)

    Returns:
        Dict con métricas (retorno de main)
    """
    simu_name, snapnum, graph_dir = task
    return main(simu_name, snapnum, graph_dir)


def main(simu_name: str, snapnum: Snapnums, graph_dir: Path) -> dict:
    """
    Procesa una simulación y un snapnum específicos.
    Busca archivos GraphML, carga temporales existentes, calcula/completa métricas
    y guarda con merge inteligente.

    Args:
        simu_name: Nombre de la simulación (ej. "fiducial", "DC_m", "randoms")
        snapnum: Snapshot a procesar ("000" a "004")
        graph_dir: Path al directorio base de grafos (ej. ./quijote/FoF)

    Returns:
        Dict con métricas (estructura temporal)
    """
    metrics = {}
    try:
        # Cargar temporales existentes
        metrics = load_temp_metrics(simu_name, snapnum)

        # Buscar archivos GraphML para esta simulación y snapnum
        simu_path = graph_dir / simu_name
        if not simu_path.exists():
            logger.error(f"No existe ruta: {simu_path}")
            return metrics

        graph_files = list(simu_path.rglob(f"*_{snapnum}*.xml"))
        if not graph_files:
            logger.warning(f"No se encontraron archivos para {simu_name}_{snapnum}")
            return metrics

        logger.info(
            f"Procesando {len(graph_files)} archivo(s) para {simu_name}_{snapnum}"
        )

        # Procesar cada archivo
        for graph_file in graph_files:
            # Determinar estructura (si es random o con realizaciones)
            if simu_name == "randoms":
                # Es random: usar dummy key "_"
                realization = "_"
            else:
                # Tiene realizaciones: usar nombre de carpeta padre (número de iteración)
                realization = graph_file.parent.name

            # Inicializar estructura si no existe
            if realization not in metrics:
                metrics[realization] = {}

            # Cargar grafo y calcular/completar métricas
            laplacian_path = graph_file.parent / f"laplacian_{snapnum}.npz"

            save_metrics(
                metrics=metrics[realization],
                graph_path=str(graph_file),
                laplacian_path=str(laplacian_path),
            )

        # Validar integridad
        for realization in metrics:
            if not validate_metrics_dict(metrics[realization], lenient=True):
                logger.warning(
                    f"Validación parcial para {simu_name}/{realization}/{snapnum}"
                )

        # Guardar temporal con merge inteligente
        save_temp_metrics(metrics, simu_name, snapnum)

    except Exception as e:
        logger.error(
            f"Error procesando {simu_name}_{snapnum}: {repr(e)}", exc_info=True
        )

    return metrics


if __name__ == "__main__":
    # Setup logging
    logger = setup_logging()

    # Crear directorio de temporales
    Path(metrics_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Directorio de temporales: {metrics_dir}")

    SIMUS_CHOICES: list[str] = [
        simu
        for simu in os.listdir(graphs_root)
        if os.path.isdir(os.path.join(graphs_root, simu))
    ]
    epilog_text = "Simulaciones disponibles: " + "".join(
        f"{simu} " for simu in SIMUS_CHOICES
    )

    # Configurar el parser de argumentos
    parser = argparse.ArgumentParser(description="Calcular métricas de grafos Delaunay")
    parser.add_argument(
        "--simus",
        type=str,
        default="",
        help=(
            "Simulación(es) a procesar. Si no se especifica, se procesan todas. "
            + 'Para escoger más de una simulación se debe separar por comas: "simu1,simu2" (default: ""). '
            + epilog_text
        ),
    )
    parser.add_argument(
        "--snapnum",
        type=str,
        choices=["000", "001", "002", "003", "004"],
        default="000",
        help="Número de snapshot a procesar (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Número de procesos paralelos (default: %(default)s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forzar recálculo incluso si temporales existen (default: False)",
    )

    # Parsear los argumentos
    args = parser.parse_args()

    simus: list[str] = parse_simus(args.simus)
    snapnum: Snapnums = args.snapnum
    workers: int = args.workers
    force: bool = args.force

    t0 = time.perf_counter()

    # Determinar cuáles simulaciones procesar
    if simus:
        simus_to_process = simus
    else:
        simus_to_process = SIMUS_CHOICES

    logger.info(f"Simulaciones a procesar: {simus_to_process}")
    logger.info(f"Snapshot: {snapnum}")
    logger.info(f"Modo force: {force}")

    # Recuperación automática: detectar cuáles temporales ya existen
    existing_temporals = set()
    if not force:
        temp_files = list(Path(metrics_dir).glob("*_*.json"))
        for temp_file in temp_files:
            # Parsear nombre: simu_name_snapnum.json
            stem = temp_file.stem  # Quitar .json
            parts = stem.rsplit("_", 1)  # Separar por último _
            if len(parts) == 2:
                simu_name, snapnum_file = parts
                if snapnum_file == snapnum:
                    existing_temporals.add(simu_name)

        if existing_temporals:
            logger.info(f"Temporales existentes (serán saltados): {existing_temporals}")

    # Filtrar simulaciones: solo procesar las que no tienen temporales (o si force=True)
    simus_to_process_filtered = [
        simu for simu in simus_to_process if force or simu not in existing_temporals
    ]

    if not simus_to_process_filtered:
        logger.info("Todas las simulaciones ya tienen temporales. Nada que procesar.")
        logger.info(f"Tiempo total: {time.perf_counter() - t0:.2f} segundos")
    else:
        logger.info(f"Procesando {len(simus_to_process_filtered)} simulación(es)")

        # Crear tareas: (simu_name, snapnum, graph_dir)
        tasks = [
            (simu_name, snapnum, Path(graphs_root))
            for simu_name in simus_to_process_filtered
        ]

        recommended_workers = 2  # TODO: Calculate this value dinamically based on workload and system resources
        workers_to_use = workers if workers is not None else recommended_workers
        logger.info(
            f"Usando {workers_to_use} procesos paralelos (cores disponibles: {mp.cpu_count()})"
        )

        # Ejecutar en paralelo
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers_to_use
        ) as executor:
            # Enviar todas las tareas
            futures = {executor.submit(_process_task, task): task for task in tasks}
            results_list = []

            # Procesar resultados conforme se completan (evita bloqueos)
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    results_list.append(result)
                    logger.debug(f"Completada tarea: {task[0]}_{task[1]}")
                except Exception as e:
                    logger.error(
                        f"Error en tarea {task[0]}_{task[1]}: {repr(e)}", exc_info=True
                    )
                    # Continuar con otras tareas, no bloquear

            logger.info(
                f"Procesadas {len(results_list)} simulación(es) (de {len(tasks)})"
            )

        logger.info(f"Tiempo total: {time.perf_counter() - t0:.2f} segundos")

        # Flush de logging para asegurar que todo se escriba
        for handler in logger.handlers:
            handler.flush()
