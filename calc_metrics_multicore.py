import argparse
import concurrent.futures
import json
import logging
import os
import time
from pathlib import Path
from typing import Literal, NamedTuple, TypedDict

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

# Setup logger
logger = logging.getLogger(__name__)


class MetricPaths(TypedDict):
    entropy: Path
    hurst: Path
    global_efficiency: Path
    avg_local_efficiency: Path
    local_efficiencies: Path
    eigenvector: Path
    closeness: Path
    betweenness: Path
    convergence: Path
    laplacian: Path
    laplacian_txt: Path


class CalculationTask(NamedTuple):
    simu_name: str
    realization: str
    snapnum: str
    graph_path: str
    task_type: str
    force: bool
    cutoff: int | None


def setup_logging(log_file: str = "metrics.log") -> logging.Logger:
    """Configura logging dual: consola (DEBUG) + archivo (DEBUG)."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # Console Handler (DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)

    # File Handler (DEBUG)
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


def global_efficiency(g: ig.Graph) -> float:
    shortest_paths = g.distances()
    efficiency = 0.0
    n = len(g.vs)
    if n <= 1:
        return 0.0

    for i in range(n):
        for j in range(i + 1, n):
            if shortest_paths[i][j] > 0:
                efficiency += 1.0 / shortest_paths[i][j]

    efficiency /= n * (n - 1) / 2.0
    return efficiency


def local_efficiency(g: ig.Graph, v: int) -> float:
    neighbors = g.neighbors(v)
    if len(neighbors) <= 1:
        return 0.0
    subgraph = g.subgraph(neighbors)
    return global_efficiency(subgraph)


def get_metric_paths(
    simu_name: str, realization: str, snapnum: str, graph_file: Path
) -> MetricPaths:
    simu_dir = Path(metrics_dir) / simu_name
    prefix = f"{realization}_{snapnum}"
    return {
        "entropy": simu_dir / f"{prefix}_entropy.txt",
        "hurst": simu_dir / f"{prefix}_hurst.txt",
        "global_efficiency": simu_dir / f"{prefix}_global_efficiency.txt",
        "avg_local_efficiency": simu_dir / f"{prefix}_avg_local_efficiency.txt",
        "local_efficiencies": simu_dir / f"{prefix}_local_efficiencies.json",
        "eigenvector": simu_dir / f"{prefix}_eigenvector.json",
        "closeness": simu_dir / f"{prefix}_closeness.json",
        "betweenness": simu_dir / f"{prefix}_betweenness.json",
        "convergence": simu_dir / f"{prefix}_convergence.json",
        "laplacian": graph_file.parent / f"laplacian_{snapnum}.npz",
        "laplacian_txt": simu_dir / f"{prefix}_laplacian.txt",
    }


def worker_task(task: CalculationTask) -> None:
    simu_name = task.simu_name
    realization = task.realization
    snapnum = task.snapnum
    graph_path = Path(task.graph_path)
    task_type = task.task_type
    force = task.force
    cutoff = task.cutoff

    task_id = f"{simu_name}/{realization}/{snapnum}"
    logger.info(f"Starting task {task_type} for {task_id}")

    paths = get_metric_paths(simu_name, realization, snapnum, graph_path)
    simu_metrics_dir = Path(metrics_dir) / simu_name

    # Check what actually needs to be calculated
    metrics_to_calc = []
    if task_type == "fast_metrics":
        fast_list = [
            "entropy",
            "hurst",
            "eigenvector",
            "local_efficiencies",
            "laplacian",
        ]
        for m in fast_list:
            if force:
                metrics_to_calc.append(m)
            else:
                if m == "local_efficiencies":
                    if (
                        not paths["local_efficiencies"].exists()
                        or not paths["avg_local_efficiency"].exists()
                    ):
                        metrics_to_calc.append(m)
                elif m == "laplacian":
                    if (
                        not paths["laplacian"].exists()
                        or not paths["laplacian_txt"].exists()
                    ):
                        metrics_to_calc.append(m)
                else:
                    if not paths[m].exists():
                        metrics_to_calc.append(m)
    else:
        if force or not paths[task_type].exists():
            metrics_to_calc.append(task_type)

    if not metrics_to_calc:
        logger.info(
            f"All sub-metrics for {task_type} of {task_id} already exist. Skipping."
        )
        return

    # Load the graph
    try:
        t_load = time.perf_counter()
        g = ig.Graph.Read_GraphML(str(graph_path))
        logger.debug(
            f"Loaded graph {task_id} in {time.perf_counter() - t_load:.2f}s "
            f"(N={len(g.vs)}, E={len(g.es)})"
        )
    except Exception as e:
        logger.error(f"Failed to load graph {graph_path} for {task_id}: {repr(e)}")
        return

    # Create directory for simulation if not exists
    simu_metrics_dir.mkdir(parents=True, exist_ok=True)

    # Perform task calculations
    if task_type == "fast_metrics":
        # Calculate degree distribution once if needed
        degree_dist = None
        if "entropy" in metrics_to_calc or "hurst" in metrics_to_calc:
            degree = g.degree()
            sum_degree = sum(degree)
            if sum_degree > 0:
                degree_dist = np.array(degree) / sum_degree
            else:
                degree_dist = np.zeros(len(degree))

        if "entropy" in metrics_to_calc:
            try:
                t0 = time.perf_counter()
                val = entropy(degree_dist) if degree_dist is not None else 0.0
                with open(paths["entropy"], "w") as f:
                    f.write(f"{val:.12f}\n")
                logger.info(
                    f"{task_id} - entropy: {val:.6f} ({time.perf_counter() - t0:.2f}s)"
                )
            except Exception as e:
                logger.error(
                    f"{task_id} - entropy calculation failed: {repr(e)}",
                    exc_info=True,
                )

        if "hurst" in metrics_to_calc:
            try:
                t0 = time.perf_counter()
                val = (
                    nolds.hurst_rs(degree_dist, fit="poly")
                    if degree_dist is not None
                    else 0.0
                )
                with open(paths["hurst"], "w") as f:
                    f.write(f"{val:.12f}\n")
                logger.info(
                    f"{task_id} - hurst: {val:.6f} ({time.perf_counter() - t0:.2f}s)"
                )
            except Exception as e:
                logger.error(
                    f"{task_id} - hurst calculation failed: {repr(e)}",
                    exc_info=True,
                )

        if "eigenvector" in metrics_to_calc:
            try:
                t0 = time.perf_counter()
                eigenvector = g.eigenvector_centrality(
                    directed=False, weights="distance", scale=False
                )
                with open(paths["eigenvector"], "w") as f:
                    json.dump(eigenvector, f)
                logger.info(
                    f"{task_id} - eigenvector centrality calculated "
                    f"({time.perf_counter() - t0:.2f}s)"
                )
            except Exception as e:
                logger.error(
                    f"{task_id} - eigenvector calculation failed: {repr(e)}",
                    exc_info=True,
                )

        if "laplacian" in metrics_to_calc:
            try:
                t0 = time.perf_counter()
                G = g.to_networkx()
                laplacian = nx.laplacian_matrix(G, weight="distance")
                paths["laplacian"].parent.mkdir(parents=True, exist_ok=True)
                sp.save_npz(str(paths["laplacian"]), laplacian)
                with open(paths["laplacian_txt"], "w") as f:
                    f.write(f"{paths['laplacian'].resolve()}\n")
                logger.info(
                    f"{task_id} - laplacian matrix saved "
                    f"({time.perf_counter() - t0:.2f}s)"
                )
            except Exception as e:
                logger.error(
                    f"{task_id} - laplacian calculation failed: {repr(e)}",
                    exc_info=True,
                )

        if "local_efficiencies" in metrics_to_calc:
            try:
                t0 = time.perf_counter()
                vertices = list(g.vs)
                n_vertices = len(vertices)
                if n_vertices > 1000:
                    n_threads = max(2, os.cpu_count() // 4)
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=n_threads
                    ) as thread_executor:
                        local_effs = list(
                            thread_executor.map(
                                lambda v: local_efficiency(g, v.index), vertices
                            )
                        )
                else:
                    local_effs = [local_efficiency(g, v.index) for v in vertices]

                avg_local_eff = sum(local_effs) / len(local_effs) if local_effs else 0.0

                with open(paths["local_efficiencies"], "w") as f:
                    json.dump(local_effs, f)
                with open(paths["avg_local_efficiency"], "w") as f:
                    f.write(f"{avg_local_eff:.12f}\n")
                logger.info(
                    f"{task_id} - local efficiencies: {avg_local_eff:.6f} "
                    f"({time.perf_counter() - t0:.2f}s)"
                )
            except Exception as e:
                logger.error(
                    f"{task_id} - local efficiency calculation failed: {repr(e)}",
                    exc_info=True,
                )

    else:
        # Slow metrics
        try:
            t0 = time.perf_counter()
            if task_type == "global_efficiency":
                val = global_efficiency(g)
                with open(paths["global_efficiency"], "w") as f:
                    f.write(f"{val:.12f}\n")
                logger.info(
                    f"{task_id} - global_efficiency: {val:.6f} "
                    f"({time.perf_counter() - t0:.2f}s)"
                )
            elif task_type == "closeness":
                closeness = g.closeness(
                    weights="distance", normalized=True, cutoff=cutoff
                )
                with open(paths["closeness"], "w") as f:
                    json.dump(closeness, f)
                logger.info(
                    f"{task_id} - closeness centrality calculated "
                    f"({time.perf_counter() - t0:.2f}s)"
                )
            elif task_type == "betweenness":
                betweenness = g.betweenness(
                    directed=False, weights="distance", cutoff=cutoff
                )
                with open(paths["betweenness"], "w") as f:
                    json.dump(betweenness, f)
                logger.info(
                    f"{task_id} - betweenness centrality calculated "
                    f"({time.perf_counter() - t0:.2f}s)"
                )
            elif task_type == "convergence":
                convergence = g.convergence_degree()
                with open(paths["convergence"], "w") as f:
                    json.dump(convergence, f)
                logger.info(
                    f"{task_id} - convergence degree calculated "
                    f"({time.perf_counter() - t0:.2f}s)"
                )
        except Exception as e:
            logger.error(
                f"{task_id} - {task_type} calculation failed: {repr(e)}",
                exc_info=True,
            )


def parse_simus(value: str, simu_choices: list[str]) -> list[str]:
    if value.strip() == "":
        return []

    # Separar las comas y quitar espacios en blanco
    selected_simus = list(map(lambda s: s.strip(), value.split(",")))

    # Verificar que todas las simulaciones sean correctas
    invalid_simus = [s for s in selected_simus if s not in simu_choices]
    if invalid_simus:
        raise argparse.ArgumentTypeError(
            f"Invalid simulation(s): {', '.join(invalid_simus)}. "
            f"Valid choices are: {', '.join(simu_choices)}"
        )

    return selected_simus


if __name__ == "__main__":
    # Setup logging
    logger = setup_logging()

    # Ensure metrics_dir exists
    Path(metrics_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Metrics directory: {metrics_dir}")

    # Determine simulation choices from graphs_root
    if not os.path.exists(graphs_root):
        logger.error(f"Graphs root folder does not exist: {graphs_root}")
        exit(1)

    simu_choices: list[str] = [
        simu
        for simu in os.listdir(graphs_root)
        if os.path.isdir(os.path.join(graphs_root, simu))
    ]
    epilog_text = "Available simulations: " + "".join(
        f"{simu} " for simu in simu_choices
    )

    # Arguments parser
    parser = argparse.ArgumentParser(
        description="Calculate Delaunay graph metrics using grouped parallel tasks"
    )
    parser.add_argument(
        "--simus",
        type=str,
        default="",
        help=(
            "Simulation(es) a procesar. Si no se especifica, se procesan todas. "
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

    args = parser.parse_args()

    simus = parse_simus(args.simus, simu_choices)
    snapnum: Snapnums = args.snapnum
    workers = args.workers
    force = args.force

    simus_to_process = simus if simus else simu_choices
    logger.info(f"Simulations to process: {simus_to_process}")
    logger.info(f"Snapshot: {snapnum}")
    logger.info(f"Force mode: {force}")

    # Build tasks
    tasks = []

    for simu_name in simus_to_process:
        simu_path = Path(graphs_root) / simu_name
        if not simu_path.exists():
            logger.error(f"Simulation folder not found: {simu_path}")
            continue

        graph_files = list(simu_path.rglob(f"*_{snapnum}*.xml"))
        if not graph_files:
            logger.warning(f"No graph XML files found for {simu_name}_{snapnum}")
            continue

        for graph_file in graph_files:
            if simu_name == "randoms":
                realization = "_"
            else:
                realization = graph_file.parent.name

            paths = get_metric_paths(simu_name, realization, snapnum, graph_file)

            # Check if fast metrics task is needed
            fast_needed = False
            if force:
                fast_needed = True
            else:
                fast_list = [
                    "entropy",
                    "hurst",
                    "eigenvector",
                    "local_efficiencies",
                    "laplacian",
                ]
                for m in fast_list:
                    if m == "local_efficiencies":
                        if (
                            not paths["local_efficiencies"].exists()
                            or not paths["avg_local_efficiency"].exists()
                        ):
                            fast_needed = True
                            break
                    elif m == "laplacian":
                        if (
                            not paths["laplacian"].exists()
                            or not paths["laplacian_txt"].exists()
                        ):
                            fast_needed = True
                            break
                    else:
                        if not paths[m].exists():
                            fast_needed = True
                            break

            if fast_needed:
                tasks.append(
                    CalculationTask(
                        simu_name=simu_name,
                        realization=realization,
                        snapnum=snapnum,
                        graph_path=str(graph_file),
                        task_type="fast_metrics",
                        force=force,
                        cutoff=None,
                    )
                )

            # Check slow metrics tasks
            for slow_m in [
                "global_efficiency",
                "closeness",
                "betweenness",
                "convergence",
            ]:
                if force or not paths[slow_m].exists():
                    tasks.append(
                        CalculationTask(
                            simu_name=simu_name,
                            realization=realization,
                            snapnum=snapnum,
                            graph_path=str(graph_file),
                            task_type=slow_m,
                            force=force,
                            cutoff=None,
                        )
                    )

    if not tasks:
        logger.info(
            "All selected metrics for the chosen simulations/snapshot are "
            "already calculated. Nothing to do."
        )
        exit(0)

    logger.info(f"Total tasks to execute: {len(tasks)}")
    fast_tasks_count = sum(1 for t in tasks if t.task_type == "fast_metrics")
    slow_tasks_count = len(tasks) - fast_tasks_count
    logger.info(f"  - Fast metrics tasks (grouped): {fast_tasks_count}")
    logger.info(f"  - Slow metrics tasks (individual): {slow_tasks_count}")

    # Determine worker count
    recommended_workers = max(1, os.cpu_count() - 1)
    workers_to_use = workers if workers is not None else recommended_workers
    logger.info(
        f"Using {workers_to_use} worker processes (available cores: {os.cpu_count()})"
    )

    # Run task pool
    t0 = time.perf_counter()
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers_to_use
        ) as executor:
            # Submit tasks
            futures = {executor.submit(worker_task, task): task for task in tasks}

            completed = 0
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                completed += 1
                try:
                    future.result()
                    logger.info(
                        f"Completed [{completed}/{len(tasks)}]: {task.task_type} for "
                        f"{task.simu_name}/{task.realization}/{task.snapnum}"
                    )
                except Exception as e:
                    logger.error(
                        f"Task failed: {task.task_type} for "
                        f"{task.simu_name}/{task.realization}/{task.snapnum}: {repr(e)}"
                    )
    except Exception as e:
        logger.critical(
            f"Process Pool Executor encountered a critical failure: {repr(e)}"
        )

    logger.info(f"Finished processing in {time.perf_counter() - t0:.2f} seconds.")
