import argparse

import time
import json
from typing import Literal
from copy import deepcopy

import os
from pathlib import Path

import concurrent.futures
from functools import partial

import numpy as np
import igraph as ig
import networkx as nx

import nolds
import scipy.sparse as sp
from scipy.stats import entropy

Snapnums = Literal["000", "001", "002", "003", "004"]

root: str = "./quijote"
graphs_root: str = f"{root}/grafos"

assert os.path.exists(graphs_root)

save = True


def calculate_metrics(
    g: ig.Graph,
    save_dict: dict,
    laplacian_path: str,
    cutoff: int | None = None,
    graph_path: str = ""                      
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
            for j in range(i+1, n):
                if shortest_paths[i][j] > 0:
                    efficiency += 1/shortest_paths[i][j]

        efficiency /= (n * (n - 1) / 2)

        return efficiency

    def local_efficiency(g: ig.Graph, v: ig.Vertex) -> float:
        neighbors = g.neighbors(v)
        subgraph = g.subgraph(neighbors)

        if len(neighbors) <= 1:
            return 0

        return global_efficiency(subgraph)

    # Métricas de eficiencia
    if 'global_efficiency' not in save_dict:
        try:
            efficiency = global_efficiency(g)
            save_dict['global_efficiency'] = efficiency
        except Exception as e:
            print(f"{graph_path} -- global_efficiency -- {repr(e)}")

    if 'local_efficiencies' not in save_dict or \
        len(save_dict['local_efficiencies']) > 0:
        try:
            local_efficiencies = [local_efficiency(g, v.index) for v in g.vs]
            avg_local_efficiency = sum(local_efficiencies) / len(local_efficiencies)

            save_dict['local_efficiencies'] = local_efficiencies
            save_dict['avg_local_efficiency'] = avg_local_efficiency
        except Exception as e:
            print(f"{graph_path} -- local_efficiencies -- {repr(e)}")

    # Calcula la centralidad de grado y normalízala
    degree = g.degree()
    degree_distribution = np.array(degree) / sum(degree)

    # Entropia
    if 'entropy' not in save_dict:
        graph_entropy = entropy(degree_distribution)
        save_dict['entropy'] = graph_entropy

    # Dimensión fractal (box counting)
    if 'fractal_dimension' not in save_dict:
        # Falta buscar la función todavía, pero se aplicaría
        # sobre `degree_distribution`
        pass

    # Hurst
    if 'hurst' not in save_dict:
        hurst = nolds.hurst_rs(degree_distribution, fit='poly')
        save_dict['hurst'] = hurst

    # Centralidad
    if 'closeness' not in save_dict or \
        len(save_dict['closeness']) > 0:
        closeness = g.closeness(weights='distance', normalized=True, cutoff=cutoff)
        save_dict['closeness'] = closeness

    if 'betweenness' not in save_dict or \
        len(save_dict['betweenness']) > 0:
        betweenness = g.betweenness(directed=False, weights='distance', cutoff=cutoff)
        save_dict['betweenness'] = betweenness

    if 'eigenvector' not in save_dict or \
        len(save_dict['eigenvector']) > 0:
        eigenvector = g.eigenvector_centrality(directed=False, weights='distance', scale=False)
        save_dict['eigenvector'] = eigenvector

    if 'convergence' not in save_dict or \
        len(save_dict['convergence']) > 0:
        convergence = g.convergence_degree()
        save_dict['convergence'] = convergence

    # Matriz laplaciana (La matriz laplaciana no se demora ada)
    G = g.to_networkx()
    laplacian = nx.laplacian_matrix(G, weight='distance')

    # Guardar matriz si es un archivo .npz
    if os.path.splitext(laplacian_path)[1] == '.npz':
        sp.save_npz(laplacian_path, laplacian)
        save_dict['laplacian'] = laplacian_path


def save_metrics(
    metrics: dict,
    graph_path: str,
    laplacian_path: str
) -> None:
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
    print(f'{graph_path} -- Vertices {len(g.vs):,} -- Aristas {len(g.es):,}')

    t0 = time.perf_counter()
    calculate_metrics(g, metrics, laplacian_path)
    tf = time.perf_counter() - t0

    print(f'{graph_path} -- Tiempo: {tf} segundos\n')


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
    name_without_ext: str = path.name.split('.')[0]
    return name_without_ext.split('_')[-1] == snapnum


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


def main(path_simu: Path, metrics: dict, snapnum: Snapnums = "000") -> dict:
    try:
        if is_random_simu(path_simu):
            simu = path_simu.parent.name
            if simu not in metrics:
                metrics[simu] = {}
            if snapnum not in metrics[simu]:
                metrics[simu][snapnum] = {}

            metrics_to_calculate = metrics[simu][snapnum]
        else:
            simu = path_simu.parents[1].name
            if simu not in metrics:
                metrics[simu] = {}
            realization = path_simu.parent.name
            if realization not in metrics[simu]:
                metrics[simu][realization] = {}
            if snapnum not in metrics[simu][realization]:
                metrics[simu][realization][snapnum] = {}

            metrics_to_calculate = metrics[simu][realization][snapnum]

        laplacian_path = path_simu.parent.joinpath(f"laplacian_{snapnum}.npz")
        save_metrics(
            metrics=metrics_to_calculate,
            graph_path=str(path_simu),
            laplacian_path=str(laplacian_path)
        )
    except Exception as e:
        print(repr(e))
    
    return metrics


if __name__ == "__main__":
    metrics_path: str = f'{graphs_root}/metrics_000.json'

    # Configurar el parser de argumentos
    parser = argparse.ArgumentParser(description='Calcular métricas de grafos Delaunay')
    parser.add_argument(
        '--snapnum', 
        type=str,
        choices=["000", "001", "002", "003", "004"],
        default="000",
        help='Número de snapshot a procesar (default: 000)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Número de procesos paralelos (default: 4)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=metrics_path,
        help=f'Ruta para guardar el archivo JSON de métricas (default: {metrics_path})'
    )
    
    # Parsear los argumentos
    args = parser.parse_args()

    SNAPNUM: Snapnums = args.snapnum
    workers = args.workers
    metrics_path = args.output

    t0 = time.perf_counter()

    path = Path(graphs_root)
    simus_paths: list[Path] = list(path.rglob("*.xml"))
    simus_path_snapnum = list(filter(lambda path: is_snapnum(path, SNAPNUM), simus_paths))

    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as json_data:
            metrics = json.load(json_data)
    else:
        metrics = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        try:
            main_func = partial(main, metrics=deepcopy(metrics), snapnum=SNAPNUM)
            all_results = executor.map(main_func, simus_path_snapnum)
        except KeyboardInterrupt:
            save = False
        except Exception as e:
            print(f"Error leyendo los resultados: {repr(e)}")

    for result in all_results:
        if not result:
            continue

        for simu, simu_data in result.items():
            if simu not in metrics:
                metrics[simu] = {}

            if simu == "randoms":
                if SNAPNUM not in metrics[simu]:
                    metrics[simu][SNAPNUM] = {}

                metrics[simu][SNAPNUM].update(simu_data[SNAPNUM])
                continue

            for realization, real_data in simu_data.items():
                if realization not in metrics[simu]:
                    metrics[simu][realization] = {}

                if SNAPNUM not in metrics[simu][realization]:
                    metrics[simu][realization][SNAPNUM] = {}
                
                snap_data = real_data[SNAPNUM]
                metrics[simu][realization][SNAPNUM].update(snap_data)

    if save:
        print('Guardando métricas...')
        with open(metrics_path, 'w') as json_data:
            json.dump(
                metrics,
                json_data,
                default=lambda x: x.tolist() if hasattr(x, 'tolist') else x,
                indent=4
            )

    print(f'Tiempo total: {time.perf_counter() - t0} segundos')
