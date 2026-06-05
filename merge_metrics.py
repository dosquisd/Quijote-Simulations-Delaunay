"""
Script para unificar archivos de métricas temporales en un JSON final.

Lee todos los archivos metrics/{simu_name}_{snapnum}.json y los combina
en una estructura jerárquica final: ./quijote/grafos/metrics_merged.json

Estructura temporal: metrics/fiducial_000.json
{
    "10001": {
        "global_efficiency": 0.166...,
        "hurst": 0.508...,
        ...
    },
    "10002": { ... }
}

Estructura temporal: metrics/randoms_000.json
{
    "_": {
        "global_efficiency": 0.166...,
        ...
    }
}

Estructura final: metrics_merged.json
{
    "fiducial": {
        "10001": {
            "000": { "global_efficiency": 0.166..., ... },
            "001": { ... },
            ...
        },
        "10002": { ... }
    },
    "randoms": {
        "_": {
            "000": { ... },
            "001": { ... }
        }
    }
}
"""

import json
import logging
from pathlib import Path
from typing import Literal

Snapnums = Literal["000", "001", "002", "003", "004"]

root: str = "./quijote"
metrics_dir: str = f"{root}/metrics"
output_file: str = f"{root}/metrics_merged.json"


def setup_logging(log_file: str = "merge_metrics.log") -> logging.Logger:
    """Configura logging dual: consola (INFO) + archivo (DEBUG)."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Handler para consola (INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)

    # Handler para archivo (DEBUG)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def merge_metrics() -> dict:
    """
    Lee todos los temporales y los combina en estructura final.

    Returns:
        Dict con estructura final jerárquica
    """
    logger = logging.getLogger(__name__)

    merged = {}

    # Listar todos los temporales
    temp_files = sorted(Path(metrics_dir).glob("*_*.json"))

    if not temp_files:
        logger.warning(f"No se encontraron archivos en {metrics_dir}")
        return merged

    logger.info(f"Encontrados {len(temp_files)} archivo(s) temporal(es)")

    for temp_file in temp_files:
        # Parsear nombre: simu_name_snapnum.json
        stem = temp_file.stem  # Quitar .json
        parts = stem.rsplit("_", 1)  # Separar por último _

        if len(parts) != 2:
            logger.warning(
                f"Nombre inválido (esperado 'simu_snapnum'): {temp_file.name}"
            )
            continue

        simu_name, snapnum = parts

        # Cargar JSON temporal
        try:
            with open(temp_file, "r") as f:
                temp_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"JSON malformado en {temp_file.name}: {repr(e)}")
            continue
        except IOError as e:
            logger.error(f"No se pudo leer {temp_file.name}: {repr(e)}")
            continue

        # Agregar a estructura final
        if simu_name not in merged:
            merged[simu_name] = {}

        # temp_data tiene estructura: {realization: {metric: value}}
        for realization, metrics_dict in temp_data.items():
            if not isinstance(metrics_dict, dict):
                logger.warning(
                    f"Estructura inválida en {simu_name}/{realization} "
                    f"(esperado dict, encontrado {type(metrics_dict).__name__})"
                )
                continue

            if realization not in merged[simu_name]:
                merged[simu_name][realization] = {}

            if snapnum not in merged[simu_name][realization]:
                merged[simu_name][realization][snapnum] = {}

            # Detectar conflictos: si una métrica ya existe con valor diferente
            for metric_key, metric_value in metrics_dict.items():
                if metric_key in merged[simu_name][realization][snapnum]:
                    existing_value = merged[simu_name][realization][snapnum][metric_key]

                    # Comparación especial para listas (tolerancia numérica)
                    if isinstance(metric_value, list) and isinstance(
                        existing_value, list
                    ):
                        if len(metric_value) == len(existing_value):
                            # Aceptar como iguales (mismo snapshot, probablemente recálculo)
                            logger.debug(
                                f"Métrica {metric_key} ya existe en "
                                f"{simu_name}/{realization}/{snapnum} (skipear)"
                            )
                        else:
                            logger.warning(
                                f"Conflicto de tamaño en {simu_name}/{realization}/{snapnum}/"
                                f"{metric_key}: {len(existing_value)} vs {len(metric_value)}"
                            )
                    elif metric_value != existing_value:
                        logger.warning(
                            f"Conflicto en {simu_name}/{realization}/{snapnum}/"
                            f"{metric_key}: {existing_value} vs {metric_value}"
                        )
                else:
                    merged[simu_name][realization][snapnum][metric_key] = metric_value

        logger.info(f"Procesado: {simu_name}_{snapnum}")

    return merged


def clean_randoms_key(merged_dict: dict) -> dict:
    """
    Convierte la key dummy "_" de randoms a la estructura final sin ese nivel.

    Antes: randoms: { "_": { "000": { ... } } }
    Después: randoms: { "000": { ... } }

    Args:
        merged_dict: Dict con estructura temporal

    Returns:
        Dict con estructura final (sin dummy key)
    """
    logger = logging.getLogger(__name__)

    if "randoms" in merged_dict and "_" in merged_dict["randoms"]:
        merged_dict["randoms"] = merged_dict["randoms"]["_"]
        logger.info("Limpiada key dummy '_' para randoms")

    return merged_dict


if __name__ == "__main__":
    # Setup logging
    logger = setup_logging()

    # Validar directorio de temporales
    metrics_path = Path(metrics_dir)
    if not metrics_path.exists():
        logger.error(f"Directorio no existe: {metrics_dir}")
        exit(1)

    logger.info(f"Directorio de temporales: {metrics_dir}")
    logger.info(f"Archivo de salida: {output_file}")

    # Mergear
    merged = merge_metrics()

    if not merged:
        logger.warning("Ninguna métrica fue mergeada. Archivo de salida no se creará.")
        exit(1)

    # Limpiar key dummy de randoms
    merged = clean_randoms_key(merged)

    # Guardar resultado
    try:
        with open(output_file, "w") as f:
            json.dump(
                merged,
                f,
                default=lambda x: x.tolist() if hasattr(x, "tolist") else x,
                indent=4,
            )
        logger.info(f"Guardado exitosamente: {output_file}")
        logger.info(f"Simulaciones mergeadas: {list(merged.keys())}")
    except IOError as e:
        logger.error(f"No se pudo guardar {output_file}: {repr(e)}")
        exit(1)
