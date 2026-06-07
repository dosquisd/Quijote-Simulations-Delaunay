import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import NamedTuple, Set

root: str = "./quijote"
metrics_dir: str = f"{root}/metrics"

# Regex pattern to parse metric filenames: {realization}_{snapnum}_{metric_name}.{ext}
# snapnum is anchored to the 5 valid snapshots to avoid mismatching if realization contains numbers
FILENAME_PATTERN = re.compile(r"^(.+?)_(000|001|002|003|004)_(.+?)\.(json|txt)$")

logger = logging.getLogger("consolidate_metrics")


class ParsedMetricFile(NamedTuple):
    realization: str
    snapnum: str
    metric_name: str
    extension: str
    file_path: Path


def setup_logging(log_file: str = "consolidate_metrics.log") -> logging.Logger:
    """Configura logging dual: consola (INFO) + archivo (DEBUG)."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("consolidate_metrics")
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # Console Handler (INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
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


def parse_filename(file_path: Path) -> ParsedMetricFile | None:
    """Parses a metric filename to extract realization, snapnum, metric name, and extension."""
    match = FILENAME_PATTERN.match(file_path.name)
    if not match:
        return None

    return ParsedMetricFile(
        realization=match.group(1),
        snapnum=match.group(2),
        metric_name=match.group(3),
        extension=match.group(4),
        file_path=file_path,
    )


def consolidate_simulation_snap(simu_name: str, snapnum: str) -> None:
    """Consolidates individual metric files for a simulation and snapshot into the main JSON."""
    simu_dir = Path(metrics_dir) / simu_name
    if not simu_dir.exists() or not simu_dir.is_dir():
        return

    combined_json_path = Path(metrics_dir) / f"{simu_name}_{snapnum}.json"

    # 1. Load existing consolidated data to preserve previously calculated metrics
    consolidated_data = {}
    if combined_json_path.exists():
        try:
            with open(combined_json_path, "r") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    consolidated_data = loaded
                    logger.debug(
                        f"Loaded existing consolidated file: {combined_json_path.name}"
                    )
                else:
                    logger.warning(
                        f"Existing consolidated file {combined_json_path.name} "
                        f"is not a JSON object, starting fresh."
                    )
        except Exception as e:
            logger.error(
                f"Error reading {combined_json_path.name}, overwriting: {repr(e)}"
            )

    # 2. Find and parse all matching individual files for this snapshot
    parsed_files = []
    for p in simu_dir.glob(f"*_{snapnum}_*"):
        parsed = parse_filename(p)
        if parsed and parsed.snapnum == snapnum:
            parsed_files.append(parsed)

    if not parsed_files:
        logger.debug(
            f"No individual files found for {simu_name} and snapshot {snapnum}."
        )
        return

    logger.info(f"Consolidating {len(parsed_files)} file(s) for {simu_name}_{snapnum}")

    # 3. Read and merge individual metrics
    updated_realizations: Set[str] = set()

    for item in parsed_files:
        realization = item.realization
        metric_name = item.metric_name

        try:
            if item.extension == "txt":
                with open(item.file_path, "r") as f:
                    content = f.read().strip()
                if metric_name == "laplacian":
                    # Path to the sparse laplacian matrix
                    value = content
                else:
                    # Scalar global metric value
                    value = float(content)
            else:  # json file
                with open(item.file_path, "r") as f:
                    value = json.load(f)

            if realization not in consolidated_data:
                consolidated_data[realization] = {}

            consolidated_data[realization][metric_name] = value
            updated_realizations.add(realization)

        except Exception as e:
            logger.error(
                f"Error processing individual file {item.file_path.name}: {repr(e)}"
            )

    # 4. Save consolidated data back
    if updated_realizations:
        try:
            # Create parent folder if not exists (should already exist)
            combined_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(combined_json_path, "w") as f:
                json.dump(
                    consolidated_data,
                    f,
                    default=lambda x: x.tolist() if hasattr(x, "tolist") else x,
                    indent=4,
                )
            logger.info(
                f"Successfully updated {combined_json_path.name} with "
                f"{len(updated_realizations)} realization(s)."
            )
        except Exception as e:
            logger.critical(
                f"Failed to write consolidated file {combined_json_path.name}: {repr(e)}"
            )


if __name__ == "__main__":
    logger = setup_logging()

    # Determine simulation choices from metrics directory
    if not os.path.exists(metrics_dir):
        logger.error(f"Metrics folder does not exist: {metrics_dir}")
        exit(1)

    simu_choices = [
        d
        for d in os.listdir(metrics_dir)
        if os.path.isdir(os.path.join(metrics_dir, d))
    ]

    epilog_text = "Available simulations: " + "".join(
        f"{simu} " for simu in simu_choices
    )

    parser = argparse.ArgumentParser(
        description="Consolidate individual metric files into {simu_name}_{snapnum}.json"
    )
    parser.add_argument(
        "--simus",
        type=str,
        default="",
        help=(
            "Simulation(es) a consolidar. Si no se especifica, se consolidan todas. "
            + 'Separa por comas: "simu1,simu2" (default: ""). '
            + epilog_text
        ),
    )
    parser.add_argument(
        "--snapnum",
        type=str,
        choices=["000", "001", "002", "003", "004"],
        default="",
        help="Snapshot a consolidar. Si no se especifica, se consolidan todos.",
    )

    args = parser.parse_args()

    # Parse simulations
    selected_simus = []
    if args.simus:
        selected_simus = [s.strip() for s in args.simus.split(",") if s.strip()]
        invalid = [s for s in selected_simus if s not in simu_choices]
        if invalid:
            logger.error(
                f"Invalid simulation choice(s): {', '.join(invalid)}. "
                f"Choices are: {', '.join(simu_choices)}"
            )
            exit(1)
    else:
        selected_simus = simu_choices

    # Parse snapshots
    selected_snaps = (
        [args.snapnum] if args.snapnum else ["000", "001", "002", "003", "004"]
    )

    logger.info(f"Consolidating simulations: {selected_simus}")
    logger.info(f"Consolidating snapshots: {selected_snaps}")

    for simu in selected_simus:
        for snap in selected_snaps:
            consolidate_simulation_snap(simu, snap)

    logger.info("Consolidation finished.")
