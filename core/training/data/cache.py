import shutil
from pathlib import Path
from typing import Literal

import orjson
import polars as pl
from clearml import Dataset
from tqdm.auto import tqdm

from core.utils import setup_logger

logger = setup_logger(__file__)

COMMANDS = Literal["RECREATE", "CREATE", "OK"]


def prepare_dataset_cache(dataset: Dataset, subsets: list[str] | None = None) -> Path:
    """ "Return a ready-to-use cache directory for the dataset, recreating it if needed."""
    if subsets is None:
        subsets = ["train", "test", "val"]

    cache_path = _get_cache_path(dataset)
    command = _check_cache(cache_path, subsets)
    match command:
        case "OK":
            pass
        case "CREATE":
            _create_cache(dataset, cache_path, subsets)
        case "RECREATE":
            _clear_cache(cache_path)
            _create_cache(dataset, cache_path, subsets)
        case _:
            raise ValueError(f"Unknown command: {command}")
    return cache_path


def _get_cache_path(dataset: Dataset) -> Path:
    return Path.home() / ".cache" / "datasets-cache" / dataset.id


def _check_cache(cache_path: Path, subsets: list[str]) -> COMMANDS:
    if cache_path.exists():
        try:
            metadata = orjson.loads((cache_path / "metadata.json").read_bytes())

            command = None
            for subset in subsets:
                arrow_files = list(cache_path.glob(f"*{subset}*.arrow"))
                len_from_metadata = metadata.get(subset, 0)
                if len(arrow_files) != len_from_metadata:
                    logger.warning(
                        f"Cache for subset '{subset}' is invalid (expected {metadata.get(subset, 0)} files, found {len(arrow_files)})."
                    )
                    logger.warning("Cache will be recreated.")
                    command = "RECREATE"

            if command is None:
                command = "OK"
            return command
        except Exception as e:
            logger.error(f"Error checking cache: {e}")
            return "RECREATE"
    else:
        return "CREATE"


def _clear_cache(cache_path: Path) -> None:
    if cache_path.exists():
        shutil.rmtree(cache_path)
    else:
        logger.warning(f"Cache path does not exist: {cache_path}")
    return


def _create_cache(
    clearml_dataset: Dataset, cache_path: Path, subsets: list[str]
) -> None:
    subset_files: dict[str, list[Path]] = {subset: [] for subset in subsets}
    cache_path.mkdir(parents=True, exist_ok=True)
    clearml_dataset.get_mutable_local_copy(target_folder=cache_path)

    for subset in subset_files.keys():
        paths: list[Path] = list(cache_path.glob(f"*{subset}*.parquet"))
        if len(paths) == 0:
            logger.warning(f"No *{subset}*.parquet files found for subset: {subset}")
        subset_files[subset].extend(paths)

    metadata = {}
    for k, v in subset_files.items():
        files_count = len(v)
        metadata[k] = files_count
        if files_count == 0:
            logger.warning(f"No *{k}*.parquet files found for subset: {k}")

    data_paths: list[Path] = []
    for v in subset_files.values():
        data_paths.extend(v)

    for data_path in tqdm(
        data_paths, desc="Converting parquet files to arrow", unit="file"
    ):
        df = pl.read_parquet(data_path)
        df.write_ipc(data_path.with_suffix(".arrow"))
        data_path.unlink()

    metadata_path = cache_path / "metadata.json"
    metadata_path.write_bytes(orjson.dumps(metadata, option=orjson.OPT_INDENT_2))

    logger.info(f"Cache created at {cache_path}")
    return
