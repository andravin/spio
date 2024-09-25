from pathlib import Path
from dataclasses import dataclass
import gzip
import requests
from typing import Type, List, Any, Dict
from packaging import version
import tarfile
import json

import xgboost as xgb
import torch
from filelock import FileLock

from .. import primary_context_guard, __version__, supported_arch
from ..util import (
    get_cache_dir,
    params_and_configs_to_dataframe,
    get_formatted_device_name,
    get_formatted_arch,
)
from ..compiler import compile_kernel_configs
from .kernel import Kernel

PERFORMANCE_MODEL_EXTENSION = ".ubj"

USER_AGENT = f"spio/{__version__}"

RELEASES_URL = "https://api.github.com/repos/andravin/spio/releases"

CACHE_DIR = get_cache_dir()

LOCK_FILE = str(Path(CACHE_DIR, "spio_download.lock"))

GITHUB_TOKEN_FILE = Path(CACHE_DIR, "GITHUB_TOKEN")

RELEASE_INFO_FILE = Path(CACHE_DIR, "release_info.json")

_download_lock = FileLock(LOCK_FILE)

_release_info = None


def _get_github_access_token():
    """Return the GitHub access token stored in the cache directory, or None if it does not exist.

    The access token is only required when accessing a private GitHub repository.
    """
    if not GITHUB_TOKEN_FILE.exists():
        return None
    with GITHUB_TOKEN_FILE.open("r") as f:
        return f.read().strip()


GITHUB_ACCESS_TOKEN = _get_github_access_token()


@dataclass(frozen=True)
class _PerformanceModelKey:
    """A key for the performance model cache."""

    kernel_name: str
    device_name: str


class PerformanceModelCache:
    """A container for kernel performance models.

    Performance models predict the latency of each kernel configuration for a given set of layer parameters.
    Because such predictions are accurate, they can be used to select an efficient kernel configuration without
    expensive auto-tuning. This greatly reduces the time to select kernels for each layer of the network.
    """

    def __init__(self):
        # _PerformanceModelKey -> xgb.Booster
        self._model_cache = {}

        # device -> archive name
        self._archive_cache = {}

    def predict_best_kernel(
        self, kernel_cls: Type[Kernel], params: List[Any], device: str, **kernel_kwargs
    ) -> Kernel:
        """Return the best kernel for the given kernel class and layer parameters.

        Returns None if no performance model is available for the given kernel and device.
        """
        kernel_name = kernel_cls.get_kernel_name(**kernel_kwargs)
        device_name = get_formatted_device_name(device)
        arch = get_formatted_arch(device)
        performance_model = self._get_performance_model(kernel_name, device_name, arch)
        if performance_model is None:
            return None

        configs = list(kernel_cls.configs(params))
        best_config = _predict_best_config(performance_model, params, configs)
        if best_config is None:
            return None

        with torch.device(device) as device_obj:
            device_ordinal = device_obj.index if device_obj.index is not None else 0
            arch = torch.cuda.get_device_capability(device=device_obj)
            primary_context_guard.set_device(device_ordinal)
            configs = [best_config]
            kernels = compile_kernel_configs(
                kernel_cls, params, configs=configs, arch=arch, **kernel_kwargs
            )
            best_kernel = kernels[0]
            device_ordinal = device_obj.index if device_obj.index is not None else 0
            best_kernel.load(device_ordinal=device_ordinal)
            return best_kernel

    def _get_performance_model(
        self, kernel_name: str, device: str, arch: str
    ) -> xgb.Booster:
        """Return the performance model for the given kernel, device, and architecture.

        Each new version of spio has a new set of performance models stored in the release assets.
        We download the listing of the performance model assets corresponding to the current software version
        and store it in a .json file in the cache directory.

        The performance models are stored in tar.gz archives, with one archive for each device and architecture.

        If the requested performance model is not in the memory cache, it is loaded from the disk-based cache or
        downloaded from the GitHub release.

        If the architecture is supported, then it must provide a performance model for every kernel.
        Additionally, there may be a performance models for the device. We prefer the device model if it exists.

        Basic control flow:

        1. Check if a performance model has been loaded for the given kernel and device.
        2. If not, check if a model archive has been downloaded for the given device.
        3. If not, ensure the release_info.json has been downloaded for the current release ..
        4. .. and try to download the device or architecture model archive from the latest GitHub release.
        5. Load the performance model from the archive and store it in the cache.
        """

        if arch not in supported_arch:
            raise NotImplementedError(
                f"NVIDIA GPU architecture {arch} is not supported."
            )

        device_model_cache_key = _PerformanceModelKey(kernel_name, device)

        model = self._model_cache.get(device_model_cache_key)
        if model is None:
            archive_name = self._archive_cache.get(device)
            if archive_name is None:
                archive_name = _get_archive_name_for_device_from_release_info(
                    device, arch
                )
                self._archive_cache[device] = archive_name
            model_file_name = _get_model_name_from_archive(
                kernel_name, device, arch, archive_name
            )
            _ensure_archive_is_downloaded(archive_name)
            model_data = _load_model_from_archive(archive_name, model_file_name)
            model = xgb.Booster()
            model.load_model(model_data)
            self._model_cache[device_model_cache_key] = model

        assert (
            model is not None
        ), f"No performance model found for kernel {kernel_name}, device {device}, and architecture {arch}."
        return model


def _get_archive_name_for_device_from_release_info(device: str, arch: str) -> str:
    """Return the archive file name for the given device and architecture.

    The archive file name is derived from the assets listed in the release info.
    A matching device model is preferred over an architecture model, but not all
    devices have a performance model.
    """
    release_info = _get_release_info()
    device_file = _get_device_archive_name(device, arch)
    arch_file = _get_arch_archive_name(arch)
    device_asset = None
    arch_asset = None
    for asset in release_info["assets"]:
        if asset["name"] == device_file:
            device_asset = asset
        elif asset["name"] == arch_file:
            arch_asset = asset
    if device_asset is not None:
        return device_file
    elif arch_asset is not None:
        return arch_file
    else:
        release_version = release_info["tag"]
        return ValueError(
            f"No performance model archive found in release {release_version} for {device} and architecture {arch}."
        )


def _get_model_name_from_archive(
    kernel_name: str, device: str, arch: str, archive_name: str
) -> str:
    """Return the performance model file name for the given kernel, device, and architecture.

    The performance model file name is derived from the archive name and the kernel name.
    """

    if archive_name.startswith("devicemodel"):
        return get_device_performance_model_file_name(kernel_name, device, arch)
    elif archive_name.startswith("archmodel"):
        return get_arch_performance_model_file_name(kernel_name, arch)
    else:
        assert False, f"Invalid archive name: {archive_name}"


def _predict_best_config(
    performance_model: xgb.Booster, params: List[Any], configs: List[Any]
):
    """Return the best configuration for the given parameters.

    Uses the given XGBoost performance model to predict the latency of each configuration
    and returns the best one.
    """
    df = params_and_configs_to_dataframe(params, configs)
    dm = xgb.DMatrix(df)
    predictions = performance_model.predict(dm)
    best_config = configs[predictions.argmin()]
    return best_config


def _load_model_from_archive(archive_name: str, model_file_name: str) -> xgb.Booster:
    """Load the performance model from the tar archive."""
    archive_path = Path(CACHE_DIR, archive_name)
    with tarfile.open(archive_path, "r:gz") as tar:
        top_level = Path(archive_name).stem
        member_name = f"{top_level}/{model_file_name}"
        return bytearray(tar.extractfile(member_name).read())


def _ensure_archive_is_downloaded(archive_name: str):
    """Download the given archive if it has not already been downloaded."""
    archive_path = Path(CACHE_DIR, archive_name)
    if not archive_path.exists():
        if not _download_archive(archive_path, archive_name):
            raise ValueError(f"Failed to download archive {archive_name}.")


@_download_lock
def _download_archive(archive_path: str, archive_name: str) -> bool:
    """Download the archive from the GitHub release and save it to the disk-based cache.

    Acquire the download lock in case multiple processes try to download the same archive.
    Yes, FileLock is recursive: https://py-filelock.readthedocs.io/en/latest/
    """
    if archive_path.exists():
        # The archive was already downloaded by another process.
        return True
    release_info = _get_release_info()
    asset = None
    for a in release_info["assets"]:
        if a["name"] == archive_name:
            asset = a
            break
    if asset is None:
        return False
    _download_asset(asset, archive_path)
    return True


def _download_asset(asset, local_asset_path: Path):
    """Download the asset from the GitHub release and save it to the disk-base cache."""
    asset_id = asset["id"]
    download_url = f"{RELEASES_URL}/assets/{asset_id}"
    headers = _get_http_headers()
    headers.update({"Accept": "application/octet-stream"})
    response = requests.get(download_url, headers=headers, stream=True)
    response.raise_for_status()
    with local_asset_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def _get_release_info():
    """Return the release info for the current software version.

    The release info is a JSON object containing the metadata for the latest GitHub release.
    Read from disk or download from the GitHub release if it is not already loaded.
    """
    global _release_info
    if _release_info is None:
        _release_info = _load_release_info()
        if _release_info is None:
            _release_info = _download_release_info()
            _clear_cache()
            _save_release_info(_release_info)
    return _release_info


@_download_lock
def _download_release_info():
    """Download the release info for the current software version from the GitHub release.

    Acquire the download lock in case multiple processes try to download the release info.
    """
    # Check if the release info was already downloaded by another process.
    release_info = _load_release_info()
    if release_info is not None:
        return release_info

    # It wasn't, so download it.
    headers = _get_http_headers()
    response = requests.get(RELEASES_URL, headers=headers)
    response.raise_for_status()
    releases = response.json()
    if not releases:
        raise ValueError("No GitHub releases found for the Spio project.")

    # We are only interested in the release for the current software version.
    for release in releases:
        if version.parse(release["tag_name"]) == version.parse(__version__):
            return release

    raise ValueError(f"No GitHub release found for software version {__version__}.")


def _load_release_info() -> None:
    """Return the release info that is stored in the cache directory, or None if it does not exist."""
    if RELEASE_INFO_FILE.exists():
        with RELEASE_INFO_FILE.open("r") as f:
            return json.load(f)
    return None


def _save_release_info(release_info) -> None:
    """Save the given release info to the cache directory."""
    with RELEASE_INFO_FILE.open("w") as f:
        json.dump(release_info, f, indent=4)


def _clear_cache() -> None:
    """Clear all cached files in the cache directory."""
    cache_dir = Path(CACHE_DIR)
    for f in cache_dir.iterdir():
        if f.is_file() and (f.name.endswith(".tgz") or f.name == "release_info.json"):
            f.unlink()


def _get_http_headers() -> str:
    """Return the HTTP headers for the GitHub API requests."""
    headers = {"User-Agent": USER_AGENT}
    if GITHUB_ACCESS_TOKEN is not None:
        headers["Authorization"] = f"token {GITHUB_ACCESS_TOKEN}"
    headers["Accept"] = "application/vnd.github.v3+json"
    return headers


def _get_device_archive_name(device: str, arch: str) -> str:
    """Return the device archive name for the given device."""
    return f"devicemodel__{device}__{arch}.tgz"


def _get_arch_archive_name(arch: str) -> str:
    """Return the architecture archive name for the given architecture."""
    return f"archmodel__{arch}.tgz"


def get_device_performance_model_file_name(
    kernel: str = None,
    device: str = None,
    arch: str = None,
    ext: str = PERFORMANCE_MODEL_EXTENSION,
) -> str:
    """Return the performance model filename for the given kernel, device, and architecture."""
    return f"devicemodel__{device}__{arch}__{kernel}{ext}"


def get_arch_performance_model_file_name(
    kernel: str = None,
    arch: str = None,
    ext: str = PERFORMANCE_MODEL_EXTENSION,
) -> str:
    """Return the performance model filename for the given kernel, device, and architecture."""
    return f"archmodel__{arch}__{kernel}{ext}"
