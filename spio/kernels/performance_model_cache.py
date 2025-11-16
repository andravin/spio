"""Performance model cache for kernel performance prediction."""

from pathlib import Path
from dataclasses import dataclass
from typing import List, Any, TYPE_CHECKING
import tarfile
import json
import os

from packaging import version
import requests
import xgboost as xgb
import numpy as np
from filelock import FileLock

from .. import __version__, supported_arch
from ..util import (
    get_cache_dir,
    get_formatted_device_name,
    get_formatted_arch,
    logger_enabled,
    time_function,
    get_device_ordinal,
)
from ..cuda.driver import get_device_attributes

from .kernel import Kernel

if TYPE_CHECKING:
    from .kernel_factory import KernelFactory

DEFAULT_PERF_THREADS = 4

NUM_PERF_THREADS = int(os.environ.get("SPIO_PERF_THREADS", f"{DEFAULT_PERF_THREADS}"))

PERFORMANCE_MODEL_EXTENSION = ".ubj"

USER_AGENT = f"spio/{__version__}"

RELEASES_URL = "https://api.github.com/repos/andravin/spio/releases"

RELEASES_TIMEOUT = 10  # seconds

ASSET_DOWNLOAD_TIMEOUT = 10  # seconds

CACHE_DIR = Path(get_cache_dir())

LOCK_FILE = CACHE_DIR / "spio_download.lock"

GITHUB_TOKEN_FILE = CACHE_DIR / "GITHUB_TOKEN"

PERF_CACHE_DIR = CACHE_DIR / "perf"

RELEASE_INFO_FILE = PERF_CACHE_DIR / "release_info.json"

_download_lock = FileLock(str(LOCK_FILE))

# pylint: disable=C0103
_release_info = None


def _get_github_access_token():
    """Return the GitHub access token stored in the cache directory.

    Returns None if the token is not found. The access token is only
    required when accessing a private GitHub repository.
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

    Performance models predict the latency of each kernel configuration
    for a given set of layer parameters. Because such predictions are
    accurate, they can be used to select an efficient kernel
    configuration without expensive auto-tuning. This greatly reduces
    the time to select kernels for each layer of the network.
    """

    def __init__(self):
        # _PerformanceModelKey -> xgb.Booster
        self._model_cache = {}

        # device name -> archive name
        self._archive_cache = {}

    def predict_best_kernel(
        self, kernel_factory: "KernelFactory", params: Any, device: str, **kernel_kwargs
    ) -> Kernel:
        """Return the best kernel for the kernel class and parameters.

        Returns None if no performance model is available for the given
        kernel and device.
        """
        kernel_name = kernel_factory.get_kernel_name(**kernel_kwargs)
        performance_model = self._get_performance_model(kernel_name, device)
        if performance_model is None:
            return None

        device_idx = get_device_ordinal(device)
        device_attr = get_device_attributes(device_idx)

        configs = list(kernel_factory.configs(params, device_attr, **kernel_kwargs))
        return _predict_best_config(
            performance_model,
            params,
            configs,
            skip_params=kernel_factory.per_model_skip_params,
        )

    def _get_performance_model(self, kernel_name: str, device) -> xgb.Booster:
        """Return a performance model for the kernel, device, and arch.

        Each new version of spio has a new set of performance models
        stored in the release assets. We download the listing of the
        performance model assets corresponding to the current software
        version and store it in a .json file in the cache directory.

        The performance models are stored in tar.gz archives, with one
        archive for each device and architecture.

        If the requested performance model is not in the memory cache,
        it is loaded from the disk-based cache or downloaded from the
        GitHub release.

        If the architecture is supported, then it must provide a
        performance model for every kernel. Additionally, there may be
        performance models for the device. We prefer device models if
        they exist.
        """

        device_name = get_formatted_device_name(device)
        arch = get_formatted_arch(device)

        if arch not in supported_arch:
            raise NotImplementedError(
                f"NVIDIA GPU architecture {arch} is not supported."
            )

        device_model_cache_key = _PerformanceModelKey(kernel_name, device_name)

        model = self._model_cache.get(device_model_cache_key)
        if model is None:
            archive_name = self._archive_cache.get(device_name)
            if archive_name is None:
                archive_name = _get_archive_name_for_device_from_release_info(
                    device_name, arch
                )
                self._archive_cache[device_name] = archive_name
            model_file_name = _get_model_name_from_archive(
                kernel_name, device_name, arch, archive_name
            )
            _ensure_archive_is_downloaded(archive_name)
            model_data = _load_model_from_archive(archive_name, model_file_name)
            model = xgb.Booster()
            model.load_model(model_data)
            model.set_param("n_jobs", NUM_PERF_THREADS)
            self._model_cache[device_model_cache_key] = model
            if logger_enabled:
                print(
                    f"spio: loaded perf. model for {kernel_name} on {device}/"
                    f"{device_name}:{arch} from {archive_name}."
                )
        assert (
            model is not None
        ), f"No performance model found for {kernel_name} on {device_name} / {device_name}:{arch}."
        return model


def _get_archive_name_for_device_from_release_info(device: str, arch: str) -> str:
    """Return the arch file name for the given device and arch.

    The archive file name is derived from the assets listed in the
    release info. A matching device model is preferred over an
    architecture model, but not all devices have a performance model.
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
    if arch_asset is not None:
        return arch_file
    release_version = release_info["tag"]
    raise ValueError(
        f"No performance model archive found in release {release_version} for "
        f"{device} and architecture {arch}."
    )


def _get_model_name_from_archive(
    kernel_name: str, device: str, arch: str, archive_name: str
) -> str:
    """Performance model file name for the kernel, device, and arch.

    The performance model file name is derived from the archive name and
    the kernel name.
    """

    if archive_name.startswith("devicemodel"):
        return get_device_performance_model_file_name(kernel_name, device, arch)
    if archive_name.startswith("archmodel"):
        return get_arch_performance_model_file_name(kernel_name, arch)
    assert False, f"Invalid archive name: {archive_name}"


@time_function("spio: predicting best kernel configuration", timer_log_level=2)
def _predict_best_config(
    performance_model: xgb.Booster,
    params: Any,
    configs: List[Any],
    skip_params: List[str] = None,
):
    """Return the best configuration for the given parameters.

    Uses the given XGBoost performance model to predict the latency of
    each configuration and returns the best one.
    """
    dm = _params_and_configs_to_dmatrix(params, configs, skip_params=skip_params)
    predictions = performance_model.predict(dm)
    best_config = configs[predictions.argmin()]
    return best_config


def _load_model_from_archive(archive_name: str, model_file_name: str) -> xgb.Booster:
    """Load the performance model from the tar archive."""
    archive_path = PERF_CACHE_DIR / archive_name
    with tarfile.open(archive_path, "r:gz") as tar:
        top_level = Path(archive_name).stem
        member_name = f"{top_level}/{model_file_name}"
        return bytearray(tar.extractfile(member_name).read())


def _ensure_archive_is_downloaded(archive_name: str):
    """Download the given archive.

    If it is already in the cache directory, do nothing.
    """
    archive_path = PERF_CACHE_DIR / archive_name
    if not archive_path.exists():
        if not _download_archive(archive_path, archive_name):
            raise ValueError(f"Failed to download archive {archive_name}.")


@_download_lock
def _download_archive(archive_path: str, archive_name: str) -> bool:
    """Download the archive from the GitHub release and save it.

    Acquire the download lock in case multiple processes try to download
    the same archive. Yes, FileLock is recursive:
    https://py-filelock.readthedocs.io/en/latest/
    """
    if archive_path.exists():
        # The archive was already downloaded by another process.
        return True
    # Make the performance model cache directory if it does not exist.
    if not PERF_CACHE_DIR.exists():
        PERF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
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
    """Download the asset from the GitHub release and save it."""
    asset_id = asset["id"]
    download_url = f"{RELEASES_URL}/assets/{asset_id}"
    headers = _get_http_headers()
    headers.update({"Accept": "application/octet-stream"})
    response = requests.get(
        download_url, headers=headers, stream=True, timeout=ASSET_DOWNLOAD_TIMEOUT
    )
    response.raise_for_status()
    with local_asset_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def _get_release_info():
    """Return the release info for the current software version.

    The release info is a JSON object containing the metadata for the
    latest GitHub release. Read from disk or download from the GitHub
    release if it is not already loaded.
    """
    # pylint: disable=global-statement
    global _release_info
    if _release_info is None:
        _release_info = _load_release_info()
        # If the release info is not up-to-date, download it.
        if _release_info is not None and version.parse(
            _release_info["tag_name"]
        ) != version.parse(_get_release_series_tag()):
            _clear_cache()
            _release_info = None
        if _release_info is None:
            _release_info = _download_release_info()
            _clear_cache()
            _save_release_info(_release_info)
    return _release_info


def _get_release_series_tag() -> str:
    """Ignore the PATCH number in the software version.

    Fetch release models using only the MAJOR.MINOR portion of the Spio package version.
    """
    v = version.Version(__version__)
    return f"{v.major}.{v.minor}.0"


@_download_lock
def _download_release_info():
    """Download the release info for the current software version.

    The release info is obtained from the the GitHub release. Acquire
    the download lock in case multiple processes try to download the
    release info.
    """
    # Check if the release info was already downloaded by another process.
    release_info = _load_release_info()
    if release_info is not None:
        return release_info

    # It wasn't, so download it.
    headers = _get_http_headers()
    response = requests.get(RELEASES_URL, headers=headers, timeout=RELEASES_TIMEOUT)
    response.raise_for_status()
    releases = response.json()
    if not releases:
        raise ValueError("No GitHub releases found for the Spio project.")

    # We are only interested in the release for the current software version.
    for release in releases:
        if version.parse(release["tag_name"]) == version.parse(
            _get_release_series_tag()
        ):
            return release

    raise ValueError(f"No GitHub release found for software version {__version__}.")


def _load_release_info() -> None:
    """Get the release info that is stored in the cache directory.

    Return None if it does not exist.
    """
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
    if PERF_CACHE_DIR.exists():
        for f in PERF_CACHE_DIR.iterdir():
            if f.is_file() and (
                f.name.endswith(".tgz") or f.name == "release_info.json"
            ):
                f.unlink()
    else:
        PERF_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_http_headers() -> str:
    """Return the HTTP headers for the GitHub API requests."""
    headers = {"User-Agent": USER_AGENT}
    if GITHUB_ACCESS_TOKEN is not None:
        headers["Authorization"] = f"token {GITHUB_ACCESS_TOKEN}"
    headers["Accept"] = "application/vnd.github.v3+json"
    return headers


def _get_device_archive_name(device: str, arch: str) -> str:
    """Return the tar archive name for the given device."""
    return f"devicemodel__{device}__{arch}.tgz"


def _get_arch_archive_name(arch: str) -> str:
    """Return the tar archive name for the arch."""
    return f"archmodel__{arch}.tgz"


def get_device_performance_model_file_name(
    kernel: str = None,
    device: str = None,
    arch: str = None,
    ext: str = PERFORMANCE_MODEL_EXTENSION,
) -> str:
    """Get the perf model filename for the kernel, device, and arch."""
    return f"devicemodel__{device}__{arch}__{kernel}{ext}"


def get_arch_performance_model_file_name(
    kernel: str = None,
    arch: str = None,
    ext: str = PERFORMANCE_MODEL_EXTENSION,
) -> str:
    """Get the perf model filename for the kernel, and arch."""
    return f"archmodel__{arch}__{kernel}{ext}"


def _params_and_configs_to_dmatrix(params, configs, skip_params=None):
    if skip_params is None:
        skip_params = []

    rows = []
    params_flt = _flatten_dataclass(params, skip_fields=skip_params)
    for config in configs:
        config_flt = _flatten_dataclass(config)
        rows.append(params_flt + config_flt)

    array = np.array(rows)

    feature_names = _get_dataclass_feature_names(
        params, prefix="Params", skip_fields=skip_params
    )
    feature_names += _get_dataclass_feature_names(configs[0], prefix="Config")

    dm = xgb.DMatrix(array, feature_names=feature_names)

    return dm


def _get_dataclass_feature_names(obj, prefix="", skip_fields=None):
    feature_names = []
    for field in obj.__dataclass_fields__:
        if skip_fields and field in skip_fields:
            continue
        value = getattr(obj, field)
        if isinstance(value, tuple):
            feature_names.extend([f"{prefix}_{field}_{i}" for i in range(len(value))])
        else:
            feature_names.append(f"{prefix}_{field}")
    return feature_names


def _flatten_dataclass(obj, skip_fields=None):
    features = []
    for field in obj.__dataclass_fields__:
        if skip_fields and field in skip_fields:
            continue
        value = getattr(obj, field)
        if isinstance(value, tuple):
            features.extend(value)
        elif isinstance(value, (int, float)):
            features.append(float(value))
        elif isinstance(value, str):
            features.append(value)
        else:
            raise TypeError(f"Unsupported type: {type(value)} in field {field}")
    return features
