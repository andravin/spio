"""Function for accessin the spio cache directory."""

import appdirs


def get_cache_dir() -> str:
    """Get the cache directory for spio."""
    return appdirs.user_cache_dir("spio")
