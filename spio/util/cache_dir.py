import appdirs


def get_cache_dir():
    return appdirs.user_cache_dir("spio")
