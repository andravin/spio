def get_full_name_with_underscores(obj):
    full_name = get_full_name(obj)
    return full_name.replace(".", "_")


def get_full_name(obj):
    return f"{obj.__module__}.{obj.__name__}"
