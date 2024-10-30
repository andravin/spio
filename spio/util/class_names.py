"""Functions for getting the class names from objects."""


def get_full_name_with_underscores(obj) -> str:
    """Full name of an object with underscores instead of dots."""
    full_name = get_full_name(obj)
    return full_name.replace(".", "_")


def get_full_name(obj) -> str:
    """Full class name of an object.

    Format: module_name.class_name
    """
    return f"{obj.__module__}.{obj.__name__}"
