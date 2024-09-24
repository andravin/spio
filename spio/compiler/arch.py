def sm_from_arch(arch):
    if isinstance(arch, tuple):
        return f"sm_{arch[0]}{arch[1]}"
    else:
        return arch
