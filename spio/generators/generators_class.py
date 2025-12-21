from typing import Iterator, Tuple, Any

from .gen_specs import GenSpecs


def _is_generator_type(obj: Any) -> bool:
    """Check if an object is a generator type that should be registered.

    Checks if the object has a _set_class_name method, which is the interface
    that generator types implement to receive their name from the Generators container.
    """
    return hasattr(obj, "_set_class_name")


class Generators:
    """A container that collects generator objects and uses attribute names as type names.

    This class provides a convenient way to define generator specifications by
    assigning them to attributes. The attribute name becomes the C++ type name
    in the generated code.

    Example:
        g = Generators()
        g.ASmem = Tensor(dtype.uint4, Dims(k16=8, i16=4, swizzle=32))
        g.BGlobal = Tensor(dtype.uint4, Dims(k=K, j=N))
        g.ComputeIndex = CompoundIndex(Dims(i=4, j=2))

        # Generate code
        code = generate(g)

    The generator objects no longer need a name as their first argument;
    the attribute name is used instead.
    """

    def __init__(self):
        """Initialize the Generators container."""
        # Use object.__setattr__ to avoid triggering our custom __setattr__
        object.__setattr__(self, "_registry", {})

    def __setattr__(self, name: str, value: Any) -> None:
        """Register a generator object when assigned to an attribute.

        If the value is a generator type (Tensor, CompoundIndex, etc.),
        it is registered with the given name. Otherwise, it's stored as
        a regular attribute.
        """
        if name.startswith("_"):
            # Private attributes are stored normally
            object.__setattr__(self, name, value)
        elif _is_generator_type(value):
            # Generator objects are registered with the name
            if hasattr(value, "_set_class_name"):
                value._set_class_name(name)
            elif hasattr(value, "class_name"):
                # For objects that use class_name directly (legacy support)
                object.__setattr__(value, "class_name", name)
            self._registry[name] = value
        else:
            # Non-generator values stored normally
            object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Any:
        """Get a registered generator by name."""
        registry = object.__getattribute__(self, "_registry")
        if name in registry:
            return registry[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __iter__(self) -> Iterator[GenSpecs]:
        """Iterate over registered generators."""
        return iter(self._registry.values())

    def __len__(self) -> int:
        """Return the number of registered generators."""
        return len(self._registry)

    def __contains__(self, name: str) -> bool:
        """Check if a generator with the given name is registered."""
        return name in self._registry

    def items(self) -> Iterator[Tuple[str, GenSpecs]]:
        """Return an iterator over (name, generator) pairs."""
        return iter(self._registry.items())

    def keys(self) -> Iterator[str]:
        """Return an iterator over registered generator names."""
        return iter(self._registry.keys())

    def values(self) -> Iterator[GenSpecs]:
        """Return an iterator over registered generators."""
        return iter(self._registry.values())
