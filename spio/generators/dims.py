"""This file implements the Dims class."""

from itertools import count
from typing import Dict, Generator, List, Tuple, Union

from .derived_dimension import DerivedDimension, SizedDerivedDimension
from .dim import StaticDim

# Type alias for dimension values: either an integer size or a derived dimension generator
DimValue = Union[int, DerivedDimension]

# Import StaticFold at runtime to avoid circular import
# Type for Dims arguments: StaticDim or StaticFold objects
DimsArg = Union[StaticDim, "StaticFold"]

# Counter for generating unique anonymous fold names
_anon_fold_counter = count()


def _counter_to_alpha(n: int) -> str:
    """Convert a counter to an alphabetic suffix (a, b, c, ..., z, aa, ab, ...)."""
    result = []
    while True:
        result.append(chr(ord("a") + n % 26))
        n = n // 26 - 1
        if n < 0:
            break
    return "".join(reversed(result))


def _is_static_dim_or_fold(arg) -> bool:
    """Check if arg is a StaticDim or StaticFold."""
    # Import here to avoid circular import
    from .fold import StaticFold

    return isinstance(arg, (StaticDim, StaticFold))


def _is_derived_dim_dict(arg) -> bool:
    """Check if arg is a single-item dict with a DerivedDimension value.

    This allows syntax like: Dims(I(16), dict(swizzle=Checkerboard(...)))
    """
    if not isinstance(arg, dict):
        return False
    if len(arg) != 1:
        return False
    value = next(iter(arg.values()))
    return isinstance(value, DerivedDimension)


def _is_valid_dims_arg(arg) -> bool:
    """Check if arg is a valid Dims positional argument."""
    return _is_static_dim_or_fold(arg) or _is_derived_dim_dict(arg)


def _parse_dim_name_and_fold(name: str) -> Tuple[str, int]:
    """Parse a dimension name into base name and fold factor.

    Args:
        name: Dimension name like 'K8', 'K4', 'K', 'I32', etc.

    Returns:
        Tuple of (base_name, fold_factor). fold_factor is 1 if not specified.
    """
    fold_factor = 1
    base_name = name
    for i, char in enumerate(name):
        if char.isdigit():
            fold_factor = int(name[i:])
            base_name = name[:i]
            break
    return base_name, fold_factor


# Type for dimension entry: (name, size, fold_factor, group_key)
# group_key is optional - if None, grouping uses parsed base name
DimEntry = Tuple[str, DimValue, int, object]


def _compute_fold_sizes(
    dim_entries: List[DimEntry],
) -> Dict[str, DimValue]:
    """Compute automatic fold sizes for dimensions specified as -1.

    Args:
        dim_entries: List of (name, size, fold_factor, group_key) tuples.
            Size of -1 indicates automatic computation based on fold factor ratios.
            group_key is used to group related dimensions (e.g., K and K8 both
            derived from the same Dim). If None, grouping uses the parsed base name.

    Returns:
        Dictionary with all -1 sizes replaced by computed values.
        The order of entries is preserved from the input.

    Raises:
        ValueError: If the coarsest fold has size -1, or if an explicit size
            doesn't match the computed value.
    """
    # Build result dict in input order first (preserves insertion order)
    result: Dict[str, DimValue] = {name: size for name, size, _, _ in dim_entries}

    # Group dimensions by group_key (or parsed base name if None)
    base_dim_folds: Dict[object, List[Tuple[str, int, DimValue]]] = {}
    for name, size, fold_factor, group_key in dim_entries:
        if group_key is None:
            group_key, _ = _parse_dim_name_and_fold(name)
        if group_key not in base_dim_folds:
            base_dim_folds[group_key] = []
        base_dim_folds[group_key].append((name, fold_factor, size))

    # Compute auto-sizes for each group
    for base_name, folds in base_dim_folds.items():
        # Sort by fold factor descending (coarsest first)
        folds_sorted = sorted(folds, key=lambda x: x[1], reverse=True)

        # Coarsest fold must have explicit size
        coarsest_name, coarsest_fold, coarsest_size = folds_sorted[0]
        if coarsest_size == -1:
            raise ValueError(
                f"Coarsest fold '{coarsest_name}' (fold factor {coarsest_fold}) "
                f"must have an explicit size, not -1."
            )

        # Compute sizes for remaining folds
        for i in range(1, len(folds_sorted)):
            name, fold_factor, specified_size = folds_sorted[i]
            prev_name, prev_fold, _prev_size = folds_sorted[i - 1]

            # Computed size = ratio of fold factors
            computed_size = prev_fold // fold_factor

            if prev_fold % fold_factor != 0:
                raise ValueError(
                    f"Fold factor {prev_fold} of '{prev_name}' is not divisible "
                    f"by fold factor {fold_factor} of '{name}'."
                )

            if specified_size == -1:
                result[name] = computed_size
            elif specified_size != computed_size:
                raise ValueError(
                    f"Dimension '{name}' has explicit size {specified_size}, "
                    f"but computed size is {computed_size} "
                    f"(from {prev_fold}/{fold_factor})."
                )

    return result


class Dims:
    """A class to represent the dimensions of a tensor.

    Can be initialized with either:
    - StaticDim objects: Dims(I(16), K(32))
    - Keyword arguments (legacy): Dims(i=16, k=32)
    """

    def __init__(self, *args: DimsArg, **dims: Dict[str, DimValue]):
        """Initialize the Dims object with the given dimensions and sizes.

        Args:
            *args: StaticDim objects specifying dimensions with their sizes.
                Example: Dims(I(16), K(32)) where I and K are Dim objects.

            **dims: (Legacy) Keyword arguments representing the dimensions.
                Each dimension is specified as a name-value pair.

                Dimensions with size -1 will have their sizes computed automatically
                based on fold factor ratios. The coarsest fold must have an explicit size.

                Names can include fold factors (e.g., 'k8', 'k4', 'k'). K8 means a fold
                of dimension K with fold factor 8.

                Names are automatically normalized to upper-case.

        Raises:
            ValueError: If both positional and keyword arguments are provided.

        Example:
            # Using StaticDim objects (preferred)
            I = Dim()
            K = Dim()
            dims = Dims(I(16), K(32))

            # Using StaticFold objects
            K8 = K.fold(8)
            dims = Dims(K(32), K8(4))

            # Using keyword arguments (legacy)
            dims = Dims(i=16, k=32)

            # Using StaticDim with derived dimensions via keyword (preferred)
            dims = Dims(I(16), K(32), swizzle=Checkerboard(...))

            # Using StaticDim with derived dimensions via dict (when ordering matters)
            # Use dict when the derived dim must appear before a positional arg
            dims = Dims(I(16), dict(swizzle=Checkerboard(...)), K(32))
        """
        # Check if kwargs contains only DerivedDimension values (allowed with positional args)
        derived_dims_from_kwargs = {
            k: v for k, v in dims.items() if isinstance(v, DerivedDimension)
        }
        regular_dims = {
            k: v for k, v in dims.items() if k not in derived_dims_from_kwargs
        }

        if args and regular_dims:
            raise ValueError(
                "Cannot mix positional StaticDim/StaticFold arguments with keyword arguments. "
                "Use either Dims(I(16), K(32)) or Dims(i=16, k=32), not both."
            )

        self._dim_args: Tuple[DimsArg, ...] = None
        self._dims_cache: Dict[str, DimValue] = None
        # Mapping from cached key to Dim/Fold object for name resolution
        self._key_to_dim_or_fold: Dict[str, object] = {}
        # Collect derived dims from both kwargs and positional dict args
        self._derived_dims: List[Tuple[str, DerivedDimension]] = []
        self._derived_dims_added: bool = False

        if args:
            # New-style: positional StaticDim/StaticFold arguments (and dict for derived dims)
            # Separate static dims from derived dim dicts, preserving order
            static_args = []
            for arg in args:
                if _is_derived_dim_dict(arg):
                    # Extract name and value from single-item dict
                    name, value = next(iter(arg.items()))
                    self._derived_dims.append((name.upper(), value))
                elif _is_static_dim_or_fold(arg):
                    static_args.append(arg)
                else:
                    raise TypeError(
                        f"Expected StaticDim, StaticFold, or dict(name=DerivedDimension), "
                        f"got {type(arg).__name__}. "
                        f"Use Dim()(size), Fold()(size), or dict(swizzle=Checkerboard(...))."
                    )
            self._dim_args = tuple(static_args) if static_args else None

            # Also add derived dims from kwargs
            for name, value in derived_dims_from_kwargs.items():
                self._derived_dims.append((name.upper(), value))

        elif regular_dims:
            # Legacy: keyword arguments - resolve immediately
            # Convert to list of (name, size, fold_factor, group_key) tuples
            # group_key=None means use parsed base name for grouping
            entries = []
            for name, size in regular_dims.items():
                name_upper = name.upper()
                _, fold_factor = _parse_dim_name_and_fold(name_upper)
                entries.append((name_upper, size, fold_factor, None))
            self._dims_cache = _compute_fold_sizes(entries)

            # Add derived dims from kwargs
            for name, value in derived_dims_from_kwargs.items():
                self._derived_dims.append((name.upper(), value))

        elif derived_dims_from_kwargs:
            # Only derived dims provided via kwargs
            for name, value in derived_dims_from_kwargs.items():
                self._derived_dims.append((name.upper(), value))

        # Note: derived dimensions are added lazily in _dims property
        # This ensures positional args are processed first

    @property
    def _dims(self) -> Dict[str, DimValue]:
        """Lazily resolve dimension names and build the dims dict."""
        if self._dims_cache is None:
            # Resolve from StaticDim/StaticFold args - convert to list of tuples
            # and reuse _compute_fold_sizes for auto-size computation
            if self._dim_args:
                entries = self._static_args_to_entries()
                self._dims_cache = _compute_fold_sizes(entries)
            else:
                self._dims_cache = {}

        # Add derived dimensions (like Checkerboard) to the cache
        if self._derived_dims and not self._derived_dims_added:
            for name, value in self._derived_dims:
                self._dims_cache[name] = value
            self._derived_dims_added = True

        return self._dims_cache

    def _static_args_to_entries(self) -> List[DimEntry]:
        """Convert StaticDim/StaticFold args to entries for _compute_fold_sizes.

        Returns:
            List of (name, size, fold_factor, group_key) tuples.
        """
        from .fold import StaticFold, Fold
        from .dim import Dim

        self._validate_no_duplicates()

        entries: List[DimEntry] = []
        seen_names: set = set()

        for arg in self._dim_args:
            dim_name = arg.dim_name
            dim_or_fold = arg.fold if isinstance(arg, StaticFold) else arg.dim

            # Auto-name anonymous folds/dims
            if dim_name is None:
                prefix = "_AnonFold" if isinstance(arg, StaticFold) else "_AnonDim"
                suffix = _counter_to_alpha(next(_anon_fold_counter))
                generated_name = f"{prefix}{suffix}".upper()
                dim_or_fold._set_class_name(generated_name)
                dim_name = generated_name

            if dim_name in seen_names:
                raise ValueError(f"Duplicate dimension name: {dim_name}")
            seen_names.add(dim_name)

            # Track the Dim/Fold object for this key so we can look up current name later
            self._key_to_dim_or_fold[dim_name] = dim_or_fold

            # Get fold factor and base Dim for grouping
            if isinstance(arg, StaticFold):
                fold_factor = arg.fold.stride
                # Group by base Dim object identity
                base_dim = arg.fold.dim
                group_key = id(base_dim) if not isinstance(base_dim, str) else base_dim
            else:
                fold_factor = 1
                # StaticDim: group by self
                group_key = id(arg.dim)

            entries.append((dim_name, arg.size, fold_factor, group_key))

        return entries

    def _validate_no_duplicates(self) -> None:
        """Validate that no dimension or fold appears twice in the args.

        Raises:
            ValueError: If the same Dim is used twice as StaticDim, or the same
                Fold is used twice as StaticFold.
        """
        from .fold import StaticFold

        # Track base dims used as StaticDim (not folds) to detect duplicates
        seen_base_dims: dict = {}  # id(dim) -> arg that used it
        # Track folds used as StaticFold to detect duplicates
        seen_folds: dict = {}  # (id(dim), stride) -> arg that used it

        for arg in self._dim_args:
            # Check for duplicate base dimensions (same Dim used twice as StaticDim)
            if isinstance(arg, StaticDim):
                base_dim_id = id(arg.dim)
                if base_dim_id in seen_base_dims:
                    prev_arg = seen_base_dims[base_dim_id]
                    dim_desc = arg.dim.dim_name or "anonymous Dim"
                    raise ValueError(
                        f"The same dimension '{dim_desc}' appears twice in Dims. "
                        f"Each base dimension can only appear once (sizes {prev_arg.size} and {arg.size}). "
                        f"Did you mean to use a fold? E.g., dim.fold(stride)(size)"
                    )
                seen_base_dims[base_dim_id] = arg

            # Check for duplicate folds (same Fold used twice as StaticFold)
            elif isinstance(arg, StaticFold):
                fold_key = arg._identity_key()  # (id(dim), stride)
                if fold_key in seen_folds:
                    prev_arg = seen_folds[fold_key]
                    fold_desc = (
                        arg.fold.fold_name or f"fold with stride {arg.fold.stride}"
                    )
                    raise ValueError(
                        f"The same fold '{fold_desc}' appears twice in Dims. "
                        f"Each fold can only appear once (sizes {prev_arg.size} and {arg.size})."
                    )
                seen_folds[fold_key] = arg

    def _resolve_current_name(self, cached_key: str) -> str:
        """Resolve the current name for a cached key.

        If the key maps to a Dim/Fold object that has been renamed,
        return the current name. Otherwise return the original key.
        """
        dim_or_fold = self._key_to_dim_or_fold.get(cached_key)
        if dim_or_fold is not None:
            current_name = dim_or_fold.get_class_name()
            if current_name is not None:
                return current_name
        return cached_key

    def items(self) -> Generator[Tuple[str, DimValue], None, None]:
        """Get the dimensions as a generator of (name, value) pairs."""
        for key, value in self._dims.items():
            yield self._resolve_current_name(key), value

    def items_with_cached_key(
        self,
    ) -> Generator[Tuple[str, str, DimValue], None, None]:
        """Get dimensions as (cached_key, current_name, value) tuples.

        Used by code generators that need both the cached key (for stride lookup)
        and the current name (for generated code).
        """
        for key, value in self._dims.items():
            yield key, self._resolve_current_name(key), value

    def keys(self) -> Generator[str, None, None]:
        """Get the names of the dimensions."""
        for key in self._dims.keys():
            yield self._resolve_current_name(key)

    def values(self) -> Generator[DimValue, None, None]:
        """Get the values of the dimensions."""
        return self._dims.values()

    def __getitem__(self, key) -> DimValue:
        """Get the value of a dimension by its name."""
        return self._dims[key]

    def __contains__(self, key) -> bool:
        """Check if a dimension is present in the Dims object."""
        return key in self._dims

    def iter_args(self) -> Generator[DimsArg, None, None]:
        """Iterate over the original StaticDim/StaticFold arguments.

        Returns the original objects in order, enabling object-based matching.
        For legacy keyword-based Dims, this returns an empty generator.
        """
        # Trigger resolution to ensure anonymous folds/dims get named
        _ = self._dims

        if self._dim_args is not None:
            yield from self._dim_args

    def has_object_args(self) -> bool:
        """Return True if this Dims was created with StaticDim/StaticFold objects."""
        return self._dim_args is not None

    def used_generators(self) -> list:
        """Return Dim and Fold objects used by this Dims instance.

        This is used to find anonymous generators that need auto-generated names.
        Only returns objects from the new StaticDim/StaticFold path; the legacy
        keyword path doesn't have access to the original Dim/Fold objects.
        """
        from .fold import StaticFold

        if self._dim_args is None:
            return []

        # Trigger resolution to ensure anonymous folds/dims get named
        _ = self._dims

        result = []
        for arg in self._dim_args:
            if isinstance(arg, StaticFold):
                # Include the Fold and its base Dim (if it's a Dim object)
                result.append(arg.fold)
                if not isinstance(arg.fold.dim, str):
                    result.append(arg.fold.dim)
            else:
                # StaticDim: include the Dim
                result.append(arg.dim)
        return result


class Strides:
    """A class to represent the strides of a tensor.

    Can be initialized with either:
    - StaticDim/StaticFold objects: Strides(I(64), J(1))
    - Keyword arguments (legacy): Strides(i=64, j=1)
    """

    def __init__(self, *args: DimsArg, **strides: Dict[str, int]):
        """Initialize the Strides object with the given strides.

        Args:
            *args: StaticDim or StaticFold objects specifying strides.
                Example: Strides(I(64)) where I is a Dim object.

            **strides: (Legacy) Keyword arguments representing the strides.
                Each stride is specified as a name-value pair.

        Raises:
            ValueError: If both positional and keyword arguments are provided.
        """
        if args and strides:
            raise ValueError(
                "Cannot mix positional StaticDim/StaticFold arguments with keyword arguments. "
                "Use either Strides(I(64)) or Strides(i=64), not both."
            )

        self._stride_args: Tuple[DimsArg, ...] = None
        self._strides_cache: Dict[str, int] = None

        if args:
            # New-style: positional StaticDim/StaticFold arguments
            for arg in args:
                if not _is_static_dim_or_fold(arg):
                    raise TypeError(
                        f"Expected StaticDim or StaticFold, got {type(arg).__name__}. "
                        f"Use Dim()(stride) or Fold()(stride), e.g., I(64)."
                    )
            self._stride_args = args
        else:
            # Legacy: keyword arguments - resolve immediately
            self._strides_cache = {key.upper(): value for key, value in strides.items()}

    @property
    def _strides(self) -> Dict[str, int]:
        """Lazily resolve stride names and build the strides dict."""
        if self._strides_cache is not None:
            return self._strides_cache

        # Resolve from StaticDim/StaticFold args
        from .fold import StaticFold

        self._strides_cache = {}
        for arg in self._stride_args:
            dim_name = arg.dim_name

            # For unnamed folds of named dims, derive dict key from base dim + stride
            if dim_name is None and isinstance(arg, StaticFold):
                fold = arg.fold
                base_name = fold.dim_name  # e.g., "J" from J.fold(8)
                if base_name is not None:
                    dim_name = f"{base_name}{fold.stride}"  # e.g., "J8" - just the key, don't set on fold

            if dim_name is None:
                raise ValueError(
                    "Dim/Fold must have a name set before using in Strides. "
                    "Assign to a Generators attribute, use a named Dim, or "
                    "create with an explicit name: Dim('I') or Fold('K', 8, 'K8')."
                )
            if dim_name in self._strides_cache:
                raise ValueError(f"Duplicate stride name: {dim_name}")
            self._strides_cache[dim_name] = arg.size

        return self._strides_cache

    def items(self) -> Generator[Tuple[str, int], None, None]:
        """Get the strides as a generator of (name, value) pairs."""
        return self._strides.items()

    def keys(self) -> Generator[str, None, None]:
        """Get the names of the strides."""
        return self._strides.keys()

    def values(self) -> Generator[int, None, None]:
        """Get the values of the strides."""
        return self._strides.values()

    def __getitem__(self, key) -> int:
        """Get the value of a stride by its name."""
        return self._strides[key]

    def __contains__(self, key) -> bool:
        """Check if a stride is present in the Strides object."""
        return key in self._strides

    def get_by_object(self, dim_arg: DimsArg) -> int:
        """Get stride by StaticDim/StaticFold object identity.

        This enables matching without relying on string names.

        Args:
            dim_arg: A StaticDim or StaticFold to find stride for.

        Returns:
            The stride value if found, None otherwise.
        """
        if self._stride_args is None:
            return None  # Legacy mode, no object matching

        for arg in self._stride_args:
            if arg == dim_arg:  # Uses __eq__ based on object identity
                return arg.size
        return None

    def has_object_args(self) -> bool:
        """Return True if this Strides was created with StaticDim/StaticFold objects."""
        return self._stride_args is not None


def compute_full_strides(
    dims: Dims,
    given_strides: Strides,
) -> Dict[str, int]:
    """Compute the full strides for the given dimensions.

    Uses object-based matching when both dims and strides were created with
    StaticDim/StaticFold objects. Falls back to string-based matching for
    legacy keyword syntax.
    """
    if given_strides is None:
        given_strides = Strides()  # Empty Strides

    all_strides = {}
    stride = 1

    # Prefer object-based matching if both sides support it
    use_object_matching = dims.has_object_args() and (
        given_strides.has_object_args() or not given_strides._strides
    )

    if use_object_matching:
        # Object-based: iterate dims in reverse, match strides by object identity
        dim_args_list = list(dims.iter_args())
        # Get resolved sizes from the dims dict (handles -1 auto-computation)
        resolved_sizes = dims._dims

        # Build combined list: first the static dims, then derived dims (in that order)
        # When reversed, derived dims are processed first (they have the highest stride),
        # then static dims in reverse order
        items_to_process = []

        # Static dims from iter_args
        for arg in dim_args_list:
            dim_name = arg.dim_name
            if dim_name is None:
                # Anonymous - derive name from object
                from .fold import StaticFold

                if isinstance(arg, StaticFold):
                    base_name = arg.fold.dim_name
                    if base_name is not None:
                        dim_name = f"{base_name}{arg.fold.stride}"
            items_to_process.append((dim_name, arg, resolved_sizes.get(dim_name)))

        # Derived dimensions - these come after static dims
        for name, value in dims._derived_dims:
            items_to_process.append(
                (name, None, value)
            )  # None = no object for stride lookup

        for dim_name, arg, dim_value in reversed(items_to_process):
            # Try object-based stride lookup (only for static dims)
            if arg is not None:
                explicit_stride = given_strides.get_by_object(arg)
                if explicit_stride is not None:
                    stride = explicit_stride
            else:
                # Derived dimension - try string-based lookup
                if dim_name in given_strides:
                    stride = given_strides[dim_name]

            all_strides[dim_name] = stride

            # Compute the default stride of the next dimension
            if isinstance(dim_value, SizedDerivedDimension):
                dim_size = dim_value.size
            else:
                dim_size = dim_value
            stride *= dim_size
    else:
        # String-based fallback for legacy keyword syntax
        for name, value in reversed(list(dims.items())):
            if name in given_strides:
                stride = given_strides[name]
            all_strides[name] = stride
            # Compute the default stride of the next dimension.
            if isinstance(value, SizedDerivedDimension):
                dim_size = value.size
            else:
                dim_size = value
            stride *= dim_size

    return all_strides
