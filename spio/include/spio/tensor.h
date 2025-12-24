#ifndef SPIO_TENSOR_VARIADIC_H_
#define SPIO_TENSOR_VARIADIC_H_

#include "spio/macros.h"
#include "spio/dim.h"
#include "spio/dim_info.h"
#include "spio/compound_index.h"
#include "spio/allocator.h"
#include "spio/coordinates.h"

namespace spio {

    /// @brief Base class for tensor data.
    template <typename _data_type> class Data {
    public:
        using data_type = _data_type;
        static constexpr int element_size = sizeof(_data_type);

        DEVICE Data(_data_type* data = nullptr) : _data(data) {}

        DEVICE _data_type* get() const {
            return _data;
        }

        DEVICE void reset(_data_type* data) {
            _data = data;
        }

        DEVICE _data_type& operator*() const {
            return *_data;
        }

        DEVICE _data_type* operator->() const {
            return _data;
        }

    private:
        _data_type* _data;
    };

    // Forward declarations
    template <typename DataType, typename... Params> class Tensor;
    template <typename DataType, typename... Params> class Cursor;
    template <typename DataType, typename... Params> class BaseCursor;

    // Implementation details
    namespace detail {

        /// @brief Update dimension info by replacing a given dimension with a new size.
        template <typename DimType, int SliceSize, typename Tuple, typename Result = tuple<>>
        struct update_dim_info;

        // Base case: empty input, return accumulated result
        template <typename DimType, int SliceSize, typename Result>
        struct update_dim_info<DimType, SliceSize, tuple<>, Result> {
            using type = Result;
        };

        // Recursive case
        template <typename DimType, int SliceSize, typename FirstInfo, typename... RestInfos,
                  typename... ResultTypes>
        struct update_dim_info<DimType, SliceSize, tuple<FirstInfo, RestInfos...>,
                               tuple<ResultTypes...>> {
            static constexpr bool is_match = is_same<DimType, typename FirstInfo::dim_type>::value;
            using updated_info =
                conditional_t<is_match,
                              DimInfo<typename FirstInfo::dim_type, SliceSize, FirstInfo::stride>,
                              FirstInfo>;
            using type = typename update_dim_info<DimType, SliceSize, tuple<RestInfos...>,
                                                  tuple<ResultTypes..., updated_info>>::type;
        };

        template <typename DimType, int SliceSize, typename... DimInfos>
        using update_dim_info_t =
            typename update_dim_info<DimType, SliceSize, tuple<DimInfos...>>::type;

        // Tuple-based version of update_dim_info
        template <typename DimType, int SliceSize, typename InfosTuple>
        using tuple_update_dim_info_t =
            typename update_dim_info<DimType, SliceSize, InfosTuple>::type;

        template <typename, typename> struct tensor_type_from_dim_info_tuple;

        template <typename DataType, typename... DimInfos>
        struct tensor_type_from_dim_info_tuple<DataType, tuple<DimInfos...>> {
            using tensor_type = Tensor<DataType, DimInfos...>;
        };

        // Helper to calculate maximum storage size needed with strides
        template <typename... DimInfos> struct calculate_storage_size;

        // Base case
        template <> struct calculate_storage_size<> {
            static constexpr int value = 1; // No dimensions, just one element
        };

        // Recursive case
        template <typename FirstDim, typename... RestDims>
        struct calculate_storage_size<FirstDim, RestDims...> {
            // Get size and stride for this dimension
            static constexpr int size = FirstDim::size;
            static constexpr int stride = FirstDim::stride;

            // Calculate max offset for this dimension plus rest of dims
            static constexpr int value =
                (size - 1) * stride + calculate_storage_size<RestDims...>::value;
        };

        // ========================================================================
        // Finest fold computation for Cursor coordinates
        // ========================================================================

        // Compare two DimInfos with same base dim, keep the one with smaller stride (finest)
        template <typename InfoA, typename InfoB> struct finer_dim_info {
            static constexpr int stride_a = InfoA::dim_type::stride;
            static constexpr int stride_b = InfoB::dim_type::stride;
            using type = conditional_t<(stride_a <= stride_b), InfoA, InfoB>;
        };

        // Find finest info from a tuple of DimInfos (smallest stride)
        template <typename InfosTuple> struct find_finest_info;

        template <typename FirstInfo> struct find_finest_info<tuple<FirstInfo>> {
            using type = FirstInfo;
        };

        template <typename FirstInfo, typename SecondInfo, typename... RestInfos>
        struct find_finest_info<tuple<FirstInfo, SecondInfo, RestInfos...>> {
            using finer = typename finer_dim_info<FirstInfo, SecondInfo>::type;
            using type = typename find_finest_info<tuple<finer, RestInfos...>>::type;
        };

        // Build tuple of finest DimInfos, one per unique base dimension
        template <typename BaseDimsTuple, typename AllInfosTuple> struct build_finest_infos;

        template <typename AllInfosTuple> struct build_finest_infos<tuple<>, AllInfosTuple> {
            using type = tuple<>;
        };

        template <typename FirstBase, typename... RestBases, typename AllInfosTuple>
        struct build_finest_infos<tuple<FirstBase, RestBases...>, AllInfosTuple> {
            // Filter to get only infos with this base dim
            using matching_infos = keep_dim_infos_by_base_t<AllInfosTuple, tuple<FirstBase>>;
            // Find finest among matching
            using finest = typename find_finest_info<matching_infos>::type;
            // Recurse for rest
            using rest = typename build_finest_infos<tuple<RestBases...>, AllInfosTuple>::type;
            using type = tuple_prepend_t<finest, rest>;
        };

        // Main entry point: get finest DimInfo per base dimension
        template <typename InfosTuple> struct finest_dim_infos {
            using unique_bases = typename dim_infos_unique_bases<InfosTuple>::type;
            using type = typename build_finest_infos<unique_bases, InfosTuple>::type;
        };

        template <typename InfosTuple>
        using finest_dim_infos_t = typename finest_dim_infos<InfosTuple>::type;

        // Extract dim_types from a tuple of DimInfos to create Coordinates type
        template <typename InfosTuple> struct coordinates_from_infos;

        template <typename... Infos> struct coordinates_from_infos<tuple<Infos...>> {
            using type = Coordinates<typename Infos::dim_type...>;
        };

        template <typename InfosTuple>
        using coordinates_from_infos_t = typename coordinates_from_infos<InfosTuple>::type;

        // Cursor's coordinate type: Coordinates with finest fold of each base dimension
        template <typename... DimInfos> struct cursor_coordinates_type {
            using finest_infos = finest_dim_infos_t<tuple<DimInfos...>>;
            using type = coordinates_from_infos_t<finest_infos>;
        };

        template <typename... DimInfos>
        using cursor_coordinates_t = typename cursor_coordinates_type<DimInfos...>::type;

        // ========================================================================
        // Offset computation from Coordinates
        // ========================================================================

        // Compute offset contribution from a single coordinate dimension to a single DimInfo
        template <typename CoordDim, typename Info> struct compute_dim_offset {
            using TargetDimType = typename Info::dim_type;
            using CoordBaseDim = typename CoordDim::dim_type;
            using TargetBaseDim = typename TargetDimType::dim_type;

            static constexpr bool same_base = is_same<CoordBaseDim, TargetBaseDim>::value;
            static constexpr int target_stride = TargetDimType::stride;

            // Compute the offset contribution from coordinate to this dimension
            DEVICE static constexpr int compute(CoordDim coord) {
                if constexpr (!same_base) {
                    return 0;
                } else {
                    // Fold coordinate to target stride and convert to memory offset
                    auto target_folded = coord.template fold<target_stride>();
                    return Info::dim_to_offset(target_folded);
                }
            }
        };

        // Sum offset contributions from one coordinate dimension to all DimInfos
        template <typename CoordDim, typename InfosTuple> struct sum_coord_to_infos;

        template <typename CoordDim> struct sum_coord_to_infos<CoordDim, tuple<>> {
            DEVICE static constexpr int compute(CoordDim) {
                return 0;
            }
        };

        template <typename CoordDim, typename FirstInfo, typename... RestInfos>
        struct sum_coord_to_infos<CoordDim, tuple<FirstInfo, RestInfos...>> {
            DEVICE static constexpr int compute(CoordDim coord) {
                return compute_dim_offset<CoordDim, FirstInfo>::compute(coord) +
                       sum_coord_to_infos<CoordDim, tuple<RestInfos...>>::compute(coord);
            }
        };

        // Compute total offset from Coordinates to all DimInfos
        template <typename CoordsTuple, typename InfosTuple> struct compute_total_offset_impl;

        template <typename InfosTuple> struct compute_total_offset_impl<tuple<>, InfosTuple> {
            template <typename Coords> DEVICE static constexpr int compute(const Coords&) {
                return 0;
            }
        };

        template <typename FirstCoord, typename... RestCoords, typename InfosTuple>
        struct compute_total_offset_impl<tuple<FirstCoord, RestCoords...>, InfosTuple> {
            template <typename Coords> DEVICE static constexpr int compute(const Coords& coords) {
                int this_offset = sum_coord_to_infos<FirstCoord, InfosTuple>::compute(
                    coords.template get<FirstCoord>());
                int rest_offset =
                    compute_total_offset_impl<tuple<RestCoords...>, InfosTuple>::compute(coords);
                return this_offset + rest_offset;
            }
        };

        template <typename... CoordDims, typename... DimInfos>
        DEVICE constexpr int compute_offset_from_coords(const Coordinates<CoordDims...>& coords,
                                                        const tuple<DimInfos...>& /* infos_tag */) {
            return compute_total_offset_impl<tuple<CoordDims...>, tuple<DimInfos...>>::compute(
                coords);
        }

        // ========================================================================
        // Helper to update a single coordinate value in Coordinates
        // Must be defined before add_subscript_dim_to_cursor_coords
        // ========================================================================

        // Helper to select value for coordinate update
        template <typename TargetDim, typename CurrentDim, typename... AllDims>
        DEVICE constexpr auto select_coord_value(const Coordinates<AllDims...>& coords,
                                                 TargetDim new_value) {
            if constexpr (is_same<CurrentDim, TargetDim>::value) {
                return new_value;
            } else {
                return coords.template get<CurrentDim>();
            }
        }

        // Helper to update a single coordinate value in Coordinates
        template <typename TargetDim, typename... Dims>
        DEVICE constexpr auto update_coord_value(const Coordinates<Dims...>& coords,
                                                 TargetDim new_value) {
            return Coordinates<Dims...>(select_coord_value<TargetDim, Dims>(coords, new_value)...);
        }

        // ========================================================================
        // Add a subscript coordinate to cursor coordinates
        // For each dimension in the subscript, if its base matches a cursor coord,
        // fold it to the cursor's dimension type and add.
        // ========================================================================

        // Process one subscript dimension: find matching cursor dim and add
        template <typename SubDim, typename... CursorDims>
        DEVICE constexpr auto
        add_subscript_dim_to_cursor_coords(SubDim sub_dim,
                                           const Coordinates<CursorDims...>& cursor_coords) {
            using sub_base = typename SubDim::dim_type;

            if constexpr (tuple_contains_base_dim<sub_base, tuple<CursorDims...>>::value) {
                // Find which cursor dimension matches this base
                using matching_cursor_dim =
                    typename tuple_find_by_base_dim<sub_base, tuple<CursorDims...>>::type;
                // Fold subscript dim to cursor dim's type and add
                constexpr int matching_dim_stride = matching_cursor_dim::stride;
                auto folded = sub_dim.template fold<matching_dim_stride>();
                auto current = cursor_coords.template get<matching_cursor_dim>();
                // Create new coordinates with the updated value
                return update_coord_value<matching_cursor_dim>(cursor_coords, current + folded);
            } else {
                // No matching dimension in cursor, ignore this subscript dimension
                return cursor_coords;
            }
        }

        // Apply all subscript dimensions to cursor coordinates
        template <typename CursorCoords, typename SubTuple> struct apply_subscript_impl;

        // Base case: no more subscript dimensions to process
        template <typename CursorCoords> struct apply_subscript_impl<CursorCoords, tuple<>> {
            template <typename SubCoords>
            DEVICE static constexpr CursorCoords apply(const CursorCoords& cursor_coords,
                                                       const SubCoords&) {
                return cursor_coords;
            }
        };

        // Recursive case: process first subscript dimension, then recurse
        template <typename CursorCoords, typename FirstSub, typename... RestSubs>
        struct apply_subscript_impl<CursorCoords, tuple<FirstSub, RestSubs...>> {
            template <typename SubCoords>
            DEVICE static constexpr CursorCoords apply(const CursorCoords& cursor_coords,
                                                       const SubCoords& sub_coords) {
                auto updated = add_subscript_dim_to_cursor_coords(
                    sub_coords.template get<FirstSub>(), cursor_coords);
                // Continue with remaining subscript dimensions
                return apply_subscript_impl<CursorCoords, tuple<RestSubs...>>::apply(updated,
                                                                                     sub_coords);
            }
        };

        template <typename... CursorDims, typename... SubDims>
        DEVICE constexpr Coordinates<CursorDims...>
        apply_subscript_to_cursor(const Coordinates<CursorDims...>& cursor_coords,
                                  const Coordinates<SubDims...>& sub_coords) {
            return apply_subscript_impl<Coordinates<CursorDims...>, tuple<SubDims...>>::apply(
                cursor_coords, sub_coords);
        }

        // ========================================================================
        // Iterative derived-dim logic for Cursor::operator[]
        // ========================================================================

        // Forward declaration for mutual recursion
        template <typename DerivedDimsTuple, typename Subscript, typename CursorType>
        struct apply_all_matching_derived_dims;

        template <typename DerivedDimsTuple, typename Subscript, typename CursorType>
        struct apply_all_matching_derived_dims_impl {
            static constexpr CursorType apply(const CursorType& cursor,
                                              const Subscript& subscript) {
                // No more derived dims to apply - apply subscript to cursor coordinates
                // Use cursor's base pointer (not get() which includes offset) since we're
                // storing new coordinates that will be used to compute offset in get()
                return CursorType(cursor.base_ptr(),
                                  apply_subscript_to_cursor(cursor.coordinates(), subscript));
            }
        };

        template <typename DerivedDimsTuple, typename Subscript, typename CursorType,
                  typename FirstMatch>
        struct apply_all_matching_derived_dims_impl2 {
            // Get the input dims of this derived dimension
            using input_dims = typename FirstMatch::input_dims;
            using output_dims = typename FirstMatch::output_dims;

            static constexpr CursorType apply(const CursorType& cursor,
                                              const Subscript& subscript) {
                // Project subscript onto derived input dims and compute derived output coords
                auto derived_output_coords = apply_derived_dim<FirstMatch>(subscript);

                // Check if any output dim type exactly matches a subscript dim type.
                // Overlapping BASE dims with different strides are OK (e.g., I_TILE and I_FINE
                // both contribute to base I but at different strides - the tensor offset
                // computation handles this correctly by folding each to the target stride).
                // Only exact type duplicates would be a problem.
                using output_coords_type = decltype(derived_output_coords);
                constexpr bool has_exact_duplicate =
                    tuples_share_type_v<typename Subscript::dims_tuple,
                                        typename output_coords_type::dims_tuple>;
                static_assert(!has_exact_duplicate,
                              "Derived dimension output has exact same type as subscript dim");

                // Concatenate coordinates - safe because no exact type duplicates.
                // Different strides for same base are handled correctly by offset computation.
                auto t2 = concat_coordinates(subscript, derived_output_coords);
                // Remove this derived dim from the tuple
                using removed = tuple<FirstMatch>;
                using next_derived = tuple_filter_out_t<DerivedDimsTuple, removed>;
                // Recurse with remaining derived dims
                return apply_all_matching_derived_dims<next_derived, decltype(t2),
                                                       CursorType>::apply(cursor, t2);
            }
        };

        template <typename DerivedDimsTuple, typename Subscript, typename CursorType>
        struct apply_all_matching_derived_dims {
            using first_match =
                find_matching_derived_dim_t<DerivedDimsTuple, typename Subscript::dims_tuple>;
            static constexpr bool has_match = !is_void_v<first_match>;

            static constexpr CursorType apply(const CursorType& cursor,
                                              const Subscript& subscript) {
                if constexpr (has_match) {
                    return apply_all_matching_derived_dims_impl2<
                        DerivedDimsTuple, Subscript, CursorType, first_match>::apply(cursor,
                                                                                     subscript);
                } else {
                    return apply_all_matching_derived_dims_impl<DerivedDimsTuple, Subscript,
                                                                CursorType>::apply(cursor,
                                                                                   subscript);
                }
            }
        };

    } // namespace detail

    /// @brief A multi-dimensional pointer that traverses tensor dimensions.
    ///
    /// Cursor stores logical coordinates and defers folding to physical offset until
    /// pointer access. This enables correct cross-fold carry behavior when accumulating
    /// subscripts or stepping through dimensions.
    ///
    /// Use rebase() to convert to a BaseCursor when you need direct pointer arithmetic
    /// for performance-critical inner loops. Note that rebase() "commits" the current
    /// logical coordinates to a physical pointer, and subsequent BaseCursor::step()
    /// calls will not handle cross-fold carry.
    ///
    /// @tparam DataType the data type of the tensor
    /// @tparam Params template parameters (DimInfo types are extracted automatically)
    template <typename DataType, typename... Params> class Cursor : public Data<DataType> {
        // Extract DimInfo types from Params
        template <typename InfosTuple> struct unpack_dim_infos;

        template <typename... Infos> struct unpack_dim_infos<detail::tuple<Infos...>> {
            using base_cursor = BaseCursor<DataType, Infos...>;
            using coords = detail::cursor_coordinates_t<Infos...>;
        };

    public:
        using dim_infos = detail::filter_dim_infos_t<Params...>;
        using derived_dims = detail::filter_derived_dims_t<Params...>;

    private:
        using unpacked = unpack_dim_infos<dim_infos>;

    public:
        using Base = Data<DataType>;
        using data_type = DataType;
        using base_cursor_type = typename unpacked::base_cursor;
        using coordinates_type = typename unpacked::coords;

        DEVICE constexpr Cursor(DataType* data = nullptr) : Base(data), _coords() {}

        DEVICE constexpr Cursor(DataType* data, const coordinates_type& coords)
            : Base(data),
              _coords(coords) {}

        // Constructor from integer offset for backward compatibility
        DEVICE constexpr Cursor(DataType* data, int offset) : Base(data), _coords() {
            if (offset != 0) { Base::reset(Base::get() + offset); }
        }

        DEVICE constexpr data_type* get() const {
            return Base::get() + compute_offset();
        }

        /// @brief Get the base pointer without applying coordinate offset.
        /// Used internally when constructing a new cursor with updated coordinates.
        DEVICE constexpr data_type* base_ptr() const {
            return Base::get();
        }

        DEVICE constexpr base_cursor_type rebase() const {
            return base_cursor_type(get());
        }

        DEVICE constexpr const coordinates_type& coordinates() const {
            return _coords;
        }

        template <typename DimType>
        static constexpr bool has_dimension_v =
            dim_traits::tuple_has_dimension<DimType, dim_infos>::value;

    private:
        DEVICE constexpr int compute_offset() const {
            return detail::compute_offset_from_coords(_coords, dim_infos{});
        }

        template <typename SourceDim>
        using matching_dim_infos_t =
            detail::keep_dim_infos_by_base_t<dim_infos,
                                             detail::tuple<typename SourceDim::dim_type>>;

        template <typename DimsTuple> struct any_coordinate_matches_impl;

        template <typename... CoordDims>
        struct any_coordinate_matches_impl<detail::tuple<CoordDims...>> {
            static constexpr bool value =
                (... || (detail::tuple_size<matching_dim_infos_t<CoordDims>>::value > 0));
        };

        template <typename DimsTuple>
        static constexpr bool any_coordinate_matches_v =
            any_coordinate_matches_impl<DimsTuple>::value;

    public:
        /// @brief Subscript operator for dimension-like types, Coordinates, or types with
        /// coordinates().
        template <typename T> DEVICE constexpr Cursor operator[](T t) const {
            if constexpr (detail::is_coordinates_v<T>) {
                using derived_tuple = derived_dims;
                using coords_tuple = typename T::dims_tuple;
                // Fast path: no derived dims, just apply subscript directly
                if constexpr (detail::tuple_size<derived_tuple>::value == 0) {
                    auto new_coords = detail::apply_subscript_to_cursor(_coords, t);
                    return Cursor(Base::get(), new_coords);
                } else {
                    // Iteratively apply all matching derived dims
                    return detail::apply_all_matching_derived_dims<derived_tuple, T, Cursor>::apply(
                        *this, t);
                }
            } else if constexpr (detail::has_coordinates_v<T>) {
                return (*this)[t.coordinates()];
            } else if constexpr (detail::is_dim_like_v<T>) {
                using matching = matching_dim_infos_t<T>;
                static_assert(detail::tuple_size<matching>::value > 0,
                              "Subscript dimension does not match any tensor dimension");
                auto new_coords = detail::add_subscript_dim_to_cursor_coords(t, _coords);
                return Cursor(Base::get(), new_coords);
            } else {
                static_assert(
                    detail::is_dim_like_v<T> || detail::is_coordinates_v<T> ||
                        detail::has_coordinates_v<T>,
                    "Subscript type must be Dim-like, Coordinates, or have coordinates()");
                return *this;
            }
        }

        template <typename DimType> DEVICE Cursor& step(DimType d = 1) {
            _coords = detail::add_subscript_dim_to_cursor_coords(d, _coords);
            return *this;
        }

        /// @brief Get size for a specific dimension as it exists in the cursor.
        /// Fails to compile if the exact dimension does not exist in the cursor.
        template <typename DimType> DEVICE static constexpr DimType size() {
            using info = typename dim_traits::tuple_find_dim_info<DimType, dim_infos>::info;
            return DimType(info::size);
        }

        /// @brief Get the total extent in the requested dimension's base type.
        /// Returns the size of the coarsest fold, converted to the requested dimension type.
        template <typename DimType> DEVICE static constexpr DimType extent() {
            using RequestedBaseDim = typename DimType::dim_type;
            using matching_infos =
                detail::keep_dim_infos_by_base_t<dim_infos, detail::tuple<RequestedBaseDim>>;

            static_assert(detail::tuple_size<matching_infos>::value > 0,
                          "Requested dimension type does not match any cursor dimension");

            // Find the coarsest info for this base dimension
            using coarsest_info =
                typename detail::find_coarsest_info_for_base<RequestedBaseDim,
                                                             matching_infos>::type;
            using coarsest_dim_type = typename coarsest_info::dim_type;

            // Get size of coarsest dimension and convert to requested type
            return DimType(size<coarsest_dim_type>());
        }

        /// @brief Get Coordinates that include the extent for each base dimension.
        DEVICE static constexpr auto extents() {
            using coarsest_infos = detail::coarsest_dim_infos_t<dim_infos>;
            return make_sizes_from_infos(coarsest_infos{});
        }

        /// @brief Check if all coordinates are within bounds.
        /// @return true if each coordinate is less than the corresponding extent.
        DEVICE constexpr bool inbounds() const {
            return _coords < extents();
        }

        DEVICE constexpr data_type& operator*() const {
            return *this->get();
        }

        DEVICE constexpr data_type* operator->() const {
            return this->get();
        }

    private:
        template <typename... Infos>
        DEVICE static constexpr auto make_sizes_from_infos(detail::tuple<Infos...>) {
            return Coordinates<typename Infos::dim_type...>(
                typename Infos::dim_type(Infos::size)...);
        }

        coordinates_type _coords;
    };

    /// @brief A cursor with direct pointer arithmetic (no coordinate tracking).
    ///
    /// BaseCursor is used for performance-critical inner loops where you want direct
    /// pointer manipulation without the overhead of coordinate tracking. It is typically
    /// created via Cursor::rebase().
    ///
    /// Note: BaseCursor::step() does direct pointer arithmetic using memory strides.
    /// It does NOT handle cross-fold carry behavior. If you need cross-fold carry,
    /// use Cursor instead.
    ///
    /// @tparam DataType the data type of the tensor
    /// @tparam Params template parameters (DimInfo types are extracted automatically)
    template <typename DataType, typename... Params> class BaseCursor : public Data<DataType> {
    public:
        using dim_infos = detail::filter_dim_infos_t<Params...>;
        using derived_dims = detail::filter_derived_dims_t<Params...>;
        using Base = Data<DataType>;
        using data_type = DataType;
        using Base::Base;
        using Base::get;
        using cursor_type = Cursor<DataType, Params...>;
        using base_cursor_type = BaseCursor<DataType, Params...>;

        template <typename DimType>
        static constexpr bool has_dimension_v =
            dim_traits::tuple_has_dimension<DimType, dim_infos>::value;

        /// @brief Subscript operator that returns a new Cursor at the specified dimension index.
        /// @tparam DimType the dimension to apply the subscript index to.
        /// @param d the subscript index.
        /// @return a new Cursor that points to the element at the specified dimension index.
        template <typename T> DEVICE constexpr cursor_type operator[](T t) const {
            return cursor_type(Base::get())[t];
        }

        /// @brief Increment the cursor in a specific dimension type.
        template <typename DimType> DEVICE BaseCursor& step(DimType d = 1) {
            constexpr int stride =
                dim_traits::tuple_find_dim_info<DimType, dim_infos>::info::stride;
            Base::reset(Base::get() + d.get() * stride);
            return *this;
        }
    };

    /// @brief Tensor class.
    /// @tparam DataType the data type of the tensor
    /// @tparam Params template parameters (DimInfo types are extracted automatically)
    template <typename DataType, typename... Params> class Tensor : public Data<DataType> {
        // Helper to unpack dim_infos tuple for template instantiation
        template <typename InfosTuple> struct unpack_for_compound_index;

        template <typename... Infos> struct unpack_for_compound_index<detail::tuple<Infos...>> {
            using type = CompoundIndex<Infos...>;
            static constexpr int total = detail::product_sizes<Infos...>::value;
            static constexpr int storage = detail::calculate_storage_size<Infos...>::value;
        };

    public:
        using dim_infos = detail::filter_dim_infos_t<Params...>;
        using derived_dims = detail::filter_derived_dims_t<Params...>;

    private:
        using unpacked = unpack_for_compound_index<dim_infos>;

    public:
        using data_type = DataType;
        using Base = Data<data_type>;
        using Base::Base;
        using Base::get;

        using cursor_type = Cursor<DataType, Params...>;
        using base_cursor_type = BaseCursor<DataType, Params...>;

        // CompoundIndex type that uses tensor's size and strides.
        using index_type = typename unpacked::type;

        // Total number of elements (product of all dimension sizes)
        static constexpr int total_size = unpacked::total;

        // Helper variable template for cleaner usage
        template <typename DimType>
        static constexpr bool has_dimension_v =
            dim_traits::tuple_has_dimension<DimType, dim_infos>::value;

        // Allocate a tensor on the stack.
        DEVICE static Tensor allocate(StackAllocator& allocator) {
            return Tensor(allocator.allocate<data_type>(storage_size()));
        }

        // Deallocate a tensor from the stack.
        DEVICE void deallocate(StackAllocator& allocator) {
            allocator.deallocate<data_type>(get(), storage_size());
        }

        // For compatibility with existing code
        DEVICE static constexpr int size() {
            return total_size;
        }

        // Calculate actual storage size (accounting for strides)
        DEVICE static constexpr int storage_size() {
            return unpacked::storage;
        }

        // Return actual bytes needed, accounting for strides
        DEVICE static constexpr int num_bytes() {
            return storage_size() * sizeof(data_type);
        }

        // Get size for a specific dimension as it exists in the tensor.
        // Fails to compile if the exact dimension does not exist in the tensor.
        template <typename DimType> DEVICE static constexpr DimType size() {
            using info = typename dim_traits::tuple_find_dim_info<DimType, dim_infos>::info;
            return DimType(info::size);
        }

        // Get the total extent of the tensor in the requested dimension's base type.
        // Returns the size of the coarsest fold, converted to the requested dimension type.
        template <typename DimType> DEVICE static constexpr DimType extent() {
            using RequestedBaseDim = typename DimType::dim_type;
            using matching_infos =
                detail::keep_dim_infos_by_base_t<dim_infos, detail::tuple<RequestedBaseDim>>;

            static_assert(detail::tuple_size<matching_infos>::value > 0,
                          "Requested dimension type does not match any tensor dimension");

            // Find the coarsest info for this base dimension
            using coarsest_info =
                typename detail::find_coarsest_info_for_base<RequestedBaseDim,
                                                             matching_infos>::type;
            using coarsest_dim_type = typename coarsest_info::dim_type;

            // Get size of coarsest dimension and convert to requested type
            return DimType(size<coarsest_dim_type>());
        }

        // Get Coordinates that include the extent for each base dimension.
        DEVICE static constexpr auto extents() {
            using coarsest_infos = detail::coarsest_dim_infos_t<dim_infos>;
            return make_sizes_from_infos(coarsest_infos{});
        }

        /// @brief Subscript operator with any dimension type or Coordinates.
        template <typename T> DEVICE constexpr cursor_type operator[](T t) const {
            return cursor_type(get())[t];
        }

        /// @brief Get a cursor at a specific offset.
        DEVICE constexpr cursor_type offset(int offset) const {
            return cursor_type(get(), offset);
        }

        /// @brief Slice method to create a view with a different offset and size in one dimension.
        template <int SliceSize, typename SliceDimType>
        DEVICE constexpr auto slice(SliceDimType slice_start) {
            using updated_infos =
                detail::tuple_update_dim_info_t<SliceDimType, SliceSize, dim_infos>;
            using tensor_type =
                typename detail::tensor_type_from_dim_info_tuple<DataType,
                                                                 updated_infos>::tensor_type;
            return tensor_type((*this)[slice_start].get());
        }

        /// @brief Load data from a source cursor that points to a shared memory buffer.
        template <typename SrcCursorType> DEVICE void load(SrcCursorType src) {
            load_dispatch<SrcCursorType, dim_infos>(*this, src);
        }

        /// @brief Apply a custom function to each element of the tensor
        template <typename F> DEVICE void apply(F func) {
            apply_dispatch<F, dim_infos>(*this, func);
        }

        /// @brief Fill the tensor with zeros.
        DEVICE void zero() {
            auto zero_func = [](auto obj) { obj->zero(); };
            apply(zero_func);
        }

        /// @brief Fill the tensor with a specified value.
        template <typename Vector> DEVICE void fill(Vector value) {
            auto fill_func = [value](auto obj) { obj->fill(value); };
            apply(fill_func);
        }

        template <typename Vector> DEVICE void add(Vector value) {
            auto add_func = [value](auto obj) { obj->add(value); };
            apply(add_func);
        }

        /// @brief Get the base pointer without applying coordinate offset.
        /// Used internally when constructing a new cursor with updated coordinates.
        DEVICE constexpr data_type* base_ptr() const {
            return Base::get();
        }

    private:
        template <typename... Infos>
        DEVICE static constexpr auto make_sizes_from_infos(detail::tuple<Infos...>) {
            return Coordinates<typename Infos::dim_type...>(
                typename Infos::dim_type(Infos::size)...);
        }

        /// @brief Base case for loading data from a source cursor.
        template <typename DstCursorType, typename SrcCursorType>
        DEVICE static void load_impl(DstCursorType dst, SrcCursorType src) {
            dst->load(src.get());
        }

        /// @brief Recursive case for loading data from a source cursor.
        template <typename DstCursorType, typename SrcCursorType, typename FirstDimInfo,
                  typename... RestDimInfos>
        DEVICE static void load_impl(DstCursorType dst, SrcCursorType src) {
            using FirstDimType = typename FirstDimInfo::dim_type;
            auto size = FirstDimType(FirstDimInfo::size);
            for (auto i : range(size)) {
                load_impl<decltype(dst[i]), decltype(src[i]), RestDimInfos...>(dst[i], src[i]);
            }
        }

        /// @brief Dispatcher that unpacks dim_infos tuple for load_impl.
        template <typename SrcCursorType, typename InfosTuple> struct load_dispatch_impl;

        template <typename SrcCursorType, typename... Infos>
        struct load_dispatch_impl<SrcCursorType, detail::tuple<Infos...>> {
            DEVICE static void dispatch(Tensor& tensor, SrcCursorType src) {
                load_impl<decltype(tensor), SrcCursorType, Infos...>(tensor, src);
            }
        };

        template <typename SrcCursorType, typename InfosTuple>
        DEVICE void load_dispatch(Tensor& tensor, SrcCursorType src) {
            load_dispatch_impl<SrcCursorType, InfosTuple>::dispatch(tensor, src);
        }

        /// @brief Base case for applying a function to tensor elements.
        template <typename F, typename CursorType>
        DEVICE static void apply_impl(CursorType obj, F func) {
            func(obj);
        }

        /// @brief Recursive case for applying a function to tensor elements.
        template <typename F, typename CursorType, typename FirstDimInfo, typename... RestDimInfos>
        DEVICE static void apply_impl(CursorType obj, F func) {
            using FirstDimType = typename FirstDimInfo::dim_type;
            auto size = FirstDimType(FirstDimInfo::size);
            for (auto i : range(size)) {
                apply_impl<F, decltype(obj[i]), RestDimInfos...>(obj[i], func);
            }
        }

        /// @brief Dispatcher that unpacks dim_infos tuple for apply_impl.
        template <typename F, typename InfosTuple> struct apply_dispatch_impl;

        template <typename F, typename... Infos>
        struct apply_dispatch_impl<F, detail::tuple<Infos...>> {
            DEVICE static void dispatch(Tensor& tensor, F func) {
                apply_impl<F, decltype(tensor), Infos...>(tensor, func);
            }
        };

        template <typename F, typename InfosTuple>
        DEVICE void apply_dispatch(Tensor& tensor, F func) {
            apply_dispatch_impl<F, InfosTuple>::dispatch(tensor, func);
        }
    };
}

#endif
