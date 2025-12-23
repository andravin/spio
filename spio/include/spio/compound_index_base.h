#ifndef SPIO_COMPOUND_INDEX_BASE_H_
#define SPIO_COMPOUND_INDEX_BASE_H_

#include "spio/macros.h"
#include "spio/dim.h"
#include "spio/coordinates.h"

namespace spio {

    template <typename OffsetType = OFFSET> class CompoundIndexBase {
    public:
        explicit DEVICE constexpr CompoundIndexBase(int offset = 0) : _offset(offset) {}

        /// @brief Construct from any dimension type compatible with OffsetType's base dimension.
        template <typename DimLike,
                  detail::enable_if_t<detail::is_dim_like_v<DimLike> &&
                                          detail::dims_compatible_v<DimLike, OffsetType>,
                                      int> = 0>
        DEVICE constexpr CompoundIndexBase(DimLike dim)
            : _offset(dim.unfold().get() / OffsetType::stride) {}

        /// @brief Construct from Coordinates by summing matching dimensions and folding to
        /// OffsetType
        template <typename... CoordDims>
        DEVICE constexpr CompoundIndexBase(const Coordinates<CoordDims...>& coords)
            : _offset(offset_from_coords(coords)) {}

        /// @brief Construct from any type with a coordinates() method
        template <
            typename T,
            detail::enable_if_t<detail::has_coordinates_v<T> && !detail::is_dim_like_v<T>, int> = 0>
        DEVICE constexpr CompoundIndexBase(const T& t) : CompoundIndexBase(t.coordinates()) {}

        DEVICE constexpr OffsetType offset() const {
            return _offset;
        }

    private:
        /// @brief Sum all coordinate dimensions matching OffsetType's base and fold to OffsetType
        template <typename... CoordDims>
        DEVICE static constexpr OffsetType
        offset_from_coords(const Coordinates<CoordDims...>& coords) {
            using offset_base = typename OffsetType::dim_type;
            using matching = detail::tuple_keep_base_dims_t<detail::tuple<CoordDims...>,
                                                            detail::tuple<offset_base>>;

            static_assert(detail::tuple_size<matching>::value > 0,
                          "Coordinates must contain at least one dimension matching OffsetType's "
                          "base dimension");

            // Sum all matching dims - addition handles stride normalization
            auto summed = sum_matching(coords, matching{});
            // Convert to base value and fold to OffsetType's stride
            constexpr int offset_stride = OffsetType::stride;
            auto offset_value = summed.template fold<offset_stride>();
            return offset_value.template cast<OffsetType>();
        }

        template <typename... CoordDims, typename... MatchingDims>
        DEVICE static constexpr auto sum_matching(const Coordinates<CoordDims...>& coords,
                                                  detail::tuple<MatchingDims...>) {
            return (coords.template get<MatchingDims>() + ...);
        }

        const OffsetType _offset;
    };
}

#endif
