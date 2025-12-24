#ifndef SPIO_CHECKERBOARD_IDX_H
#define SPIO_CHECKERBOARD_IDX_H

#include "spio/macros.h"
#include "spio/dim.h"
#include "spio/dim_info.h"
#include "spio/compound_index_base.h"
#include "spio/coordinates.h"

namespace spio {

    /// @brief An index class for checkerboard memory layout.
    ///
    /// Arrange an N x 2 matrix into a checkerboard grid with the given number
    /// of banks (i.e., grid columns).
    ///
    /// CompoundIndex (x, color) is the x and color index of the element in the grid.
    ///
    /// Notice that cells with the same color are never adjacent to each other. This structure
    /// is useful for storing a matrix that can be accessed both row-wise (i.e., color-wise) and
    /// column-wise (i.e., x-wise) without bank conflicts.
    ///
    /// Example with ranks = 8:
    ///
    /// This checkerboard is appropriate for loading a 16x16 matrix using 8-element [half2]
    /// vectors for the minor dimension, so that is is logically a 16 x 2 matrix. There
    /// are logically 32 elements of size 16-bytes each, which can be loaded from
    /// global memory by a warp with a single 128-bit load instruction.
    ///
    /// If the the 16 x 2 [8 half2] = 512 bytes submatrix is contiguous in global memory, then
    /// the 128-bit load will produce a single wavefront in the memory pipeline.
    ///
    /// Shared memory has 32 banks of 4 bytes each. When using 16-byte elements,
    /// shared memory has effectively 8 banks which corresponds to the 8
    /// ranks of the checkerboard layout.
    ///
    /// Loads from shared memory of elements (x, 0) or (x, 1) for 8 contiguous
    /// values of x will have zero bank conflicts.
    ///
    ///                          Rank
    ///              0        2        4        6
    ///         +--------+--------+--------+--------+
    ///       0 | (0, 0) | (0, 1) | (1, 0) | (1, 1) |
    ///         +--------+--------+--------+--------+
    ///       8 | (2, 1) | (2, 0) | (3, 1) | (3, 0) |
    ///         + --------+--------+--------+--------
    ///      16 | (4, 0) | (4, 1) | (5, 0) | (5, 1) |
    ///         +--------+--------+--------+--------+
    ///      24 | (6, 1) | (6, 0) | (7, 1) | (7, 0) |
    ///         +--------+--------+--------+--------+
    ///
    /// @tparam ranks the number of column-ranks in the checkerboard.
    /// @tparam PairDim the dimension type for the pair index
    /// @tparam ColorDim the dimension type for the color index
    template <int ranks, typename PairDim, typename ColorDim, typename OffsetDim, int Size = 32>
    class CheckerboardIndex : public CompoundIndexBase<OffsetDim> {
    public:
        using CompoundIndexBase<OffsetDim>::offset;

        //
        // Derived dimension interface.
        //
        static constexpr int size = Size;
        static constexpr int num_colors = 2;
        static constexpr int num_pairs = Size / num_colors;

        using input_dims =
            detail::tuple<DimSize<PairDim, num_pairs>, DimSize<ColorDim, num_colors>>;
        using output_dims = detail::tuple<DimSize<OffsetDim, size>>;

        using input_coordinates = Coordinates<PairDim, ColorDim>;

        DEVICE static constexpr auto compute_coordinates(const input_coordinates& coords) {
            return CheckerboardIndex(coords.template get<PairDim>(),
                                     coords.template get<ColorDim>())
                .coordinates();
        }

        //
        // Constructors
        //

        // Explicit int constructor
        explicit DEVICE constexpr CheckerboardIndex(int offset = 0)
            : CompoundIndexBase<OffsetDim>(offset) {}

        /// @brief Construct from a compatible dimension type
        template <typename DimLike,
                  detail::enable_if_t<detail::is_dim_like_v<DimLike> &&
                                          detail::dims_compatible_v<DimLike, OffsetDim>,
                                      int> = 0>
        DEVICE constexpr CheckerboardIndex(DimLike dim) : CompoundIndexBase<OffsetDim>(dim) {}

        /// @brief Construct index from pair and color dimensions
        DEVICE inline constexpr CheckerboardIndex(PairDim pair_dim, ColorDim color_dim)
            : CompoundIndexBase<OffsetDim>(compute_offset(pair_dim.get(), color_dim.get())) {}

        /// @brief Construct from any index type that has get<PairDim>() and get<ColorDim>() methods
        /// This handles fragment load indices that don't have coordinates() but do have typed
        /// getters. Excluded: dim-like types (handled above) and types compatible with OffsetDim.
        template <typename IndexType,
                  detail::enable_if_t<!detail::is_dim_like_v<IndexType> &&
                                          !detail::dims_compatible_v<IndexType, OffsetDim>,
                                      int> = 0>
        DEVICE inline constexpr CheckerboardIndex(IndexType idx)
            : CompoundIndexBase<OffsetDim>(compute_offset(idx.template get<PairDim>().get(),
                                                          idx.template get<ColorDim>().get())) {}

        /// @brief Get dimension value by type
        /// @tparam Dim The dimension type to retrieve
        /// @return The dimension value with the proper type
        template <typename Dim> DEVICE constexpr auto get() const {
            if constexpr (detail::is_same<Dim, PairDim>::value) {
                return PairDim(offset().get() >> 1);
            } else if constexpr (detail::is_same<Dim, ColorDim>::value) {
                const int row =
                    static_cast<unsigned>(offset().get()) / static_cast<unsigned>(ranks);
                return ColorDim((offset().get() & 1) ^ (row & 1));
            } else if constexpr (detail::is_same<Dim, OffsetDim>::value) {
                return offset();
            } else {
                static_assert(detail::is_same<Dim, PairDim>::value ||
                                  detail::is_same<Dim, ColorDim>::value ||
                                  detail::is_same<Dim, OffsetDim>::value,
                              "Invalid dimension type for CheckerboardIndex");
                return Dim(0);
            }
        }

        // Convenience methods for backward compatibility
        DEVICE inline constexpr int pair() const {
            return get<PairDim>().get();
        }

        DEVICE inline constexpr int color() const {
            return get<ColorDim>().get();
        }

        /// @brief Calculate offset from pair and color indices
        /// @param pair the index of a pair of grid elements.
        /// @param color the black (0) or white (1) grid element in the pair.
        /// @return Offset into the checkeboard grid.
        DEVICE inline static constexpr OffsetDim compute_offset(int pair, int color) {
            int row = static_cast<unsigned>(pair) / (static_cast<unsigned>(ranks) >> 1);
            return (pair << 1 | color) ^ (row & 1);
        }

        /// @brief Calculate offset from typed pair and color dimensions
        /// @param pair_dim The pair dimension
        /// @param color_dim The color dimension
        /// @return Offset into the checkerboard grid
        DEVICE inline static constexpr OffsetDim compute_offset(PairDim pair_dim,
                                                                ColorDim color_dim) {
            return compute_offset(pair_dim.get(), color_dim.get());
        }

        DEVICE constexpr auto coordinates() const {
            return make_coordinates(get<OffsetDim>());
        }
    };
}

#endif
