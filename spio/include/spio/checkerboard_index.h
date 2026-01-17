#ifndef SPIO_CHECKERBOARD_IDX_H
#define SPIO_CHECKERBOARD_IDX_H

#include "spio/macros.h"
#include "spio/dim.h"
#include "spio/dim_info.h"
#include "spio/compound_index_base.h"
#include "spio/coordinates.h"

namespace spio {

    /// Index class for checkerboard memory layout.
    ///
    /// Arranges an N x 2 matrix into a checkerboard grid with the given number
    /// of banks (grid columns). CompoundIndex (x, color) is the x and color index
    /// of the element in the grid.
    ///
    /// Cells with the same color are never adjacent, making this useful for storing
    /// matrices that can be accessed both row-wise and column-wise without bank conflicts.
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
    /// Template parameters:
    ///   ranks      Number of column-ranks in the checkerboard.
    ///   PairDim    Dimension type for the pair index.
    ///   ColorDim   Dimension type for the color index.
    ///   OffsetDim  Dimension type for the linear offset.
    ///   Size       Total number of elements (default 32).
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

        /// Constructs from a compatible dimension type.
        template <typename DimLike,
                  detail::enable_if_t<detail::is_dim_like_v<DimLike> &&
                                          detail::dims_compatible_v<DimLike, OffsetDim>,
                                      int> = 0>
        DEVICE constexpr CheckerboardIndex(DimLike dim) : CompoundIndexBase<OffsetDim>(dim) {}

        /// Constructs index from pair and color dimensions.
        DEVICE inline constexpr CheckerboardIndex(PairDim pair_dim, ColorDim color_dim)
            : CompoundIndexBase<OffsetDim>(compute_offset(pair_dim.get(), color_dim.get())) {}

        /// Constructs from any index type with get<PairDim>() and get<ColorDim>() methods.
        ///
        /// Handles fragment load indices that have typed getters but not coordinates().
        template <typename IndexType,
                  detail::enable_if_t<!detail::is_dim_like_v<IndexType> &&
                                          !detail::dims_compatible_v<IndexType, OffsetDim>,
                                      int> = 0>
        DEVICE inline constexpr CheckerboardIndex(IndexType idx)
            : CompoundIndexBase<OffsetDim>(compute_offset(idx.template get<PairDim>().get(),
                                                          idx.template get<ColorDim>().get())) {}

        /// Gets dimension value by type.
        ///
        /// Template parameters:
        ///   Dim   The dimension type to retrieve.
        ///
        /// Returns:
        ///   The dimension value with the proper type.
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

        /// Calculates offset from pair and color indices.
        ///
        /// Parameters:
        ///   pair    Index of a pair of grid elements.
        ///   color   The black (0) or white (1) grid element in the pair.
        ///
        /// Returns:
        ///   Offset into the checkerboard grid.
        DEVICE inline static constexpr OffsetDim compute_offset(int pair, int color) {
            int row = static_cast<unsigned>(pair) / (static_cast<unsigned>(ranks) >> 1);
            return (pair << 1 | color) ^ (row & 1);
        }

        /// Calculates offset from typed pair and color dimensions.
        ///
        /// Parameters:
        ///   pair_dim    The pair dimension.
        ///   color_dim   The color dimension.
        ///
        /// Returns:
        ///   Offset into the checkerboard grid.
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
