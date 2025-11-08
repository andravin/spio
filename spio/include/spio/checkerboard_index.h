#ifndef SPIO_CHECKERBOARD_IDX_H
#define SPIO_CHECKERBOARD_IDX_H

#include "spio/macros.h"
#include "spio/dim.h"
#include "spio/index_base.h"

namespace spio
{
    /// @brief An index class for checkerboard memory layout.
    ///
    /// Arrange an N x 2 matrix into a checkerboard grid with the given number
    /// of banks (i.e., grid columns).
    ///
    /// Index (x, color) is the x and color index of the element in the grid.
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
    ///             0        2        4        6
    ///         +--------+--------+--------+--------+
    ///       0 | (0, 0) | (0, 1) | (1, 0) | (1, 1) |
    ///         +--------+--------+--------+--------+
    ///       8 | (2, 1) | (2, 0) | (3, 1) | (3, 0) |
    ///         + --------+--------+--------+--------
    ///      16 | (4, 0) | (4, 1) | (5, 0) | (5, 1) |
    ///         +--------+--------+--------+--------+
    ///      24 | (6, 1) | (6, 0) | (7, 1) | (7, 0) |
    /// Row     +--------+--------+--------+--------+
    ///      32 | (8, 0) | (8, 1) | (9, 0) | (9, 1) |
    ///         +--------+--------+--------+--------+
    ///      40 |(10, 1) |(10, 0) |(11, 1) |(11, 0) |
    ///       + --------+--------+--------+--------
    ///      48 |(12, 0) |(12, 1) |(13, 0) |(13, 1) |
    ///         +--------+--------+--------+--------+
    ///      56 |(14, 1) |(14, 0) |(15, 1) |(15, 0) |
    ///         +--------+--------+--------+--------+
    ///
    /// @tparam ranks the number of column-ranks in the checkerboard.
    /// @tparam PairDim the dimension type for the pair index
    /// @tparam ColorDim the dimension type for the color index
    template <int ranks, typename PairDim, typename ColorDim, typename OffsetDim>
    class CheckerboardIndex : public IndexBase
    {
    public:
        using IndexBase::IndexBase;

        /// @brief Construct index from pair and color dimensions
        /// @param pair_dim The pair dimension value
        /// @param color_dim The color dimension value
        DEVICE inline constexpr CheckerboardIndex(PairDim pair_dim, ColorDim color_dim)
            : IndexBase(compute_offset(pair_dim.get(), color_dim.get())) {}
        
        /// @brief Get dimension value by type
        /// @tparam Dim The dimension type to retrieve
        /// @return The dimension value with the proper type
        template <typename Dim>
        DEVICE constexpr auto get() const {
            if constexpr (std::is_same_v<Dim, PairDim>) {
                return PairDim(offset() >> 1);
            } else if constexpr (std::is_same_v<Dim, ColorDim>) {
                const int row = static_cast<unsigned>(offset()) / static_cast<unsigned>(ranks);
                return ColorDim((offset() & 1) ^ (row & 1));
            } else if constexpr (std::is_same_v<Dim, OffsetDim>) {
                return OffsetDim(offset());
            } else {
                static_assert(
                    std::is_same_v<Dim, PairDim> || 
                    std::is_same_v<Dim, ColorDim>,
                    "Invalid dimension type for CheckerboardIndex"
                );
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
        DEVICE inline static constexpr int compute_offset(int pair, int color)
        {
            int row = static_cast<unsigned>(pair) / (static_cast<unsigned>(ranks) >> 1);
            return (pair << 1 | color) ^ (row & 1);
        }
        
        /// @brief Calculate offset from typed pair and color dimensions
        /// @param pair_dim The pair dimension
        /// @param color_dim The color dimension
        /// @return Offset into the checkerboard grid
        DEVICE inline static constexpr int compute_offset(PairDim pair_dim, ColorDim color_dim)
        {
            return compute_offset(pair_dim.get(), color_dim.get());
        }
    };
}

#endif
