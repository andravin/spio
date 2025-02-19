#ifndef SPIO_CHECKERBOARD_IDX_H
#define SPIO_CHECKERBOARD_IDX_H

#include "spio/macros.h"

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
    template <int ranks>
    class CheckerboardIndex
    {
    public:
        DEVICE inline constexpr CheckerboardIndex(unsigned offset)
            : _offset(offset) {}

        DEVICE inline constexpr unsigned offset() const { return _offset; }

        DEVICE inline constexpr int pair() const
        {
            return _offset >> 1;
        }

        DEVICE inline constexpr int color() const
        {
            const unsigned row = _offset / ranks;
            return (_offset & 1) ^ (row & 1);
        }

        /// @param pair the index of a pair of grid elements.
        /// @param color the black (0) or white (1) grid element in the pair.
        /// @return Offset into the checkeboard grid.
        DEVICE inline static constexpr int offset(unsigned pair, unsigned color)
        {
            unsigned row = pair / (ranks / 2);
            return (pair << 1 | color) ^ (row & 1);
        }

    private:
        const unsigned _offset;
    };
}

#endif
