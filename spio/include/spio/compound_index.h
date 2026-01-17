#ifndef SPIO_COMPOUND_INDEX_H_
#define SPIO_COMPOUND_INDEX_H_

#include "spio/macros.h"
#include "spio/compound_index_base.h"
#include "spio/dim.h"
#include "spio/dim_info.h"
#include "spio/meta.h"
#include "spio/coordinates.h"

namespace spio {

    /// Compound index for mapping a linear offset to multidimensional coordinates.
    ///
    /// The inverse of Tensor - maps a linear offset (like a thread index) back to
    /// typed dimension coordinates.
    ///
    /// Template parameters:
    ///   DimInfos   Dimension information types (same as in Tensor).
    template <typename... DimInfos> class CompoundIndex : public CompoundIndexBase<OFFSET> {
    public:
        // Total number of elements (product of all dimension sizes)
        static constexpr int total_size = detail::product_sizes<DimInfos...>::value;

        using CompoundIndexBase::CompoundIndexBase;

        /// Casts dimensions by replacing all occurrences of a base dimension.
        ///
        /// Replaces all dimensions with base type FromDim to use base type ToDim,
        /// preserving any fold structure.
        ///
        /// Template parameters:
        ///   FromDim   The base dimension to replace.
        ///   ToDim     The base dimension to replace with.
        template <typename FromDim, typename ToDim> DEVICE constexpr auto cast() const {
            using NewIndex = CompoundIndex<
                typename dim_traits::replace_dimension<FromDim, ToDim, DimInfos>::type...>;
            return NewIndex(offset());
        }

        /// Gets the typed coordinate for a specific dimension.
        ///
        /// Template parameters:
        ///   DimType   The dimension type to extract.
        ///
        /// Returns:
        ///   A typed dimension value.
        template <typename DimType> DEVICE constexpr auto get() const {
            using dim_info = typename dim_traits::find_dim_info<DimType, DimInfos...>::info;
            return dim_info::offset_to_dim(offset().get());
        }

        // Alternative method form if you prefer function syntax
        DEVICE static constexpr int size() {
            return total_size;
        }

        // Get size for a specific dimension
        template <typename DimType> DEVICE static constexpr DimType size() {
            return dim_traits::dimension_size<DimType, DimInfos...>::value;
        }

        /// Returns the number of dimensions in this index.
        DEVICE static constexpr int num_dims() {
            return sizeof...(DimInfos);
        }

        /// Converts this CompoundIndex to a Coordinates object.
        /// Returns Coordinates with Module types to preserve size/stride info for optimization.
        DEVICE constexpr auto coordinates() const {
            // get<DimType>() returns Module<base_dim, size, stride>
            // We need to deduce the return type from what get() returns
            return Coordinates<decltype(get<typename DimInfos::dim_type>())...>(
                get<typename DimInfos::dim_type>()...);
        }

        // ========================================================================
        // Partition iterator and range for cooperative iteration
        // ========================================================================

        /// Iterator that yields IndexType for each partitioned offset.
        template <typename IndexType, int Stride> class PartitionIterator {
        public:
            DEVICE constexpr PartitionIterator(int offset, int end) : _offset(offset), _end(end) {}

            DEVICE constexpr IndexType operator*() const {
                return IndexType(_offset);
            }

            DEVICE constexpr PartitionIterator& operator++() {
                _offset += Stride;
                return *this;
            }

            DEVICE constexpr bool operator!=(const PartitionIterator&) const {
                return _offset < _end;
            }

        private:
            int _offset;
            int _end;
        };

        /// A range that partitions an index's offset space.
        template <typename IndexType, int Stride> class PartitionRange {
        public:
            DEVICE constexpr PartitionRange(int start, int end) : _start(start), _end(end) {}

            DEVICE constexpr auto begin() const {
                return PartitionIterator<IndexType, Stride>(_start, _end);
            }

            DEVICE constexpr auto end() const {
                return PartitionIterator<IndexType, Stride>(_end, _end);
            }

        private:
            int _start;
            int _end;
        };

        /// Partitions this index's offset space by PartitionDim from partition_idx.
        ///
        /// Each thread handles offsets: start, start+stride, start+2*stride, ...
        ///
        /// Template parameters:
        ///   PartitionDim   The dimension to partition by (e.g., LANE).
        ///
        /// Parameters:
        ///   partition_idx   Provides get<PartitionDim>() for start and
        ///                   size<PartitionDim>() for stride.
        template <typename PartitionDim, typename PartitionIndexType>
        DEVICE static constexpr auto partition(PartitionIndexType partition_idx) {
            constexpr int stride = PartitionIndexType::template size<PartitionDim>().get();
            int start = partition_idx.template get<PartitionDim>().get();
            return PartitionRange<CompoundIndex, stride>(start, size());
        }

        // ========================================================================
        // Derived dimensions interface
        // ========================================================================
        //
        // Input: OFFSET (0 to total_size-1)
        // Output: All dimensions from DimInfos
        //
        // This allows CompoundIndex to be used as a derived dimension type,
        // enabling cursor subscripting with linear offsets that automatically
        // expand to multi-dimensional coordinates.

        using input_dims = detail::tuple<DimSize<OFFSET, total_size>>;
        using output_dims = detail::tuple<DimSize<typename DimInfos::dim_type, DimInfos::size>...>;
        using input_coordinates = Coordinates<OFFSET>;

        DEVICE static constexpr auto compute_coordinates(const input_coordinates& coords) {
            return CompoundIndex(coords.template get<OFFSET>()).coordinates();
        }
    };
}

#endif
