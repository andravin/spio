#ifndef SPIO_TENSOR_VARIADIC_H_
#define SPIO_TENSOR_VARIADIC_H_

#include "spio/macros.h"
#include "spio/dim.h"
#include <type_traits>
#include <tuple>

namespace spio
{
    /// @brief The private dimension type for linear offsets.
    class _OffsetDim : public Dim
    {
    public:
        DEVICE constexpr _OffsetDim(int offset) : Dim(offset) {}

        // Allow addition of offset dimensions
        DEVICE constexpr _OffsetDim operator+(_OffsetDim other) const
        {
            return _OffsetDim(_add(other));
        }
    };

    /// @brief Store information about a tensor dimension.
    /// @tparam DimType the dimension type
    /// @tparam Size the size of the dimension
    /// @tparam Stride the stride of the dimension
    template <typename DimType, int Size, unsigned Stride>
    struct DimInfo
    {
        using dim_type = DimType;
        static constexpr DimType size = DimType(Size);

        /// @brief  How this dimension folds the tensor's linear offset dimension.
        using fold_type = Fold<_OffsetDim, Stride>;

        /// @brief Map from dimension to offset.
        /// Convert a dimension index to a folded offset dimension and unfold it.
        /// @param d the dimension
        DEVICE constexpr static _OffsetDim to_offset(DimType d)
        {
            return fold_type(d.get()).unfold();
        }
    };

    /// @brief Base class for tensor data.
    template <typename _data_type>
    class Data
    {
    public:
        using data_type = _data_type;
        static constexpr int element_size = sizeof(_data_type);
        DEVICE constexpr Data(_data_type *data = nullptr) : _data(data) {}
        DEVICE constexpr _data_type *get() const { return _data; }
        DEVICE void reset(_data_type *data) { _data = data; }
        DEVICE constexpr _data_type &operator*() const { return *_data; }
        DEVICE constexpr _data_type *operator->() const { return _data; }

    private:
        _data_type *_data;
    };

    // Forward declaration of Tensor class
    template <typename DataType, typename... DimInfos>
    class Tensor;

    // Implementation details
    namespace detail
    {
        /// @brief Check if a dimension exists in the tensor.
        /// @tparap DimType the dimension type to check
        /// @tparam DimInfos the dimension infos
        template <typename DimType, typename... DimInfos>
        struct has_dim;

        template <typename DimType, typename FirstDimInfo, typename... RestDimInfos>
        struct has_dim<DimType, FirstDimInfo, RestDimInfos...>
        {
            static constexpr bool value =
                std::is_same<DimType, typename FirstDimInfo::dim_type>::value ||
                has_dim<DimType, RestDimInfos...>::value;
        };

        template <typename DimType>
        struct has_dim<DimType>
        {
            static constexpr bool value = false;
        };

        template <typename DimType, typename... DimInfos>
        struct find_dim_info_impl;

        template <typename DimType, typename FirstDimInfo, typename... RestDimInfos>
        struct find_dim_info_impl<DimType, FirstDimInfo, RestDimInfos...>
        {
            static constexpr bool is_match = std::is_same<DimType, typename FirstDimInfo::dim_type>::value;
            using info = std::conditional_t<
                is_match,
                FirstDimInfo,
                typename find_dim_info_impl<DimType, RestDimInfos...>::info>;
        };

        /// @brief Base case with dummy DimInfo instantiation for error handling.
        template <typename DimType>
        struct find_dim_info_impl<DimType>
        {
            using info = DimInfo<DimType, 0, 1>;
        };

        /// @brief Update dimension info by replacing a given dimension with a new size.
        /// @tparam DimType the dimension type to update
        /// @tparam SliceSize the new size of the dimension
        /// @tparam DimInfos the dimension infos
        template <typename DimType, int SliceSize, typename... DimInfos>
        struct update_dim_info;

        template <typename DimType, int SliceSize, typename FirstInfo, typename... RestInfos>
        struct update_dim_info<DimType, SliceSize, FirstInfo, RestInfos...>
        {
            static constexpr bool is_match = std::is_same<DimType, typename FirstInfo::dim_type>::value;
            using current = std::conditional_t<
                is_match,
                DimInfo<typename FirstInfo::dim_type, SliceSize, FirstInfo::fold_type::stride.get()>,
                FirstInfo>;
            using next = typename update_dim_info<DimType, SliceSize, RestInfos...>::dim_type;
            using dim_type = decltype(std::tuple_cat(
                std::tuple<current>(),
                std::declval<next>()));
        };

        template <typename DimType, int SliceSize>
        struct update_dim_info<DimType, SliceSize>
        {
            using dim_type = std::tuple<>;
        };

        template <typename, typename>
        struct tensor_type_from_dim_info_tuple;

        /// @brief Create a tensor type from a tuple of dimension infos.
        /// @tparam DataType the data type of the tensor
        /// @tparam DimInfos the dimension infos
        template <typename DataType, typename... DimInfos>
        struct tensor_type_from_dim_info_tuple<DataType, std::tuple<DimInfos...>>
        {
            using tensor_type = Tensor<DataType, DimInfos...>;
        };
    }

    // Type traits for tensor operations
    namespace tensor_traits
    {
        /// @brief Find dimension info for a given dimension type.
        /// @tparam DimType the dimension type to find
        /// @tparam ...DimInfos the dimension infos
        template <typename DimType, typename... DimInfos>
        struct find_dim_info
        {
            // First check if dimension exists and show a clear error message if it doesn't.
            static_assert(detail::has_dim<DimType, DimInfos...>::value,
                          "Dimension type not found in tensor - ensure you're using the correct dimension type");

            // Then find the dimension info.
            using impl = detail::find_dim_info_impl<DimType, DimInfos...>;
            using info = typename impl::info;
        };

        /// @brief Check if a dimension exists in the tensor (public interface).
        /// @tparam DimType the dimension type to check
        /// @tparam DimInfos the dimension infos
        template <typename DimType, typename... DimInfos>
        struct has_dimension
        {
            static constexpr bool value = detail::has_dim<DimType, DimInfos...>::value;
        };

        /// @brief Get the size of a specific dimension.
        /// @tparam DimType the dimension type
        /// @tparam DimInfos the dimension infos
        template <typename DimType, typename... DimInfos>
        struct dimension_size
        {
            static constexpr DimType value = find_dim_info<DimType, DimInfos...>::info::size;
        };
    }

    /// @brief Cursor with folded dimensions.
    /// Cursor is a class that represents a position in a tensor. It provides a subscript
    /// operator to access elements at a specific index in a given dimension.
    /// @tparam DataType the data type of the tensor
    /// @tparam DimInfos the dimension infos
    template <typename DataType, typename... DimInfos>
    class Cursor : public Data<DataType>
    {
    public:
        using data_type = DataType;
        using Data<data_type>::Data;
        using Data<data_type>::get;

        /// @brief Subscript operator that returns a new Cursor at the specified dimension index.
        /// @tparam DimType the dimension to apply the subscript index to.
        /// @param d the subscript index.
        /// @return a new Cursor that points to the element at the specified dimension index.
        template <typename DimType>
        DEVICE constexpr Cursor operator[](DimType d) const
        {
            // Get the offset for this dimension
            _OffsetDim offset = tensor_traits::find_dim_info<DimType, DimInfos...>::info::to_offset(d);

            // Return new cursor at the offset position
            return Cursor(get() + offset.get());
        }
    };

    /// @brief Tensor class.
    /// Tensor is a class that represents a multi-dimensional array. It provides
    /// a subscript operator to access elements at a specific position. It also
    /// provides methods to get the size of a specific dimension and to slice the
    /// tensor along a specific dimension.
    ///
    /// Tensor uses "typed dimensions" to provide compile-time checks for dimension
    /// sizes and strides. Each dimension is a unique subclass of Dim. The subscript
    /// and slice methods are overloaded to by dimension type, so any attempt to
    /// use it is not possible to accidentally use a dimension index with the wrong dimension.
    ///
    /// Dim encapsulates an integer index and Dim subclasses implement arithmetic and comparison
    /// operators, so it is possible to add dimensions and compare them. But any attempt to
    /// use add or compare different dimension types will result in a compile-time error.

    /// @tparam DataType the data type of the tensor
    /// @tparam DimInfos the dimension infos
    template <typename DataType, typename... DimInfos>
    class Tensor : public Data<DataType>
    {
    public:
        using data_type = DataType;
        using Data<data_type>::Data;
        using Data<data_type>::get;

        using cursor_type = Cursor<DataType, DimInfos...>;

        // Get size for a specific dimension
        template <typename DimType>
        static constexpr DimType get_size()
        {
            return tensor_traits::dimension_size<DimType, DimInfos...>::value;
        }

        /// @brief Subscript operator with any dimension type.
        /// @tparam DimType the dimension to apply the subscript index to.
        /// @return a Cursor that points to the element at the specified position.
        template <typename DimType>
        DEVICE constexpr cursor_type operator[](DimType d) const
        {
            _OffsetDim offset = tensor_traits::find_dim_info<DimType, DimInfos...>::info::to_offset(d);
            return cursor_type(get() + offset.get());
        }

        /// @brief Slice method to create a view with a different offset and size in one dimension.
        /// @tparam SliceSize the new size of the dimension
        /// @tparam SliceDimType the dimension to slice. SliceDimType is inferred from the type of the slice_start argument.
        /// @param slice_start the start index of the slice.
        /// @return a new Tensor that is a view of the original tensor with the specified dimension's size updated.
        template <int SliceSize, typename SliceDimType>
        DEVICE constexpr auto slice(SliceDimType slice_start)
        {
            using updated_infos = typename detail::update_dim_info<SliceDimType, SliceSize, DimInfos...>::dim_type;
            using tensor_type = typename detail::tensor_type_from_dim_info_tuple<DataType, updated_infos>::tensor_type;
            return tensor_type((*this)[slice_start].get());
        }
    };
}

#endif