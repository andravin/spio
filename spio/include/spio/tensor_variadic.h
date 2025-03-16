#ifndef SPIO_TENSOR_VARIADIC_H_
#define SPIO_TENSOR_VARIADIC_H_

#include "spio/macros.h"
#include "spio/dim.h"
#include "spio/dim_info.h"
#include "spio/index_variadic.h"

namespace spio
{

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
        // Helper function to concatenate tuples
        template<typename T1, typename T2>
        struct tuple_cat_impl;
        
        // Specialization for empty first tuple
        template<typename... Ts>
        struct tuple_cat_impl<spio::detail::tuple<>, spio::detail::tuple<Ts...>> {
            using type = spio::detail::tuple<Ts...>;
            
            template<typename... Args>
            DEVICE static constexpr spio::detail::tuple<Ts...> concat(
                const spio::detail::tuple<>&,
                const spio::detail::tuple<Ts...>& t2,
                Args&&... args) {
                return t2;
            }
        };
        
        // Specialization for non-empty first tuple
        template<typename T, typename... Ts, typename... Us>
        struct tuple_cat_impl<spio::detail::tuple<T, Ts...>, spio::detail::tuple<Us...>> {
            using rest_cat = typename tuple_cat_impl<spio::detail::tuple<Ts...>, spio::detail::tuple<Us...>>::type;
            using type = spio::detail::tuple<T, Us...>;
            
            template<typename... Args>
            DEVICE static constexpr type concat(
                const spio::detail::tuple<T, Ts...>& t1,
                const spio::detail::tuple<Us...>& t2,
                Args&&... args) {
                return type(t1.first, tuple_cat_impl<spio::detail::tuple<Ts...>, 
                           spio::detail::tuple<Us...>>::concat(t1.rest, t2, args...));
            }
        };
        
        // Helper function to concatenate two tuples
        template<typename T1, typename T2>
        using tuple_cat_t = typename tuple_cat_impl<T1, T2>::type;
        
        template<typename T1, typename T2>
        DEVICE constexpr auto tuple_cat(const T1& t1, const T2& t2) {
            return tuple_cat_impl<T1, T2>::concat(t1, t2);
        }

        /// @brief Update dimension info by replacing a given dimension with a new size.
        /// @tparam DimType the dimension type to update
        /// @tparam SliceSize the new size of the dimension
        /// @tparam DimInfos the dimension infos
        template <typename DimType, int SliceSize, typename... DimInfos>
        struct update_dim_info;

        template <typename DimType, int SliceSize, typename FirstInfo, typename... RestInfos>
        struct update_dim_info<DimType, SliceSize, FirstInfo, RestInfos...>
        {
            static constexpr bool is_match = detail::is_same<DimType, typename FirstInfo::dim_type>::value;
            using current = detail::conditional_t<
                is_match,
                DimInfo<typename FirstInfo::dim_type, SliceSize, FirstInfo::module_type::stride.get()>,
                FirstInfo>;
            using next = typename update_dim_info<DimType, SliceSize, RestInfos...>::dim_type;
            using dim_type = tuple_cat_t<
                spio::detail::tuple<current>,
                next>;
        };

        template <typename DimType, int SliceSize>
        struct update_dim_info<DimType, SliceSize>
        {
            using dim_type = spio::detail::tuple<>;
        };

        template <typename, typename>
        struct tensor_type_from_dim_info_tuple;

        /// @brief Create a tensor type from a tuple of dimension infos.
        /// @tparam DataType the data type of the tensor
        /// @tparam DimInfos the dimension infos
        template <typename DataType, typename... DimInfos>
        struct tensor_type_from_dim_info_tuple<DataType, spio::detail::tuple<DimInfos...>>
        {
            using tensor_type = Tensor<DataType, DimInfos...>;
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
            _OffsetDim offset = dim_traits::find_dim_info<DimType, DimInfos...>::info::to_offset(d);

            // Return new cursor at the offset position
            return Cursor(get() + offset.get());
        }
        
        /// @brief Subscript operator that takes an Index object and applies all dimensions
        /// @tparam IndexDimInfos The dimension infos in the Index
        /// @param idx The index containing coordinates for multiple dimensions
        /// @return A cursor pointing to the element at the specified position
        template <typename... IndexDimInfos>
        DEVICE constexpr Cursor operator[](const Index<IndexDimInfos...>& idx) const
        {
            // Create a cursor pointing to the right place by applying each dimension
            data_type* result = get();
            
            // Apply offsets for each dimension (ignoring dimensions not in tensor)
            int dummy[] = {0, ((
                result += apply_dimension<typename IndexDimInfos::dim_type>(idx)
            ), 0)...};
            (void)dummy; // Suppress unused variable warning
            
            return Cursor(result);
        }
        
    private:
        // Helper to apply a single dimension if it exists in the tensor
        template <typename DimType, typename... IndexDimInfos>
        DEVICE constexpr int apply_dimension(const Index<IndexDimInfos...>& idx) const {
            // Check if this dimension exists in our tensor using tag dispatching
            // instead of constexpr if (which isn't supported in all CUDA versions)
            return apply_dimension_impl<DimType>(
                idx, 
                spio::detail::conditional_t<
                    dim_traits::has_dimension<DimType, DimInfos...>::value,
                    spio::detail::true_type,
                    spio::detail::false_type
                >()
            );
        }
        
        // Implementation for when dimension exists
        template <typename DimType, typename... IndexDimInfos, typename = void>
        DEVICE constexpr int apply_dimension_impl(
            const Index<IndexDimInfos...>& idx, 
            const spio::detail::true_type&
        ) const {
            // Get dimension value and apply offset
            auto dim_value = idx.template get<DimType>();
            return dim_traits::find_dim_info<DimType, DimInfos...>::info::to_offset(dim_value).get();
        }
        
        // Implementation for when dimension doesn't exist
        template <typename DimType, typename... IndexDimInfos>
        DEVICE constexpr int apply_dimension_impl(
            const Index<IndexDimInfos...>&, 
            const spio::detail::false_type&
        ) const {
            // Dimension doesn't exist in this tensor, no offset change
            return 0;
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
        DEVICE static constexpr DimType get_size()
        {
            return dim_traits::dimension_size<DimType, DimInfos...>::value;
        }

        /// @brief Subscript operator with any dimension type.
        /// @tparam DimType the dimension to apply the subscript index to.
        /// @return a Cursor that points to the element at the specified position.
        template <typename DimType>
        DEVICE constexpr cursor_type operator[](DimType d) const
        {
            _OffsetDim offset = dim_traits::find_dim_info<DimType, DimInfos...>::info::to_offset(d);
            return cursor_type(get() + offset.get());
        }
        
        /// @brief Subscript operator that takes an Index object and applies all dimensions
        /// @tparam IndexDimInfos The dimension infos in the Index
        /// @param idx The index containing coordinates for multiple dimensions
        /// @return A cursor pointing to the element at the specified position
        template <typename... IndexDimInfos>
        DEVICE constexpr cursor_type operator[](const Index<IndexDimInfos...>& idx) const
        {
            // Create a cursor pointing to the right place by applying each dimension
            data_type* result = get();
            
            // Apply offsets for each dimension (ignoring dimensions not in tensor)
            int dummy[] = {0, ((
                result += apply_dimension<typename IndexDimInfos::dim_type>(idx)
            ), 0)...};
            (void)dummy; // Suppress unused variable warning
            
            return cursor_type(result);
        }
        
    private:
        // Helper to apply a single dimension if it exists in the tensor
        template <typename DimType, typename... IndexDimInfos>
        DEVICE constexpr int apply_dimension(const Index<IndexDimInfos...>& idx) const {
            // Check if this dimension exists in our tensor using tag dispatching
            // instead of constexpr if (which isn't supported in all CUDA versions)
            return apply_dimension_impl<DimType>(
                idx, 
                spio::detail::conditional_t<
                    dim_traits::has_dimension<DimType, DimInfos...>::value,
                    spio::detail::true_type,
                    spio::detail::false_type
                >()
            );
        }
        
        // Implementation for when dimension exists
        template <typename DimType, typename... IndexDimInfos, typename = void>
        DEVICE constexpr int apply_dimension_impl(
            const Index<IndexDimInfos...>& idx, 
            const spio::detail::true_type&
        ) const {
            // Get dimension value and apply offset
            auto dim_value = idx.template get<DimType>();
            return dim_traits::find_dim_info<DimType, DimInfos...>::info::to_offset(dim_value).get();
        }
        
        // Implementation for when dimension doesn't exist
        template <typename DimType, typename... IndexDimInfos>
        DEVICE constexpr int apply_dimension_impl(
            const Index<IndexDimInfos...>&, 
            const spio::detail::false_type&
        ) const {
            // Dimension doesn't exist in this tensor, no offset change
            return 0;
        }
    
    public:
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