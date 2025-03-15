#ifndef SPIO_TENSOR_VARIADIC_H_
#define SPIO_TENSOR_VARIADIC_H_

#include "spio/macros.h"
#include "spio/dim.h"
#include <type_traits>
#include <tuple>

namespace spio
{
    // The private dimension type for linear offsets
    class _OffsetDim : public Dim
    {
    public:
        DEVICE constexpr _OffsetDim(int offset) : Dim(offset) {}
        
        // Allow addition of offset dimensions
        DEVICE constexpr _OffsetDim operator+( _OffsetDim other) const {
            return _OffsetDim(_add(other));
        }
    };
    
    // Forward declarations
    template <typename DataType>
    class TensorBase;
    
    // Store dimension information (user type, size, fold)
    template <typename DimType, int Size, unsigned Stride>
    struct DimInfo {
        using type = DimType;
        static constexpr DimType size = DimType(Size);
        
        // Create fold type for this dimension
        using fold_type = Fold<_OffsetDim, Stride>;
        
        // Map from dimension to offset.
        // Convert the user dimension to a folded tensor dimension
        // and unfold it.
        DEVICE constexpr static _OffsetDim to_offset(DimType d) {
            return fold_type(d.get()).unfold();
        }
    };
    
    // Base tensor class
    template <typename _data_type>
    class TensorBase {
    public:
        using data_type = _data_type;
        static constexpr int element_size = sizeof(_data_type);
        DEVICE constexpr TensorBase(_data_type *data = nullptr) : _data(data) {}
        DEVICE constexpr _data_type *get() const { return _data; }
        DEVICE void reset(_data_type *data) { _data = data; }
        DEVICE constexpr _data_type &operator*() const { return *_data; }
        DEVICE constexpr _data_type *operator->() const { return _data; }

    private:
        _data_type *_data;
    };
    
    // First, check if dimension exists in tensor
    template <typename DimType, typename... DimInfos>
    struct has_dim;

    template <typename DimType, typename FirstDimInfo, typename... RestDimInfos>
    struct has_dim<DimType, FirstDimInfo, RestDimInfos...> {
        static constexpr bool value = 
            std::is_same<DimType, typename FirstDimInfo::type>::value || 
            has_dim<DimType, RestDimInfos...>::value;
    };

    template <typename DimType>
    struct has_dim<DimType> {
        static constexpr bool value = false;
    };

    // Find dimension info without static assertion
    template <typename DimType, typename... DimInfos>
    struct find_dim_info_impl;

    template <typename DimType, typename FirstDimInfo, typename... RestDimInfos>
    struct find_dim_info_impl<DimType, FirstDimInfo, RestDimInfos...> {
        static constexpr bool is_match = std::is_same<DimType, typename FirstDimInfo::type>::value;

        using info = std::conditional_t<
            is_match,
            FirstDimInfo,
            typename find_dim_info_impl<DimType, RestDimInfos...>::info
        >;
    };

    // Base case with dummy DimInfo instantiation for error handling.
    template <typename DimType>
    struct find_dim_info_impl<DimType> {
        // Use DefaultDimInfo instead of void
        using info = DimInfo<DimType, 0, 1>;
    };

    // Public interface with static assertion first
    template <typename DimType, typename... DimInfos>
    struct find_dim_info {
        // First check if dimension exists - fail early with clear message
        static_assert(has_dim<DimType, DimInfos...>::value,
                     "Dimension type not found in tensor - ensure you're using the correct dimension type");
        
        // If we get here, dimension exists, so safe to use find_dim_info_impl
        using impl = find_dim_info_impl<DimType, DimInfos...>;
        using info = typename impl::info;
    };
    
    // Helper to update dimension info for slicing
    template <typename DimType, int NewSize, typename... DimInfos>
    struct update_dim_info;
    
    template <typename DimType, int NewSize, typename FirstInfo, typename... RestInfos>
    struct update_dim_info<DimType, NewSize, FirstInfo, RestInfos...> {
        // Check if this is the dimension to update
        static constexpr bool is_match = std::is_same<DimType, typename FirstInfo::type>::value;
        
        // Create updated info (either with new size or unchanged)
        using current = std::conditional_t<
            is_match,
            DimInfo<typename FirstInfo::type, NewSize, FirstInfo::fold_type::stride.get()>,
            FirstInfo
        >;
        
        // Recursive case: process rest of infos
        using next = typename update_dim_info<DimType, NewSize, RestInfos...>::type;
        
        // Combine with the rest
        using type = decltype(std::tuple_cat(
            std::tuple<current>(),
            std::declval<next>()
        ));
    };
    
    // Base case for update_dim_info
    template <typename DimType, int NewSize>
    struct update_dim_info<DimType, NewSize> {
        using type = std::tuple<>;
    };
    
    // Forward declare expansion helper
    template <typename, typename>
    struct expand_dim_infos;
    
    // Cursor with folded dimensions
    template <typename DataType, typename... DimInfos>
    class Cursor : public TensorBase<DataType>
    {
    public:
        using data_type = DataType;
        using TensorBase<data_type>::TensorBase;
        using TensorBase<data_type>::get;
        
        // Index with any dimension type
        template <typename DimType>
        DEVICE constexpr Cursor operator[](DimType d) const
        {
            // Get the offset for this dimension
            _OffsetDim offset = find_dim_info<DimType, DimInfos...>::info::to_offset(d);
            
            // Return new cursor at the offset position
            return Cursor(get() + offset.get());
        }
    };
    
    // Tensor class
    template <typename DataType, typename... DimInfos>
    class Tensor : public TensorBase<DataType>
    {
    public:
        using data_type = DataType;
        using TensorBase<data_type>::TensorBase;
        using TensorBase<data_type>::get;
        
        // Define cursor type
        using cursor_type = Cursor<DataType, DimInfos...>;
        
        // Get size for a specific dimension
        template <typename DimType>
        static constexpr DimType get_size() {
            return find_dim_info<DimType, DimInfos...>::info::size;
        }
        
        // Index with any dimension type
        template <typename DimType>
        DEVICE constexpr cursor_type operator[](DimType d) const {
            // Get the offset for this dimension
            _OffsetDim offset = find_dim_info<DimType, DimInfos...>::info::to_offset(d);
            
            // Return cursor at offset
            return cursor_type(get() + offset.get());
        }
        
        // Slice method to create a view with a different size in one dimension
        template <int NewSize, typename SliceDimType>
        DEVICE constexpr auto slice(SliceDimType slice_start) {
            // Create new dim infos with the specified dimension's size updated
            using updated_infos = typename update_dim_info<SliceDimType, NewSize, DimInfos...>::type;

            // Create new tensor type with the updated dim infos
            using result_type = typename expand_dim_infos<DataType, updated_infos>::type;
            
            // Use existing operator[] to get the cursor at the correct offset
            // and construct the new tensor at that position
            return result_type((*this)[slice_start].get());
        }
    };
    
    // Now define the expansion helper
    template <typename DataType, typename... InfoTypes>
    struct expand_dim_infos<DataType, std::tuple<InfoTypes...>> {
        using type = Tensor<DataType, InfoTypes...>;
    };
    
    // 2D tensor creation helper
    template <typename DataType, typename HeightDimType, typename WidthDimType,
              int HeightSize, int WidthSize>
    DEVICE constexpr auto make_tensor(DataType* data = nullptr) {
        return Tensor<
            DataType,
            DimInfo<HeightDimType, HeightSize, WidthSize>, // Height folded by width
            DimInfo<WidthDimType, WidthSize, 1>            // Width with unit stride
        >(data);
    }
    
    // Version with custom strides
    template <typename DataType, typename HeightDimType, typename WidthDimType,
              int HeightSize, int WidthSize, int HeightStride, int WidthStride>
    DEVICE constexpr auto make_tensor_with_strides(DataType* data = nullptr) {
        return Tensor<
            DataType,
            DimInfo<HeightDimType, HeightSize, HeightStride>,
            DimInfo<WidthDimType, WidthSize, WidthStride>
        >(data);
    }
}

#endif