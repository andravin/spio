#ifndef SPIO_TENSOR_VARIADIC_H_
#define SPIO_TENSOR_VARIADIC_H_

#include "spio/macros.h"
#include <type_traits>
#include <tuple>

namespace spio
{
    // Forward declarations
    template <typename DataType>
    class TensorBase;
    
    template <typename DataType, typename... DimTraits>
    class Tensor;
    
    // Store dimension traits (type, size, stride)
    template <typename DimType, int Size, int Stride>
    struct DimTrait {
        using type = DimType;
        static constexpr int size = Size;
        static constexpr int stride = Stride;
    };
    
    // Base tensor class
    template <typename _data_type>
    class TensorBase
    {
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
    
    // Helper to find which trait corresponds to a dimension type
    template <typename DimType, typename... Traits>
    struct find_trait;
    
    template <typename DimType, typename FirstTrait, typename... RestTraits>
    struct find_trait<DimType, FirstTrait, RestTraits...> {
        static constexpr bool is_match = std::is_same<DimType, typename FirstTrait::type>::value;
        
        static constexpr int stride = is_match ? 
            FirstTrait::stride : 
            find_trait<DimType, RestTraits...>::stride;
            
        static constexpr int size = is_match ?
            FirstTrait::size :
            find_trait<DimType, RestTraits...>::size;
    };
    
    // Base case - not found
    template <typename DimType>
    struct find_trait<DimType> {
        static constexpr int stride = 0; 
        static constexpr int size = 0;
    };
    
    // Helper to update a trait in a parameter pack
    template <typename DimType, int NewSize, typename... Traits>
    struct update_trait;
    
    template <typename DimType, int NewSize, typename Trait, typename... Rest>
    struct update_trait<DimType, NewSize, Trait, Rest...> {
        // Check if this is the dimension to update
        using current = std::conditional_t<
            std::is_same<DimType, typename Trait::type>::value,
            DimTrait<typename Trait::type, NewSize, Trait::stride>,
            Trait
        >;
        
        // Recursive case: process rest of traits
        using next = typename update_trait<DimType, NewSize, Rest...>::type;
        using type = decltype(std::tuple_cat(
            std::tuple<current>(),
            std::declval<next>()
        ));
    };
    
    // Base case
    template <typename DimType, int NewSize>
    struct update_trait<DimType, NewSize> {
        using type = std::tuple<>;
    };
    
    // Forward declare expand_traits
    template <typename, typename>
    struct expand_traits;
    
    // Cursor with dimension traits
    template <typename DataType, typename... DimTraits>
    class CursorWithTraits : public TensorBase<DataType> {
    public:
        using data_type = DataType;
        using TensorBase<data_type>::TensorBase;
        using TensorBase<data_type>::get;
        
        // Index with any dimension type
        template <typename DimType>
        DEVICE constexpr CursorWithTraits operator[](DimType d) const {
            // Find the stride for this dimension type
            constexpr int stride = find_trait<DimType, DimTraits...>::stride;
            return CursorWithTraits(get() + d.get() * stride);
        }
    };
    
    // Tensor class with variadic dimensions
    template <typename DataType, typename... DimTraits>
    class Tensor : public TensorBase<DataType> {
    public:
        using data_type = DataType;
        using TensorBase<data_type>::TensorBase;
        using TensorBase<data_type>::get;
        
        // Define cursor type
        using cursor_type = CursorWithTraits<DataType, DimTraits...>;

        template <typename DimType>
        static constexpr int get_size() {
            return find_trait<DimType, DimTraits...>::size;
        }
        
        
        // Generate indexing operators for each dimension
        template <typename DimType>
        DEVICE constexpr cursor_type operator[](DimType d) const {
            constexpr int stride = find_trait<DimType, DimTraits...>::stride;
            return cursor_type(get() + d.get() * stride);
        }
        
        // Truly generic slice method
        template <int NewSize, typename SliceDimType>
        DEVICE constexpr auto slice(SliceDimType slice_start) {
            // Create new traits with the specified dimension's size updated
            using updated_traits = typename update_trait<SliceDimType, NewSize, DimTraits...>::type;
            
            // Create new tensor type with the updated traits
            using result_type = typename expand_traits<DataType, updated_traits>::type;
            
            // Return the new tensor
            return result_type((*this)[slice_start].get());
        }
    };
    
    // Now define expand_traits after Tensor is defined
    template <typename DataType, typename... Traits>
    struct expand_traits<DataType, std::tuple<Traits...>> {
        using type = Tensor<DataType, Traits...>;
    };
    
    // // Tensor creation helper
    // template <typename DataType, typename Dim0, typename Dim1, int Size0, int Size1>
    // constexpr auto make_tensor(DataType* data = nullptr) {
    //     return Tensor<DataType, 
    //                  DimTrait<Dim0, Size0, Size1>, 
    //                  DimTrait<Dim1, Size1, 1>>(data);
    // }

    // Tensor creation helper with optional custom strides
    template <typename DataType, typename Dim0, typename Dim1, int Size0, int Size1, 
              int Stride0 = Size1, int Stride1 = 1>
    constexpr auto make_tensor(DataType* data = nullptr) {
        return Tensor<DataType, 
                     DimTrait<Dim0, Size0, Stride0>, 
                     DimTrait<Dim1, Size1, Stride1>>(data);
    }
}

#endif