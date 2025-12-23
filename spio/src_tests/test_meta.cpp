#include "dim_test_types.h"

#include "spio/meta.h"

using namespace spio;
using namespace spio::detail;

// ============================================================================
// bool_constant, true_type, false_type
// ============================================================================

UTEST(Meta, bool_constant_true) {
    EXPECT_TRUE(bool_constant<true>::value);
}

UTEST(Meta, bool_constant_false) {
    EXPECT_FALSE(bool_constant<false>::value);
}

UTEST(Meta, true_type_value) {
    EXPECT_TRUE(true_type::value);
}

UTEST(Meta, false_type_value) {
    EXPECT_FALSE(false_type::value);
}

// ============================================================================
// is_same
// ============================================================================

UTEST(Meta, is_same_true) {
    EXPECT_TRUE((is_same<int, int>::value));
    EXPECT_TRUE((is_same<I, I>::value));
    EXPECT_TRUE((is_same<const float, const float>::value));
}

UTEST(Meta, is_same_false) {
    EXPECT_FALSE((is_same<int, float>::value));
    EXPECT_FALSE((is_same<I, J>::value));
    EXPECT_FALSE((is_same<int, const int>::value));
}

// ============================================================================
// conditional / conditional_t
// ============================================================================

UTEST(Meta, conditional_true) {
    using Result = conditional_t<true, int, float>;
    EXPECT_TRUE((is_same<Result, int>::value));
}

UTEST(Meta, conditional_false) {
    using Result = conditional_t<false, int, float>;
    EXPECT_TRUE((is_same<Result, float>::value));
}

// ============================================================================
// remove_reference / remove_reference_t
// ============================================================================

UTEST(Meta, remove_reference_plain) {
    using Result = remove_reference_t<int>;
    EXPECT_TRUE((is_same<Result, int>::value));
}

UTEST(Meta, remove_reference_lvalue) {
    using Result = remove_reference_t<int&>;
    EXPECT_TRUE((is_same<Result, int>::value));
}

UTEST(Meta, remove_reference_rvalue) {
    using Result = remove_reference_t<int&&>;
    EXPECT_TRUE((is_same<Result, int>::value));
}

UTEST(Meta, remove_reference_const_lvalue) {
    using Result = remove_reference_t<const int&>;
    EXPECT_TRUE((is_same<Result, const int>::value));
}

// ============================================================================
// remove_const / remove_const_t
// ============================================================================

UTEST(Meta, remove_const_plain) {
    using Result = remove_const_t<int>;
    EXPECT_TRUE((is_same<Result, int>::value));
}

UTEST(Meta, remove_const_const) {
    using Result = remove_const_t<const int>;
    EXPECT_TRUE((is_same<Result, int>::value));
}

UTEST(Meta, remove_const_reference_to_const) {
    // const is "inside" the reference, not top-level
    using Result = remove_const_t<const int&>;
    EXPECT_TRUE((is_same<Result, const int&>::value));
}

// ============================================================================
// decay / decay_t
// ============================================================================

UTEST(Meta, decay_plain) {
    using Result = decay_t<int>;
    EXPECT_TRUE((is_same<Result, int>::value));
}

UTEST(Meta, decay_reference) {
    using Result = decay_t<int&>;
    EXPECT_TRUE((is_same<Result, int>::value));
}

UTEST(Meta, decay_const) {
    using Result = decay_t<const int>;
    EXPECT_TRUE((is_same<Result, int>::value));
}

UTEST(Meta, decay_const_reference) {
    using Result = decay_t<const int&>;
    EXPECT_TRUE((is_same<Result, int>::value));
}

UTEST(Meta, decay_const_rvalue_reference) {
    using Result = decay_t<const int&&>;
    EXPECT_TRUE((is_same<Result, int>::value));
}

// ============================================================================
// tuple
// ============================================================================

UTEST(Meta, tuple_empty_construction) {
    tuple<> t;
    (void)t; // Just verify it compiles
    EXPECT_TRUE(true);
}

UTEST(Meta, tuple_single_element) {
    tuple<int> t(42);
    EXPECT_EQ(t.first, 42);
}

UTEST(Meta, tuple_multiple_elements) {
    tuple<int, float, double> t(1, 2.5f, 3.0);
    EXPECT_EQ(t.first, 1);
    EXPECT_EQ(t.rest.first, 2.5f);
    EXPECT_EQ(t.rest.rest.first, 3.0);
}

// ============================================================================
// tuple_size
// ============================================================================

UTEST(Meta, tuple_size_empty) {
    EXPECT_EQ(tuple_size<tuple<>>::value, 0u);
}

UTEST(Meta, tuple_size_one) {
    EXPECT_EQ(tuple_size<tuple<int>>::value, 1u);
}

UTEST(Meta, tuple_size_three) {
    using Tuple3 = tuple<int, float, double>;
    EXPECT_EQ(tuple_size<Tuple3>::value, 3u);
}

// ============================================================================
// get (index-based)
// ============================================================================

UTEST(Meta, get_first) {
    tuple<int, float, double> t(1, 2.5f, 3.0);
    EXPECT_EQ(get<0>(t), 1);
}

UTEST(Meta, get_second) {
    tuple<int, float, double> t(1, 2.5f, 3.0);
    EXPECT_EQ(get<1>(t), 2.5f);
}

UTEST(Meta, get_third) {
    tuple<int, float, double> t(1, 2.5f, 3.0);
    EXPECT_EQ(get<2>(t), 3.0);
}

UTEST(Meta, get_mutable) {
    tuple<int, float> t(1, 2.5f);
    get<0>(t) = 100;
    EXPECT_EQ(get<0>(t), 100);
}

UTEST(Meta, get_const) {
    const tuple<int, float> t(1, 2.5f);
    EXPECT_EQ(get<0>(t), 1);
    EXPECT_EQ(get<1>(t), 2.5f);
}

// ============================================================================
// tuple_contains
// ============================================================================

UTEST(Meta, tuple_contains_true) {
    using Tuple3 = tuple<float, int, double>;
    EXPECT_TRUE((tuple_contains<int, Tuple3>::value));
}

UTEST(Meta, tuple_contains_false) {
    using Tuple3 = tuple<float, int, long>;
    EXPECT_FALSE((tuple_contains<double, Tuple3>::value));
}

UTEST(Meta, tuple_contains_first) {
    using Tuple3 = tuple<float, int, double>;
    EXPECT_TRUE((tuple_contains<float, Tuple3>::value));
}

UTEST(Meta, tuple_contains_last) {
    using Tuple3 = tuple<float, int, double>;
    EXPECT_TRUE((tuple_contains<double, Tuple3>::value));
}

UTEST(Meta, tuple_contains_empty_tuple) {
    EXPECT_FALSE((tuple_contains<int, tuple<>>::value));
}

// ============================================================================
// tuple_get_by_type
// ============================================================================

UTEST(Meta, tuple_get_by_type_first) {
    tuple<I, J, K> t(I(1), J(2), K(3));
    EXPECT_EQ(tuple_get_by_type<I>(t).get(), 1);
}

UTEST(Meta, tuple_get_by_type_middle) {
    tuple<I, J, K> t(I(1), J(2), K(3));
    EXPECT_EQ(tuple_get_by_type<J>(t).get(), 2);
}

UTEST(Meta, tuple_get_by_type_last) {
    tuple<I, J, K> t(I(1), J(2), K(3));
    EXPECT_EQ(tuple_get_by_type<K>(t).get(), 3);
}

UTEST(Meta, tuple_get_by_type_mutable) {
    tuple<int, float> t(1, 2.5f);
    tuple_get_by_type<int>(t) = 100;
    EXPECT_EQ(tuple_get_by_type<int>(t), 100);
}

UTEST(Meta, tuple_get_by_type_const) {
    const tuple<I, J> t(I(1), J(2));
    EXPECT_EQ(tuple_get_by_type<I>(t).get(), 1);
    EXPECT_EQ(tuple_get_by_type<J>(t).get(), 2);
}

// ============================================================================
// tuple_cat_t
// ============================================================================

UTEST(Meta, tuple_cat_both_non_empty) {
    using T1 = tuple<int, float>;
    using T2 = tuple<double, long>;
    using Result = tuple_cat_t<T1, T2>;
    using Expected = tuple<int, float, double, long>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

UTEST(Meta, tuple_cat_first_empty) {
    using Result = tuple_cat_t<tuple<>, tuple<int, float>>;
    EXPECT_TRUE((is_same<Result, tuple<int, float>>::value));
}

UTEST(Meta, tuple_cat_second_empty) {
    using Result = tuple_cat_t<tuple<int, float>, tuple<>>;
    EXPECT_TRUE((is_same<Result, tuple<int, float>>::value));
}

UTEST(Meta, tuple_cat_both_empty) {
    using Result = tuple_cat_t<tuple<>, tuple<>>;
    EXPECT_TRUE((is_same<Result, tuple<>>::value));
}

// ============================================================================
// tuple_filter_out_t
// ============================================================================

UTEST(Meta, tuple_filter_out_remove_one) {
    using Source = tuple<int, float, double>;
    using Filter = tuple<float>;
    using Result = tuple_filter_out_t<Source, Filter>;
    using Expected = tuple<int, double>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

UTEST(Meta, tuple_filter_out_remove_multiple) {
    using Source = tuple<int, float, double, long>;
    using Filter = tuple<float, long>;
    using Result = tuple_filter_out_t<Source, Filter>;
    using Expected = tuple<int, double>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

UTEST(Meta, tuple_filter_out_remove_none) {
    using Source = tuple<int, float, double>;
    using Filter = tuple<long>;
    using Result = tuple_filter_out_t<Source, Filter>;
    EXPECT_TRUE((is_same<Result, Source>::value));
}

UTEST(Meta, tuple_filter_out_remove_all) {
    using Source = tuple<int, float>;
    using Filter = tuple<int, float>;
    using Result = tuple_filter_out_t<Source, Filter>;
    EXPECT_TRUE((is_same<Result, tuple<>>::value));
}

UTEST(Meta, tuple_filter_out_empty_source) {
    using Result = tuple_filter_out_t<tuple<>, tuple<int, float>>;
    EXPECT_TRUE((is_same<Result, tuple<>>::value));
}

UTEST(Meta, tuple_filter_out_empty_filter) {
    using Source = tuple<int, float>;
    using Result = tuple_filter_out_t<Source, tuple<>>;
    EXPECT_TRUE((is_same<Result, Source>::value));
}

// ============================================================================
// index_sequence / make_index_sequence
// ============================================================================

UTEST(Meta, make_index_sequence_zero) {
    using Result = make_index_sequence<0>;
    EXPECT_TRUE((is_same<Result, index_sequence<>>::value));
}

UTEST(Meta, make_index_sequence_one) {
    using Result = make_index_sequence<1>;
    EXPECT_TRUE((is_same<Result, index_sequence<0>>::value));
}

UTEST(Meta, make_index_sequence_three) {
    using Result = make_index_sequence<3>;
    EXPECT_TRUE((is_same<Result, index_sequence<0, 1, 2>>::value));
}

UTEST(Meta, make_index_sequence_five) {
    using Result = make_index_sequence<5>;
    EXPECT_TRUE((is_same<Result, index_sequence<0, 1, 2, 3, 4>>::value));
}

// ============================================================================
// T::dim_type
// ============================================================================

UTEST(Meta, dim_type_plain_dim) {
    using Result = I::dim_type;
    EXPECT_TRUE((is_same<Result, I>::value));
}

UTEST(Meta, dim_type_fold) {
    using FoldI8 = Fold<I, 8>;
    using Result = FoldI8::dim_type;
    EXPECT_TRUE((is_same<Result, I>::value));
}

UTEST(Meta, dim_type_different_folds_same_base) {
    using FoldI4 = Fold<I, 4>;
    using FoldI8 = Fold<I, 8>;
    using Result4 = FoldI4::dim_type;
    using Result8 = FoldI8::dim_type;
    EXPECT_TRUE((is_same<Result4, I>::value));
    EXPECT_TRUE((is_same<Result8, I>::value));
    EXPECT_TRUE((is_same<Result4, Result8>::value));
}

// ============================================================================
// dim_type::stride
// ============================================================================

UTEST(Meta, dim_stride_plain_dim) {
    EXPECT_EQ(I::stride, 1);
}

UTEST(Meta, dim_stride_fold_8) {
    using FoldI8 = Fold<I, 8>;
    EXPECT_EQ(FoldI8::stride, 8);
}

UTEST(Meta, dim_stride_fold_4) {
    using FoldJ4 = Fold<J, 4>;
    EXPECT_EQ(FoldJ4::stride, 4);
}

// ============================================================================
// tuple_contains_base_dim
// ============================================================================

UTEST(Meta, tuple_contains_base_dim_plain_match) {
    EXPECT_TRUE((tuple_contains_base_dim<I, tuple<I, J, K>>::value));
}

UTEST(Meta, tuple_contains_base_dim_fold_match) {
    using FoldI8 = Fold<I, 8>;
    EXPECT_TRUE((tuple_contains_base_dim<I, tuple<FoldI8, J>>::value));
}

UTEST(Meta, tuple_contains_base_dim_no_match) {
    EXPECT_FALSE((tuple_contains_base_dim<K, tuple<I, J>>::value));
}

UTEST(Meta, tuple_contains_base_dim_empty_tuple) {
    EXPECT_FALSE((tuple_contains_base_dim<I, tuple<>>::value));
}

UTEST(Meta, tuple_contains_base_dim_mixed) {
    using FoldI8 = Fold<I, 8>;
    using FoldJ4 = Fold<J, 4>;
    EXPECT_TRUE((tuple_contains_base_dim<I, tuple<FoldI8, FoldJ4, K>>::value));
    EXPECT_TRUE((tuple_contains_base_dim<J, tuple<FoldI8, FoldJ4, K>>::value));
    EXPECT_TRUE((tuple_contains_base_dim<K, tuple<FoldI8, FoldJ4, K>>::value));
}

// ============================================================================
// tuple_unique_base_dims_t
// ============================================================================

UTEST(Meta, tuple_unique_base_dims_plain) {
    using Result = tuple_unique_base_dims_t<tuple<I, J, K>>;
    EXPECT_TRUE((is_same<Result, tuple<I, J, K>>::value));
}

UTEST(Meta, tuple_unique_base_dims_duplicates) {
    using Result = tuple_unique_base_dims_t<tuple<I, I, J>>;
    EXPECT_TRUE((is_same<Result, tuple<I, J>>::value));
}

UTEST(Meta, tuple_unique_base_dims_folds) {
    using FoldI8 = Fold<I, 8>;
    using FoldI4 = Fold<I, 4>;
    using Result = tuple_unique_base_dims_t<tuple<FoldI8, FoldI4, J>>;
    // Both folds have base I, so only I appears once
    EXPECT_TRUE((is_same<Result, tuple<I, J>>::value));
}

UTEST(Meta, tuple_unique_base_dims_mixed) {
    using FoldI8 = Fold<I, 8>;
    using Result = tuple_unique_base_dims_t<tuple<I, FoldI8, J>>;
    // I and FoldI8 both have base I
    EXPECT_TRUE((is_same<Result, tuple<I, J>>::value));
}

UTEST(Meta, tuple_unique_base_dims_empty) {
    using Result = tuple_unique_base_dims_t<tuple<>>;
    EXPECT_TRUE((is_same<Result, tuple<>>::value));
}

// ============================================================================
// tuple_get_by_base_dim
// ============================================================================

UTEST(Meta, tuple_get_by_base_dim_plain) {
    tuple<I, J, K> t(I(1), J(2), K(3));
    using TupleType = tuple<I, J, K>;
    auto& result = tuple_get_by_base_dim<I, TupleType>::get(t);
    EXPECT_EQ(result, I(1));
}

UTEST(Meta, tuple_get_by_base_dim_fold) {
    using FoldI8 = Fold<I, 8>;
    using TupleType = tuple<FoldI8, J>;
    tuple<FoldI8, J> t(FoldI8(5), J(2));
    auto& result = tuple_get_by_base_dim<I, TupleType>::get(t);
    EXPECT_EQ(result, FoldI8(5));
}

UTEST(Meta, tuple_get_by_base_dim_mutable) {
    using TupleType = tuple<I, J>;
    tuple<I, J> t(I(1), J(2));
    tuple_get_by_base_dim<I, TupleType>::get(t) = I(100);
    EXPECT_EQ((tuple_get_by_base_dim<I, TupleType>::get(t)), I(100));
}

UTEST(Meta, tuple_get_by_base_dim_const) {
    using TupleType = tuple<I, J>;
    const tuple<I, J> t(I(1), J(2));
    const auto& result = tuple_get_by_base_dim<J, TupleType>::get(t);
    EXPECT_EQ(result, J(2));
}

UTEST(Meta, tuple_get_by_base_dim_returns_first_match) {
    using FoldI8 = Fold<I, 8>;
    using FoldI2 = Fold<I, 2>;
    using TupleType = tuple<FoldI8, J, FoldI2>;
    tuple<FoldI8, J, FoldI2> t(FoldI8(10), J(20), FoldI2(30));

    // Should return FoldI8, not FoldI2
    auto& result = tuple_get_by_base_dim<I, TupleType>::get(t);
    EXPECT_EQ(result, FoldI8(10));

    // Verify it's actually the FoldI8 type
    EXPECT_TRUE((is_same<decay_t<decltype(result)>, FoldI8>::value));
}

// ============================================================================
// tuple_exclude_base_dims_t
// ============================================================================

UTEST(Meta, tuple_exclude_base_dims_simple) {
    using Result = tuple_exclude_base_dims_t<tuple<I, J, K>, tuple<J>>;
    EXPECT_TRUE((is_same<Result, tuple<I, K>>::value));
}

UTEST(Meta, tuple_exclude_base_dims_with_folds) {
    using FoldI8 = Fold<I, 8>;
    using Result = tuple_exclude_base_dims_t<tuple<FoldI8, J, K>, tuple<I>>;
    // FoldI8 has base I, so it gets excluded
    EXPECT_TRUE((is_same<Result, tuple<J, K>>::value));
}

UTEST(Meta, tuple_exclude_base_dims_none_excluded) {
    using Result = tuple_exclude_base_dims_t<tuple<I, J>, tuple<K>>;
    EXPECT_TRUE((is_same<Result, tuple<I, J>>::value));
}

UTEST(Meta, tuple_exclude_base_dims_all_excluded) {
    using Result = tuple_exclude_base_dims_t<tuple<I, J>, tuple<I, J>>;
    EXPECT_TRUE((is_same<Result, tuple<>>::value));
}

UTEST(Meta, tuple_exclude_base_dims_empty_source) {
    using Result = tuple_exclude_base_dims_t<tuple<>, tuple<I, J>>;
    EXPECT_TRUE((is_same<Result, tuple<>>::value));
}

UTEST(Meta, tuple_exclude_base_dims_empty_filter) {
    using Result = tuple_exclude_base_dims_t<tuple<I, J>, tuple<>>;
    EXPECT_TRUE((is_same<Result, tuple<I, J>>::value));
}

UTEST(Meta, tuple_exclude_base_dims_multiple_folds_same_base) {
    using FoldI8 = Fold<I, 8>;
    using FoldI2 = Fold<I, 2>;
    using FoldJ4 = Fold<J, 4>;
    using Result = tuple_exclude_base_dims_t<tuple<FoldI8, FoldJ4, FoldI2, K>, tuple<I>>;
    // Both FoldI8 and FoldI2 have base I, so both get excluded
    EXPECT_TRUE((is_same<Result, tuple<FoldJ4, K>>::value));
}

// ============================================================================
// tuple_keep_base_dims_t
// ============================================================================

UTEST(Meta, tuple_keep_base_dims_simple) {
    using Result = tuple_keep_base_dims_t<tuple<I, J, K>, tuple<J>>;
    EXPECT_TRUE((is_same<Result, tuple<J>>::value));
}

UTEST(Meta, tuple_keep_base_dims_with_folds) {
    using FoldI8 = Fold<I, 8>;
    using Result = tuple_keep_base_dims_t<tuple<FoldI8, J, K>, tuple<I>>;
    // FoldI8 has base I, so it gets kept
    EXPECT_TRUE((is_same<Result, tuple<FoldI8>>::value));
}

UTEST(Meta, tuple_keep_base_dims_multiple) {
    using Result = tuple_keep_base_dims_t<tuple<I, J, K>, tuple<I, K>>;
    EXPECT_TRUE((is_same<Result, tuple<I, K>>::value));
}

UTEST(Meta, utest_tuple_keep_base_dims_multiple_folds_same_bas) {
    using FoldI8 = Fold<I, 8>;
    using FoldI2 = Fold<I, 2>;
    using FoldJ4 = Fold<J, 4>;
    using Result = tuple_keep_base_dims_t<tuple<FoldI8, FoldJ4, FoldI2, K>, tuple<I, J>>;
    EXPECT_TRUE((is_same<Result, tuple<FoldI8, FoldJ4, FoldI2>>::value));
}

UTEST(Meta, tuple_keep_base_dims_none_kept) {
    using Result = tuple_keep_base_dims_t<tuple<I, J>, tuple<K>>;
    EXPECT_TRUE((is_same<Result, tuple<>>::value));
}

UTEST(Meta, tuple_keep_base_dims_all_kept) {
    using Result = tuple_keep_base_dims_t<tuple<I, J>, tuple<I, J, K>>;
    EXPECT_TRUE((is_same<Result, tuple<I, J>>::value));
}

UTEST(Meta, tuple_keep_base_dims_empty_source) {
    using Result = tuple_keep_base_dims_t<tuple<>, tuple<I, J>>;
    EXPECT_TRUE((is_same<Result, tuple<>>::value));
}

UTEST(Meta, tuple_keep_base_dims_empty_filter) {
    using Result = tuple_keep_base_dims_t<tuple<I, J>, tuple<>>;
    EXPECT_TRUE((is_same<Result, tuple<>>::value));
}

// ============================================================================
// enable_if / enable_if_t (compile-time tests)
// ============================================================================

template <typename T> enable_if_t<is_same<T, int>::value, bool> test_enable_if_helper() {
    return true;
}

template <typename T> enable_if_t<!is_same<T, int>::value, bool> test_enable_if_helper() {
    return false;
}

UTEST(Meta, enable_if_true_branch) {
    EXPECT_TRUE(test_enable_if_helper<int>());
}

UTEST(Meta, enable_if_false_branch) {
    EXPECT_FALSE(test_enable_if_helper<float>());
}
