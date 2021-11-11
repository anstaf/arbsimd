#pragma once

/*
 * Convenience functions, structs used across
 * more than one unit test.
 */

#include <cmath>
#include <iterator>
#include <type_traits>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

namespace testing {
// Google Test assertion-returning predicates:

// Assert two values are 'almost equal', with exact test for non-floating point types.
// (Uses internal class `FloatingPoint` from gtest.)

template <typename FPType>
::testing::AssertionResult almost_eq_(FPType a, FPType b, std::true_type) {
    using FP = testing::internal::FloatingPoint<FPType>;

    if ((std::isnan(a) && std::isnan(b)) || FP{a}.AlmostEquals(FP{b})) {
        return ::testing::AssertionSuccess();
    }

    return ::testing::AssertionFailure() << "floating point numbers " << a << " and " << b << " differ";
}

template <typename X>
::testing::AssertionResult almost_eq_(const X& a, const X& b, std::false_type) {
    if (a==b) {
        return ::testing::AssertionSuccess();
    }

    return ::testing::AssertionFailure() << "values " << a << " and " << b << " differ";
}

template <typename X>
::testing::AssertionResult almost_eq(const X& a, const X& b) {
    return almost_eq_(a, b, typename std::is_floating_point<X>::type{});
}

// Assert two sequences of floating point values are almost equal, with explicit
// specification of floating point type.

template <typename FPType, typename Seq1, typename Seq2>
::testing::AssertionResult seq_almost_eq(Seq1&& seq1, Seq2&& seq2) {
    using std::begin;
    using std::end;

    auto i1 = begin(seq1);
    auto i2 = begin(seq2);

    auto e1 = end(seq1);
    auto e2 = end(seq2);

    for (std::size_t j = 0; i1!=e1 && i2!=e2; ++i1, ++i2, ++j) {

        auto v1 = *i1;
        auto v2 = *i2;

        // Cast to FPType to avoid warnings about lowering conversion
        // if FPType has lower precision than Seq{12}::value_type.

        auto status = almost_eq((FPType)(v1), (FPType)(v2));
        if (!status) return status << " at index " << j;
    }

    if (i1!=e1 || i2!=e2) {
        return ::testing::AssertionFailure() << "sequences differ in length";
    }
    return ::testing::AssertionSuccess();
}

template <typename V>
bool generic_isnan(const V& x) { return false; }
inline bool generic_isnan(float x) { return std::isnan(x); }
inline bool generic_isnan(double x) { return std::isnan(x); }
inline bool generic_isnan(long double x) { return std::isnan(x); }

template <typename U, typename V>
bool equiv(const U& u, const V& v) {
    return u==v || (generic_isnan(u) && generic_isnan(v));
}

template <typename Seq1, typename Seq2>
::testing::AssertionResult seq_eq(Seq1&& seq1, Seq2&& seq2) {
    using std::begin;
    using std::end;

    auto i1 = begin(seq1);
    auto i2 = begin(seq2);

    auto e1 = end(seq1);
    auto e2 = end(seq2);

    for (std::size_t j = 0; i1!=e1 && i2!=e2; ++i1, ++i2, ++j) {
        auto v1 = *i1;
        auto v2 = *i2;

        if (!equiv(v1, v2)) {
            return ::testing::AssertionFailure() << "values " << v1 << " and " << v2 << " differ at index " << j;
        }
    }

    if (i1!=e1 || i2!=e2) {
        return ::testing::AssertionFailure() << "sequences differ in length";
    }
    return ::testing::AssertionSuccess();
}

// Assert elements 0..n-1 inclusive of two indexed collections are exactly equal.

template <typename Arr1, typename Arr2>
::testing::AssertionResult indexed_eq_n(int n, Arr1&& a1, Arr2&& a2) {
    for (int i = 0; i<n; ++i) {
        auto v1 = a1[i];
        auto v2 = a2[i];

        if (!equiv(v1,v2)) {
            return ::testing::AssertionFailure() << "values " << v1 << " and " << v2 << " differ at index " << i;
        }
    }

    return ::testing::AssertionSuccess();
}

// Assert elements 0..n-1 inclusive of two indexed collections are almost equal.

template <typename Arr1, typename Arr2>
::testing::AssertionResult indexed_almost_eq_n(int n, Arr1&& a1, Arr2&& a2) {
    for (int i = 0; i<n; ++i) {
        auto v1 = a1[i];
        auto v2 = a2[i];

        auto status = almost_eq(v1, v2);
        if (!status) return status << " at index " << i;
    }

    return ::testing::AssertionSuccess();
}
} // namespace testing
