#pragma once

/* Collection of compatibility workarounds to deal with compiler defects */

// Note: workarounds for xlC removed; use of xlC to build Arbor is deprecated.

#include <cstddef>
#include <cmath>

namespace arbsimd {
namespace compat {

// Work-around for bad vectorization of fma in gcc.
// Bug fixed in 6.4.1, 7.3.1, 8.1.1 and 9.0: refer to gcc bug #85597,
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=85597

template <typename T>
#if !defined(__clang__) && defined(__GNUC__) &&\
    ( __GNUC__<6 ||\
     (__GNUC__==6 && __GNUC_MINOR__*100 + __GNUC_PATCHLEVEL__ < 401) ||\
     (__GNUC__==7 && __GNUC_MINOR__*100 + __GNUC_PATCHLEVEL__ < 301) ||\
     (__GNUC__==8 && __GNUC_MINOR__*100 + __GNUC_PATCHLEVEL__ < 101))
__attribute((optimize("no-tree-vectorize")))
#endif
inline auto fma(T a, T b, T c) {
    return std::fma(a, b, c);
}

} // namespace compat
} // namespace arbsimd
