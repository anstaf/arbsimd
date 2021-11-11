#include <arbsimd/simd.hpp>

#include <random>
#include <type_traits>

#include "common.hpp"

namespace {

using namespace arbsimd;

// Use different distributions in `fill_random`, based on the value type in question:
//
//     * floating point type => uniform_real_distribution, default interval [-1, 1).
//     * bool                => uniform_int_distribution, default interval [0, 1).
//     * other integral type => uniform_int_distribution, default interval [L, U]
//                              such that L^2+L and U^2+U fit within the integer range.

template <typename V, typename = std::enable_if_t<std::is_floating_point<V>::value>>
std::uniform_real_distribution<V> make_udist(V lb = -1., V ub = 1.) {
    return std::uniform_real_distribution<V>(lb, ub);
}

template <typename V, typename = std::enable_if_t<std::is_integral<V>::value && !std::is_same<V, bool>::value>>
std::uniform_int_distribution<V> make_udist(
    V lb = std::numeric_limits<V>::lowest() / (2 << std::numeric_limits<V>::digits/2),
    V ub = std::numeric_limits<V>::max() >> (1+std::numeric_limits<V>::digits/2))
{
    return std::uniform_int_distribution<V>(lb, ub);
}

template <typename V, typename = std::enable_if_t<std::is_same<V, bool>::value>>
std::uniform_int_distribution<> make_udist(V lb = false, V ub = true) {
    return std::uniform_int_distribution<>(0, 1);
}

template <typename Seq, typename Rng>
void fill_random(Seq&& seq, Rng& rng) {
    using V = std::decay_t<decltype(*std::begin(seq))>;

    auto u = make_udist<V>();
    for (auto& x: seq) { x = u(rng); }
}

template <typename Seq, typename Rng, typename B1, typename B2>
void fill_random(Seq&& seq, Rng& rng, B1 lb, B2 ub) {
    using V = std::decay_t<decltype(*std::begin(seq))>;

    auto u = make_udist<V>(lb, ub);
    for (auto& x: seq) { x = u(rng); }
}

template <typename Simd, typename Rng, typename B1, typename B2, typename = std::enable_if_t<is_simd<Simd>::value>>
void fill_random(Simd& s, Rng& rng, B1 lb, B2 ub) {
    using V = typename Simd::scalar_type;
    constexpr unsigned N = Simd::width;

    V v[N];
    fill_random(v, rng, lb, ub);
    s.copy_from(v);
}

template <typename Simd, typename Rng, typename = std::enable_if_t<is_simd<Simd>::value>>
void fill_random(Simd& s, Rng& rng) {
    using V = typename Simd::scalar_type;
    constexpr unsigned N = Simd::width;

    V v[N];
    fill_random(v, rng);
    s.copy_from(v);
}

template <typename Simd>
::testing::AssertionResult simd_eq(Simd a, Simd b) {
    constexpr unsigned N = Simd::width;
    using V = typename Simd::scalar_type;

    V as[N], bs[N];
    a.copy_to(as);
    b.copy_to(bs);

    return ::testing::seq_eq(as, bs);
}

constexpr unsigned nrounds = 20u;

template <typename TypeParam>
struct simd_value : ::testing::Test {
    using simd = TypeParam;
    using scalar_type = typename TypeParam::scalar_type;

    static std::array<scalar_type, simd::width> make_random() {
        std::array<scalar_type, simd::width> res;
        fill_random(res, rng);
        return res;
    }
  private:
    static std::minstd_rand rng;
};

template <typename TypeParam>
std::minstd_rand simd_value<TypeParam>::rng{1001};


TYPED_TEST_SUITE_P(simd_value);

// Test agreement between simd::width(), simd::min_align() and corresponding type attributes.
TYPED_TEST_P(simd_value, meta) {
    using simd = TypeParam;
    using scalar = typename simd::scalar_type;

    static_assert(simd::width == arbsimd::width(simd{}));
    static_assert(simd::min_align == arbsimd::min_align(simd{}));

    static_assert(alignof(scalar) <= simd::min_align);
}

// Initialization and element access.
TYPED_TEST_P(simd_value, elements) {
    using simd = TypeParam;
    using scalar = typename simd::scalar_type;
    constexpr unsigned N = simd::width;

    std::minstd_rand rng(1001);

    // broadcast:
    simd a(2);
    for (unsigned i = 0; i<N; ++i) {
       EXPECT_EQ(2., a[i]);
    }

    // scalar assignment:
    a = 3;
    for (unsigned i = 0; i<N; ++i) {
        EXPECT_EQ(3, a[i]);
    }

    scalar /*bv[N],*/ cv[N], dv[N];
//
//    fill_random(bv, rng);
    fill_random(cv, rng);
    fill_random(dv, rng);
    auto bv = TestFixture::make_random();

    // array initialization:
    simd b(bv.data());
//    EXPECT_THAT(b, testing::ContainerEq(bv));
    EXPECT_TRUE(testing::indexed_eq_n(N, bv, b));

    // array rvalue initialization:
    auto cv_copy = cv;
    simd c(std::move(cv));
    EXPECT_TRUE(testing::indexed_eq_n(N, cv_copy, c));

    // pointer initialization:
    simd d(&dv[0]);
    EXPECT_TRUE(testing::indexed_eq_n(N, dv, d));

    // copy construction:
    simd e(d);
    EXPECT_TRUE(testing::indexed_eq_n(N, dv, e));

    // copy assignment:
    b = d;
    EXPECT_TRUE(testing::indexed_eq_n(N, dv, b));
}

TYPED_TEST_P(simd_value, element_lvalue) {
using simd = TypeParam;
constexpr unsigned N = simd::width;

simd a(3);
ASSERT_GT(N, 1u);
a[N-2] = 5;

for (unsigned i = 0; i<N; ++i) {
EXPECT_EQ(i==N-2? 5: 3, a[i]);
}
}

TYPED_TEST_P(simd_value, copy_to_from) {
    using simd = TypeParam;
    using scalar = typename simd::scalar_type;
    constexpr unsigned N = simd::width;

    std::minstd_rand rng(1010);

    scalar buf1[N], buf2[N];
    fill_random(buf1, rng);
    fill_random(buf2, rng);

    simd s;
    s.copy_from(buf1);
    s.copy_to(buf2);

    EXPECT_TRUE(testing::indexed_eq_n(N, buf1, s));
    EXPECT_TRUE(testing::seq_eq(buf1, buf2));
}

TYPED_TEST_P(simd_value, copy_to_from_masked) {
    using simd = TypeParam;
    using mask = typename simd::simd_mask;
    using scalar = typename simd::scalar_type;
    constexpr unsigned N = simd::width;

    std::minstd_rand rng(1031);

    for (unsigned i = 0; i<nrounds; ++i) {
        scalar buf1[N], buf2[N], buf3[N], buf4[N];
        fill_random(buf1, rng);
        fill_random(buf2, rng);
        fill_random(buf3, rng);
        fill_random(buf4, rng);

        bool mbuf1[N], mbuf2[N];
        fill_random(mbuf1, rng);
        fill_random(mbuf2, rng);
        mask m1(mbuf1);
        mask m2(mbuf2);

        scalar expected[N];
        for (unsigned i = 0; i<N; ++i) {
            expected[i] = mbuf1[i]? buf2[i]: buf1[i];
        }

        simd s(buf1);
        where(m1, s) = indirect(buf2, N);
        EXPECT_TRUE(testing::indexed_eq_n(N, expected, s));

        for (unsigned i = 0; i<N; ++i) {
            if (!mbuf2[i]) expected[i] = buf3[i];
        }

        indirect(buf3, N) = where(m2, s);
        EXPECT_TRUE(testing::indexed_eq_n(N, expected, buf3));

        for (unsigned i = 0; i<N; ++i) {
            expected[i] = mbuf2[i]? buf1[i]: buf4[i];
        }

        simd b(buf1);
        indirect(buf4, N) = where(m2, b);
        EXPECT_TRUE(testing::indexed_eq_n(N, expected, buf4));
    }
}

TYPED_TEST_P(simd_value, construct_masked) {
using simd = TypeParam;
using mask = typename simd::simd_mask;
using scalar = typename simd::scalar_type;
constexpr unsigned N = simd::width;

std::minstd_rand rng(1031);

for (unsigned i = 0; i<nrounds; ++i) {
scalar buf[N];
fill_random(buf, rng);

bool mbuf[N];
fill_random(mbuf, rng);

mask m(mbuf);
simd s(buf, m);

for (unsigned i = 0; i<N; ++i) {
if (!mbuf[i]) continue;
EXPECT_EQ(buf[i], s[i]);
}
}
}

TYPED_TEST_P(simd_value, arithmetic) {
using simd = TypeParam;
using scalar = typename simd::scalar_type;
constexpr unsigned N = simd::width;

std::minstd_rand rng(1002);
scalar u[N], v[N], w[N], r[N];

for (unsigned i = 0; i<nrounds; ++i) {
fill_random(u, rng);
fill_random(v, rng);
fill_random(w, rng);

scalar neg_u[N];
for (unsigned i = 0; i<N; ++i) neg_u[i] = -u[i];

scalar u_plus_v[N];
for (unsigned i = 0; i<N; ++i) u_plus_v[i] = u[i]+v[i];

scalar u_minus_v[N];
for (unsigned i = 0; i<N; ++i) u_minus_v[i] = u[i]-v[i];

scalar u_times_v[N];
for (unsigned i = 0; i<N; ++i) u_times_v[i] = u[i]*v[i];

scalar u_divide_v[N];
for (unsigned i = 0; i<N; ++i) u_divide_v[i] = u[i]/v[i];

scalar fma_u_v_w[N];
for (unsigned i = 0; i<N; ++i) fma_u_v_w[i] = compat::fma(u[i],v[i],w[i]);

simd us(u), vs(v), ws(w);

(-us).copy_to(r);
EXPECT_TRUE(testing::seq_eq(neg_u, r));

(us+vs).copy_to(r);
EXPECT_TRUE(testing::seq_eq(u_plus_v, r));

(us-vs).copy_to(r);
EXPECT_TRUE(testing::seq_eq(u_minus_v, r));

(us*vs).copy_to(r);
EXPECT_TRUE(testing::seq_eq(u_times_v, r));

(us/vs).copy_to(r);
#if defined(__INTEL_COMPILER)
// icpc will by default use an approximation for scalar division,
        // and a different one for vectorized scalar division; the latter,
        // in particular, is often out by 1 ulp for normal quotients.
        //
        // Unfortunately, we can't check at compile time the floating
        // point dodginess quotient.

        if (std::is_floating_point<scalar>::value) {
            EXPECT_TRUE(testing::seq_almost_eq<scalar>(u_divide_v, r));
        }
        else {
            EXPECT_TRUE(testing::seq_eq(u_divide_v, r));
        }
#else
EXPECT_TRUE(testing::seq_eq(u_divide_v, r));
#endif

(fma(us, vs, ws)).copy_to(r);
EXPECT_TRUE(testing::seq_eq(fma_u_v_w, r));
}
}

TYPED_TEST_P(simd_value, compound_assignment) {
using simd = TypeParam;

simd a, b, r;

std::minstd_rand rng(1003);
fill_random(a, rng);
fill_random(b, rng);

EXPECT_TRUE(simd_eq(a+b, (r = a)+=b));
EXPECT_TRUE(simd_eq(a-b, (r = a)-=b));
EXPECT_TRUE(simd_eq(a*b, (r = a)*=b));
EXPECT_TRUE(simd_eq(a/b, (r = a)/=b));
}

TYPED_TEST_P(simd_value, comparison) {
using simd = TypeParam;
using mask = typename simd::simd_mask;
constexpr unsigned N = simd::width;

std::minstd_rand rng(1004);
std::uniform_int_distribution<> sgn(-1, 1); // -1, 0 or 1.

for (unsigned i = 0; i<nrounds; ++i) {
int cmp[N];
bool test[N];
simd a, b;

fill_random(b, rng);

for (unsigned j = 0; j<N; ++j) {
cmp[j] = sgn(rng);
a[j] = b[j]+17*cmp[j];
}

mask gt = a>b;
for (unsigned j = 0; j<N; ++j) { test[j] = cmp[j]>0; }
EXPECT_TRUE(testing::indexed_eq_n(N, test, gt));

mask geq = a>=b;
for (unsigned j = 0; j<N; ++j) { test[j] = cmp[j]>=0; }
EXPECT_TRUE(testing::indexed_eq_n(N, test, geq));

mask lt = a<b;
for (unsigned j = 0; j<N; ++j) { test[j] = cmp[j]<0; }
EXPECT_TRUE(testing::indexed_eq_n(N, test, lt));

mask leq = a<=b;
for (unsigned j = 0; j<N; ++j) { test[j] = cmp[j]<=0; }
EXPECT_TRUE(testing::indexed_eq_n(N, test, leq));

mask eq = a==b;
for (unsigned j = 0; j<N; ++j) { test[j] = cmp[j]==0; }
EXPECT_TRUE(testing::indexed_eq_n(N, test, eq));

mask ne = a!=b;
for (unsigned j = 0; j<N; ++j) { test[j] = cmp[j]!=0; }
EXPECT_TRUE(testing::indexed_eq_n(N, test, ne));
}
}

TYPED_TEST_P(simd_value, mask_elements) {
using simd = TypeParam;
using mask = typename simd::simd_mask;
constexpr unsigned N = simd::width;

std::minstd_rand rng(1005);

// bool broadcast:
mask a(true);
for (unsigned i = 0; i<N; ++i) {
EXPECT_EQ(true, a[i]);
}

// scalar assignment:
mask d;
d = false;
for (unsigned i = 0; i<N; ++i) {
EXPECT_EQ(false, d[i]);
}
d = true;
for (unsigned i = 0; i<N; ++i) {
EXPECT_EQ(true, d[i]);
}

for (unsigned i = 0; i<nrounds; ++i) {
bool bv[N], cv[N], dv[N];

fill_random(bv, rng);
fill_random(cv, rng);
fill_random(dv, rng);

// array initialization:
mask b(bv);
EXPECT_TRUE(testing::indexed_eq_n(N, bv, b));

// array rvalue initialization:
auto cv_copy = cv;
mask c(std::move(cv));
EXPECT_TRUE(testing::indexed_eq_n(N, cv_copy, c));

// pointer initialization:
mask d(&dv[0]);
EXPECT_TRUE(testing::indexed_eq_n(N, dv, d));

// copy construction:
mask e(d);
EXPECT_TRUE(testing::indexed_eq_n(N, dv, e));

// copy assignment:
b = d;
EXPECT_TRUE(testing::indexed_eq_n(N, dv, b));
}
}

TYPED_TEST_P(simd_value, mask_element_lvalue) {
using simd = TypeParam;
using mask = typename simd::simd_mask;
constexpr unsigned N = simd::width;

std::minstd_rand rng(1006);

for (unsigned i = 0; i<nrounds; ++i) {
bool v[N];
fill_random(v, rng);

mask m(v);
for (unsigned j = 0; j<N; ++j) {
bool b = v[j];
m[j] = !b;
v[j] = !b;

EXPECT_EQ(m[j], !b);
EXPECT_TRUE(testing::indexed_eq_n(N, v, m));

m[j] = b;
v[j] = b;
EXPECT_EQ(m[j], b);
EXPECT_TRUE(testing::indexed_eq_n(N, v, m));
}
}
}

TYPED_TEST_P(simd_value, mask_copy_to_from) {
using simd = TypeParam;
using simd_mask = typename simd::simd_mask;
constexpr unsigned N = simd::width;

std::minstd_rand rng(1012);

for (unsigned i = 0; i<nrounds; ++i) {
bool buf1[N], buf2[N];
fill_random(buf1, rng);
fill_random(buf2, rng);

simd_mask m;
m.copy_from(buf1);
m.copy_to(buf2);

EXPECT_TRUE(testing::indexed_eq_n(N, buf1, m));
EXPECT_TRUE(testing::seq_eq(buf1, buf2));
}
}

TYPED_TEST_P(simd_value, mask_unpack) {
using simd = TypeParam;
using mask = typename simd::simd_mask;
constexpr unsigned N = simd::width;

std::minstd_rand rng(1035);
std::uniform_int_distribution<unsigned long long> U(0, (1ull<<N)-1);

for (unsigned i = 0; i<nrounds; ++i) {
unsigned long long packed = U(rng);
bool b[N];
mask::unpack(packed).copy_to(b);

for (unsigned j = 0; j<N; ++j) {
EXPECT_EQ((bool)(packed&(1ull<<j)), b[j]);
}
}
}

TYPED_TEST_P(simd_value, maths) {
// min, max, abs tests valid for both fp and int types.

using simd = TypeParam;
using scalar = typename simd::scalar_type;
constexpr unsigned N = simd::width;

std::minstd_rand rng(1013);

for (unsigned i = 0; i<nrounds; ++i) {
scalar a[N], b[N], test[N];
fill_random(a, rng);
fill_random(b, rng);

simd as(a), bs(b);

for (unsigned j = 0; j<N; ++j) { test[j] = std::abs(a[j]); }
EXPECT_TRUE(testing::indexed_eq_n(N, test, abs(as)));

for (unsigned j = 0; j<N; ++j) { test[j] = std::min(a[j], b[j]); }
EXPECT_TRUE(testing::indexed_eq_n(N, test, min(as, bs)));

for (unsigned j = 0; j<N; ++j) { test[j] = std::max(a[j], b[j]); }
EXPECT_TRUE(testing::indexed_eq_n(N, test, max(as, bs)));
}
}

TYPED_TEST_P(simd_value, reductions) {
// Only addition for now.

using simd = TypeParam;
using scalar = typename simd::scalar_type;
constexpr unsigned N = simd::width;

std::minstd_rand rng(1041);

for (unsigned i = 0; i<nrounds; ++i) {
scalar a[N], test = 0;

// To avoid discrepancies due to catastrophic cancelation,
// keep f.p. values non-negative.

if (std::is_floating_point<scalar>::value) {
fill_random(a, rng, 0, 1);
}
else {
fill_random(a, rng);
}

simd as(a);

for (unsigned j = 0; j<N; ++j) { test += a[j]; }
EXPECT_TRUE(testing::almost_eq(test, as.sum()));
}
}

TYPED_TEST_P(simd_value, simd_array_cast) {
    // Test conversion to/from array of scalar type.

    using simd = TypeParam;
    using scalar = typename simd::scalar_type;
    constexpr unsigned N = simd::width;

    std::minstd_rand rng(1032);

    for (unsigned i = 0; i<nrounds; ++i) {
        std::array<scalar, N> a;

        fill_random(a, rng);
        simd as = simd_cast<simd>(a);
        EXPECT_TRUE(testing::indexed_eq_n(N, as, a));
        EXPECT_TRUE(testing::seq_eq(a, simd_cast<std::array<scalar, N>>(as)));
    }
}

REGISTER_TYPED_TEST_SUITE_P(simd_value,
    meta,
    elements,
    element_lvalue,
    copy_to_from,
    copy_to_from_masked,
    construct_masked,
    arithmetic,
    compound_assignment,
    comparison,
    mask_elements,
    mask_element_lvalue,
    mask_copy_to_from,
    mask_unpack, maths,
    simd_array_cast,
    reductions);

using simd_test_types = ::testing::Types<
#ifdef __AVX__
    simd<int, 4, simd_abi::avx>,
    simd<double, 4, simd_abi::avx>,
#endif
#ifdef __AVX2__
    simd<int, 4, simd_abi::avx2>,
    simd<double, 4, simd_abi::avx2>,
#endif
#ifdef __AVX512F__
    simd<int, 8, simd_abi::avx512>,
    simd<double, 8, simd_abi::avx512>,
#endif
#if defined(__ARM_NEON)
    simd<int, 2, simd_abi::neon>,
    simd<double, 2, simd_abi::neon>,
#endif
    simd<int, 4, simd_abi::generic>,
    simd<double, 4, simd_abi::generic>,
    simd<float, 16, simd_abi::generic>,
    simd<int, 4, simd_abi::default_abi>,
    simd<double, 4, simd_abi::default_abi>,
    simd<int, 8, simd_abi::default_abi>,
    simd<double, 8, simd_abi::default_abi>
>;

INSTANTIATE_TYPED_TEST_SUITE_P(S, simd_value, simd_test_types);

}