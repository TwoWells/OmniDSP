// SPDX-License-Identifier: MIT OR Apache-2.0
// OmniDSP Highway shim — extern "C" wrappers over Highway SIMD kernels.

#include <stddef.h>

// Disable scalar target — we have RustVecOps for that.
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS HWY_SCALAR
#endif

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "omnidsp_hwy.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

// Force inlining of all callees (Highway ops) before the loop unroller runs.
// Apple clang's unroller bails when it sees inline candidates in the loop body
// (LLVM issue #88141). flatten ensures LoadU/Mul/StoreU are inlined first.
#if HWY_COMPILER_CLANG || HWY_COMPILER_GCC
#define OMNIDSP_FLATTEN __attribute__((flatten))
#else
#define OMNIDSP_FLATTEN
#endif

// Unroll factor: ARM A64 NEON holds 2 f64 lanes, so unroll 8 = 16 elements
// per iteration to match Apple Silicon's wide execution units and expose
// more ILP.  x86 AVX2 has 4 f64 lanes; unroll 4 = 16 elements is sufficient.
#if HWY_ARCH_ARM_A64
#define OMNIDSP_UNROLL HWY_UNROLL(8)
#else
#define OMNIDSP_UNROLL HWY_UNROLL(4)
#endif

HWY_BEFORE_NAMESPACE();
namespace omnidsp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// ---- Shared inline helpers (ADR-003: platform specialization) ----

// Portable complex multiply on deinterleaved re/im vectors.
// Math is identical on all targets — no platform conditionals.
template <class V>
HWY_INLINE void CmulVec(V a_re, V a_im, V b_re, V b_im,
                         V& out_re, V& out_im) {
    out_re = hn::NegMulAdd(a_im, b_im, hn::Mul(a_re, b_re));
    out_im = hn::MulAdd(a_im, b_re, hn::Mul(a_re, b_im));
}

// Load interleaved complex data into deinterleaved re/im vectors.
// Loads N complex elements (2*N scalar values) from ptr.
// ARM A64: LoadU + InterleaveEven/Odd (ldp + trn, proven faster).
// Default: Highway's portable LoadInterleaved2.
template <class D>
HWY_INLINE void DeinterleaveLoad(D d, const hn::TFromD<D>* ptr,
                                  hn::Vec<D>& re, hn::Vec<D>& im) {
#if HWY_ARCH_ARM_A64
    const size_t N = hn::Lanes(d);
    const auto v0 = hn::LoadU(d, ptr);
    const auto v1 = hn::LoadU(d, ptr + N);
    re = hn::InterleaveEven(v0, v1);
    im = hn::InterleaveOdd(d, v0, v1);
#else
    hn::LoadInterleaved2(d, ptr, re, im);
#endif
}

// Store deinterleaved re/im vectors as interleaved complex data.
// Stores N complex elements (2*N scalar values) to ptr.
// ARM A64: InterleaveEven/Odd + StoreU (proven faster).
// Default: Highway's portable StoreInterleaved2.
template <class D>
HWY_INLINE void InterleaveStore(hn::Vec<D> re, hn::Vec<D> im, D d,
                                hn::TFromD<D>* ptr) {
#if HWY_ARCH_ARM_A64
    const size_t N = hn::Lanes(d);
    hn::StoreU(hn::InterleaveEven(re, im), d, ptr);
    hn::StoreU(hn::InterleaveOdd(d, re, im), d, ptr + N);
#else
    hn::StoreInterleaved2(re, im, d, ptr);
#endif
}

// --- Element-wise multiply (f32) ---
// HWY_RESTRICT on inputs: safe because we load before store each iteration,
// even when out aliases an input (mul_inplace).
OMNIDSP_FLATTEN void MulF32(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
            float* out, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    OMNIDSP_UNROLL
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        hn::StoreU(hn::Mul(va, vb), d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = a[i] * b[i];
    }
}

// --- Element-wise multiply (f64) ---
OMNIDSP_FLATTEN void MulF64(const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
            double* out, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    OMNIDSP_UNROLL
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        hn::StoreU(hn::Mul(va, vb), d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = a[i] * b[i];
    }
}

// --- Element-wise add (f32) ---
OMNIDSP_FLATTEN void AddF32(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
            float* out, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    OMNIDSP_UNROLL
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        hn::StoreU(hn::Add(va, vb), d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = a[i] + b[i];
    }
}

// --- Element-wise add (f64) ---
OMNIDSP_FLATTEN void AddF64(const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
            double* out, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    OMNIDSP_UNROLL
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        hn::StoreU(hn::Add(va, vb), d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = a[i] + b[i];
    }
}

// --- Scale / broadcast multiply (f32) ---
OMNIDSP_FLATTEN void ScaleF32(float* data, float scalar, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto vs = hn::Set(d, scalar);
    size_t i = 0;
    OMNIDSP_UNROLL
    for (; i + N <= count; i += N) {
        const auto v = hn::LoadU(d, data + i);
        hn::StoreU(hn::Mul(v, vs), d, data + i);
    }
    for (; i < count; ++i) {
        data[i] *= scalar;
    }
}

// --- Scale / broadcast multiply (f64) ---
OMNIDSP_FLATTEN void ScaleF64(double* data, double scalar, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    const auto vs = hn::Set(d, scalar);
    size_t i = 0;
    OMNIDSP_UNROLL
    for (; i + N <= count; i += N) {
        const auto v = hn::LoadU(d, data + i);
        hn::StoreU(hn::Mul(v, vs), d, data + i);
    }
    for (; i < count; ++i) {
        data[i] *= scalar;
    }
}

// --- Dot product (f32) ---
OMNIDSP_FLATTEN float DotF32(const float* a, const float* b, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    auto sum = hn::Zero(d);
    size_t i = 0;
    OMNIDSP_UNROLL
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        sum = hn::MulAdd(va, vb, sum);
    }
    float total = hn::ReduceSum(d, sum);
    for (; i < count; ++i) {
        total += a[i] * b[i];
    }
    return total;
}

// --- Dot product (f64) ---
OMNIDSP_FLATTEN double DotF64(const double* a, const double* b, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    auto sum = hn::Zero(d);
    size_t i = 0;
    OMNIDSP_UNROLL
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        sum = hn::MulAdd(va, vb, sum);
    }
    double total = hn::ReduceSum(d, sum);
    for (; i < count; ++i) {
        total += a[i] * b[i];
    }
    return total;
}

// --- Complex multiply (f32, interleaved re/im layout) ---
// count = number of complex elements; raw floats = count * 2.
OMNIDSP_FLATTEN void CmulF32(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
             float* out, size_t count) {
    const hn::ScalableTag<float> d;
    using V = hn::Vec<decltype(d)>;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    OMNIDSP_UNROLL
    for (; i + N <= count; i += N) {
        V a_re, a_im, b_re, b_im, out_re, out_im;
        DeinterleaveLoad(d, a + i * 2, a_re, a_im);
        DeinterleaveLoad(d, b + i * 2, b_re, b_im);
        CmulVec(a_re, a_im, b_re, b_im, out_re, out_im);
        InterleaveStore(out_re, out_im, d, out + i * 2);
    }
    for (; i < count; ++i) {
        const size_t j = i * 2;
        const float are = a[j], aim = a[j + 1];
        const float bre = b[j], bim = b[j + 1];
        out[j]     = are * bre - aim * bim;
        out[j + 1] = are * bim + aim * bre;
    }
}

// --- Complex multiply (f64, interleaved re/im layout) ---
OMNIDSP_FLATTEN void CmulF64(const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
             double* out, size_t count) {
    const hn::ScalableTag<double> d;
    using V = hn::Vec<decltype(d)>;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    OMNIDSP_UNROLL
    for (; i + N <= count; i += N) {
        V a_re, a_im, b_re, b_im, out_re, out_im;
        DeinterleaveLoad(d, a + i * 2, a_re, a_im);
        DeinterleaveLoad(d, b + i * 2, b_re, b_im);
        CmulVec(a_re, a_im, b_re, b_im, out_re, out_im);
        InterleaveStore(out_re, out_im, d, out + i * 2);
    }
    for (; i < count; ++i) {
        const size_t j = i * 2;
        const double are = a[j], aim = a[j + 1];
        const double bre = b[j], bim = b[j + 1];
        out[j]     = are * bre - aim * bim;
        out[j + 1] = are * bim + aim * bre;
    }
}

// --- No-alias element-wise multiply (f32) ---
// Caller guarantees a, b, and out do not alias each other.
// HWY_RESTRICT on all three pointers gives the compiler maximum freedom.
OMNIDSP_FLATTEN void MulF32NoAlias(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
            float* HWY_RESTRICT out, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    OMNIDSP_UNROLL
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        hn::StoreU(hn::Mul(va, vb), d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = a[i] * b[i];
    }
}

// --- No-alias element-wise multiply (f64) ---
OMNIDSP_FLATTEN void MulF64NoAlias(const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
            double* HWY_RESTRICT out, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    OMNIDSP_UNROLL
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        hn::StoreU(hn::Mul(va, vb), d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = a[i] * b[i];
    }
}

// --- No-alias complex multiply (f32, interleaved re/im layout) ---
OMNIDSP_FLATTEN void CmulF32NoAlias(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
             float* HWY_RESTRICT out, size_t count) {
    const hn::ScalableTag<float> d;
    using V = hn::Vec<decltype(d)>;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    OMNIDSP_UNROLL
    for (; i + N <= count; i += N) {
        V a_re, a_im, b_re, b_im, out_re, out_im;
        DeinterleaveLoad(d, a + i * 2, a_re, a_im);
        DeinterleaveLoad(d, b + i * 2, b_re, b_im);
        CmulVec(a_re, a_im, b_re, b_im, out_re, out_im);
        InterleaveStore(out_re, out_im, d, out + i * 2);
    }
    for (; i < count; ++i) {
        const size_t j = i * 2;
        const float are = a[j], aim = a[j + 1];
        const float bre = b[j], bim = b[j + 1];
        out[j]     = are * bre - aim * bim;
        out[j + 1] = are * bim + aim * bre;
    }
}

// --- No-alias complex multiply (f64, interleaved re/im layout) ---
OMNIDSP_FLATTEN void CmulF64NoAlias(const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
             double* HWY_RESTRICT out, size_t count) {
    const hn::ScalableTag<double> d;
    using V = hn::Vec<decltype(d)>;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    OMNIDSP_UNROLL
    for (; i + N <= count; i += N) {
        V a_re, a_im, b_re, b_im, out_re, out_im;
        DeinterleaveLoad(d, a + i * 2, a_re, a_im);
        DeinterleaveLoad(d, b + i * 2, b_re, b_im);
        CmulVec(a_re, a_im, b_re, b_im, out_re, out_im);
        InterleaveStore(out_re, out_im, d, out + i * 2);
    }
    for (; i < count; ++i) {
        const size_t j = i * 2;
        const double are = a[j], aim = a[j + 1];
        const double bre = b[j], bim = b[j + 1];
        out[j]     = are * bre - aim * bim;
        out[j + 1] = are * bim + aim * bre;
    }
}

// --- Butterfly stage (f32) ---
// Processes one complete stage of a radix-2 Cooley-Tukey FFT.
// `data` is interleaved [re, im] complex, length = n * 2 floats.
// `twiddles` is interleaved [re, im], length = half_len * 2 floats.
// `half_len` is the butterfly half-size for this stage (1, 2, 4, ...).
// `n` is the FFT length (number of complex elements).
OMNIDSP_FLATTEN void ButterflyStageF32(float* data, const float* twiddles,
                       size_t half_len, size_t n) {
    const hn::ScalableTag<float> d;
    using V = hn::Vec<decltype(d)>;
    const size_t N = hn::Lanes(d);
    const size_t full_len = half_len * 2;
    const size_t half_floats = half_len * 2;

    for (size_t group_start = 0; group_start < n; group_start += full_len) {
        float* even_ptr = data + group_start * 2;
        float* odd_ptr  = even_ptr + half_floats;

        size_t i = 0;
        OMNIDSP_UNROLL
        for (; i + N <= half_len; i += N) {
            V tw_re, tw_im, odd_re, odd_im, t_re, t_im, ev_re, ev_im;
            DeinterleaveLoad(d, twiddles + i * 2, tw_re, tw_im);
            DeinterleaveLoad(d, odd_ptr + i * 2, odd_re, odd_im);
            CmulVec(tw_re, tw_im, odd_re, odd_im, t_re, t_im);
            DeinterleaveLoad(d, even_ptr + i * 2, ev_re, ev_im);
            InterleaveStore(hn::Add(ev_re, t_re), hn::Add(ev_im, t_im),
                            d, even_ptr + i * 2);
            InterleaveStore(hn::Sub(ev_re, t_re), hn::Sub(ev_im, t_im),
                            d, odd_ptr + i * 2);
        }
        for (; i < half_len; ++i) {
            const size_t j = i * 2;
            const float tw_re = twiddles[j], tw_im = twiddles[j + 1];
            const float odd_re = odd_ptr[j], odd_im = odd_ptr[j + 1];
            const float t_re = tw_re * odd_re - tw_im * odd_im;
            const float t_im = tw_re * odd_im + tw_im * odd_re;
            const float e_re = even_ptr[j], e_im = even_ptr[j + 1];
            even_ptr[j]     = e_re + t_re;
            even_ptr[j + 1] = e_im + t_im;
            odd_ptr[j]      = e_re - t_re;
            odd_ptr[j + 1]  = e_im - t_im;
        }
    }
}

// --- Butterfly stage (f64) ---
OMNIDSP_FLATTEN void ButterflyStageF64(double* data, const double* twiddles,
                       size_t half_len, size_t n) {
    const hn::ScalableTag<double> d;
    using V = hn::Vec<decltype(d)>;
    const size_t N = hn::Lanes(d);
    const size_t full_len = half_len * 2;
    const size_t half_floats = half_len * 2;

    for (size_t group_start = 0; group_start < n; group_start += full_len) {
        double* even_ptr = data + group_start * 2;
        double* odd_ptr  = even_ptr + half_floats;

        size_t i = 0;
        OMNIDSP_UNROLL
        for (; i + N <= half_len; i += N) {
            V tw_re, tw_im, odd_re, odd_im, t_re, t_im, ev_re, ev_im;
            DeinterleaveLoad(d, twiddles + i * 2, tw_re, tw_im);
            DeinterleaveLoad(d, odd_ptr + i * 2, odd_re, odd_im);
            CmulVec(tw_re, tw_im, odd_re, odd_im, t_re, t_im);
            DeinterleaveLoad(d, even_ptr + i * 2, ev_re, ev_im);
            InterleaveStore(hn::Add(ev_re, t_re), hn::Add(ev_im, t_im),
                            d, even_ptr + i * 2);
            InterleaveStore(hn::Sub(ev_re, t_re), hn::Sub(ev_im, t_im),
                            d, odd_ptr + i * 2);
        }
        for (; i < half_len; ++i) {
            const size_t j = i * 2;
            const double tw_re = twiddles[j], tw_im = twiddles[j + 1];
            const double odd_re = odd_ptr[j], odd_im = odd_ptr[j + 1];
            const double t_re = tw_re * odd_re - tw_im * odd_im;
            const double t_im = tw_re * odd_im + tw_im * odd_re;
            const double e_re = even_ptr[j], e_im = even_ptr[j + 1];
            even_ptr[j]     = e_re + t_re;
            even_ptr[j + 1] = e_im + t_im;
            odd_ptr[j]      = e_re - t_re;
            odd_ptr[j + 1]  = e_im - t_im;
        }
    }
}

// --- Radix-4 butterfly stage (f32) ---
// Fuses two consecutive radix-2 stages into a single pass over the data.
// `twiddles` layout: [W1[0..q], W2[0..q], W3[0..q]] where q = quarter_len.
//   W1[k] applied to sub-block 2, W2[k] to sub-block 1, W3[k] to sub-block 3.
// `forward`: 1 for forward FFT (-j rotation), 0 for inverse (+j rotation).
OMNIDSP_FLATTEN void ButterflyStage4F32(float* data, const float* twiddles,
                        size_t quarter_len, size_t n, int forward) {
    const hn::ScalableTag<float> d;
    using V = hn::Vec<decltype(d)>;
    const size_t N = hn::Lanes(d);
    const size_t full_len = quarter_len * 4;
    const size_t q_floats = quarter_len * 2;

    const float* tw1 = twiddles;
    const float* tw2 = twiddles + q_floats;
    const float* tw3 = twiddles + q_floats * 2;

    // j rotation: forward = -j (re=+q_im, im=-q_re),
    //             inverse = +j (re=-q_im, im=+q_re).
    const auto j_re_sign = hn::Set(d, forward ? 1.0f : -1.0f);
    const auto j_im_sign = hn::Neg(j_re_sign);

    for (size_t group_start = 0; group_start < n; group_start += full_len) {
        float* p0 = data + group_start * 2;
        float* p1 = p0 + q_floats;
        float* p2 = p1 + q_floats;
        float* p3 = p2 + q_floats;

        size_t i = 0;
        OMNIDSP_UNROLL
        for (; i + N <= quarter_len; i += N) {
            V x0_re, x0_im, x1_re, x1_im, x2_re, x2_im, x3_re, x3_im;
            DeinterleaveLoad(d, p0 + i * 2, x0_re, x0_im);
            DeinterleaveLoad(d, p1 + i * 2, x1_re, x1_im);
            DeinterleaveLoad(d, p2 + i * 2, x2_re, x2_im);
            DeinterleaveLoad(d, p3 + i * 2, x3_re, x3_im);

            V tw_re, tw_im;
            V tx2_re, tx2_im, tx1_re, tx1_im, tx3_re, tx3_im;
            DeinterleaveLoad(d, tw1 + i * 2, tw_re, tw_im);
            CmulVec(tw_re, tw_im, x2_re, x2_im, tx2_re, tx2_im);
            DeinterleaveLoad(d, tw2 + i * 2, tw_re, tw_im);
            CmulVec(tw_re, tw_im, x1_re, x1_im, tx1_re, tx1_im);
            DeinterleaveLoad(d, tw3 + i * 2, tw_re, tw_im);
            CmulVec(tw_re, tw_im, x3_re, x3_im, tx3_re, tx3_im);

            const auto u_re = hn::Add(x0_re, tx1_re);
            const auto u_im = hn::Add(x0_im, tx1_im);
            const auto v_re = hn::Sub(x0_re, tx1_re);
            const auto v_im = hn::Sub(x0_im, tx1_im);
            const auto p_re = hn::Add(tx2_re, tx3_re);
            const auto p_im = hn::Add(tx2_im, tx3_im);
            const auto q_re = hn::Sub(tx2_re, tx3_re);
            const auto q_im = hn::Sub(tx2_im, tx3_im);

            const auto jq_re = hn::Mul(q_im, j_re_sign);
            const auto jq_im = hn::Mul(q_re, j_im_sign);

            InterleaveStore(hn::Add(u_re, p_re), hn::Add(u_im, p_im),
                            d, p0 + i * 2);
            InterleaveStore(hn::Add(v_re, jq_re), hn::Add(v_im, jq_im),
                            d, p1 + i * 2);
            InterleaveStore(hn::Sub(u_re, p_re), hn::Sub(u_im, p_im),
                            d, p2 + i * 2);
            InterleaveStore(hn::Sub(v_re, jq_re), hn::Sub(v_im, jq_im),
                            d, p3 + i * 2);
        }
        for (; i < quarter_len; ++i) {
            const size_t j = i * 2;
            const float x0r = p0[j], x0i = p0[j + 1];
            const float x1r = p1[j], x1i = p1[j + 1];
            const float x2r = p2[j], x2i = p2[j + 1];
            const float x3r = p3[j], x3i = p3[j + 1];

            const float t1r = tw1[j], t1i = tw1[j + 1];
            const float tx2r = t1r * x2r - t1i * x2i;
            const float tx2i = t1r * x2i + t1i * x2r;

            const float t2r = tw2[j], t2i = tw2[j + 1];
            const float tx1r = t2r * x1r - t2i * x1i;
            const float tx1i = t2r * x1i + t2i * x1r;

            const float t3r = tw3[j], t3i = tw3[j + 1];
            const float tx3r = t3r * x3r - t3i * x3i;
            const float tx3i = t3r * x3i + t3i * x3r;

            const float ur = x0r + tx1r, ui = x0i + tx1i;
            const float vr = x0r - tx1r, vi = x0i - tx1i;
            const float pr = tx2r + tx3r, pi = tx2i + tx3i;
            const float qr = tx2r - tx3r, qi = tx2i - tx3i;

            const float jqr = forward ? qi : -qi;
            const float jqi = forward ? -qr : qr;

            p0[j] = ur + pr; p0[j + 1] = ui + pi;
            p1[j] = vr + jqr; p1[j + 1] = vi + jqi;
            p2[j] = ur - pr; p2[j + 1] = ui - pi;
            p3[j] = vr - jqr; p3[j + 1] = vi - jqi;
        }
    }
}

// --- Radix-4 butterfly stage (f64) ---
OMNIDSP_FLATTEN void ButterflyStage4F64(double* data, const double* twiddles,
                        size_t quarter_len, size_t n, int forward) {
    const hn::ScalableTag<double> d;
    using V = hn::Vec<decltype(d)>;
    const size_t N = hn::Lanes(d);
    const size_t full_len = quarter_len * 4;
    const size_t q_floats = quarter_len * 2;

    const double* tw1 = twiddles;
    const double* tw2 = twiddles + q_floats;
    const double* tw3 = twiddles + q_floats * 2;

    const auto j_re_sign = hn::Set(d, forward ? 1.0 : -1.0);
    const auto j_im_sign = hn::Neg(j_re_sign);

    for (size_t group_start = 0; group_start < n; group_start += full_len) {
        double* p0 = data + group_start * 2;
        double* p1 = p0 + q_floats;
        double* p2 = p1 + q_floats;
        double* p3 = p2 + q_floats;

        size_t i = 0;
        OMNIDSP_UNROLL
        for (; i + N <= quarter_len; i += N) {
            V x0_re, x0_im, x1_re, x1_im, x2_re, x2_im, x3_re, x3_im;
            DeinterleaveLoad(d, p0 + i * 2, x0_re, x0_im);
            DeinterleaveLoad(d, p1 + i * 2, x1_re, x1_im);
            DeinterleaveLoad(d, p2 + i * 2, x2_re, x2_im);
            DeinterleaveLoad(d, p3 + i * 2, x3_re, x3_im);

            V tw_re, tw_im;
            V tx2_re, tx2_im, tx1_re, tx1_im, tx3_re, tx3_im;
            DeinterleaveLoad(d, tw1 + i * 2, tw_re, tw_im);
            CmulVec(tw_re, tw_im, x2_re, x2_im, tx2_re, tx2_im);
            DeinterleaveLoad(d, tw2 + i * 2, tw_re, tw_im);
            CmulVec(tw_re, tw_im, x1_re, x1_im, tx1_re, tx1_im);
            DeinterleaveLoad(d, tw3 + i * 2, tw_re, tw_im);
            CmulVec(tw_re, tw_im, x3_re, x3_im, tx3_re, tx3_im);

            const auto u_re = hn::Add(x0_re, tx1_re);
            const auto u_im = hn::Add(x0_im, tx1_im);
            const auto v_re = hn::Sub(x0_re, tx1_re);
            const auto v_im = hn::Sub(x0_im, tx1_im);
            const auto p_re = hn::Add(tx2_re, tx3_re);
            const auto p_im = hn::Add(tx2_im, tx3_im);
            const auto q_re = hn::Sub(tx2_re, tx3_re);
            const auto q_im = hn::Sub(tx2_im, tx3_im);

            const auto jq_re = hn::Mul(q_im, j_re_sign);
            const auto jq_im = hn::Mul(q_re, j_im_sign);

            InterleaveStore(hn::Add(u_re, p_re), hn::Add(u_im, p_im),
                            d, p0 + i * 2);
            InterleaveStore(hn::Add(v_re, jq_re), hn::Add(v_im, jq_im),
                            d, p1 + i * 2);
            InterleaveStore(hn::Sub(u_re, p_re), hn::Sub(u_im, p_im),
                            d, p2 + i * 2);
            InterleaveStore(hn::Sub(v_re, jq_re), hn::Sub(v_im, jq_im),
                            d, p3 + i * 2);
        }
        for (; i < quarter_len; ++i) {
            const size_t j = i * 2;
            const double x0r = p0[j], x0i = p0[j + 1];
            const double x1r = p1[j], x1i = p1[j + 1];
            const double x2r = p2[j], x2i = p2[j + 1];
            const double x3r = p3[j], x3i = p3[j + 1];

            const double t1r = tw1[j], t1i = tw1[j + 1];
            const double tx2r = t1r * x2r - t1i * x2i;
            const double tx2i = t1r * x2i + t1i * x2r;

            const double t2r = tw2[j], t2i = tw2[j + 1];
            const double tx1r = t2r * x1r - t2i * x1i;
            const double tx1i = t2r * x1i + t2i * x1r;

            const double t3r = tw3[j], t3i = tw3[j + 1];
            const double tx3r = t3r * x3r - t3i * x3i;
            const double tx3i = t3r * x3i + t3i * x3r;

            const double ur = x0r + tx1r, ui = x0i + tx1i;
            const double vr = x0r - tx1r, vi = x0i - tx1i;
            const double pr = tx2r + tx3r, pi = tx2i + tx3i;
            const double qr = tx2r - tx3r, qi = tx2i - tx3i;

            const double jqr = forward ? qi : -qi;
            const double jqi = forward ? -qr : qr;

            p0[j] = ur + pr; p0[j + 1] = ui + pi;
            p1[j] = vr + jqr; p1[j + 1] = vi + jqi;
            p2[j] = ur - pr; p2[j + 1] = ui - pi;
            p3[j] = vr - jqr; p3[j + 1] = vi - jqi;
        }
    }
}

// --- First-stage radix-4 butterfly (f32) ---
// Specialized for quarter_len=1 where all twiddles are trivial (1+0j).
// Eliminates 3 complex multiplies per group — the main bottleneck for
// small and medium FFT sizes (see ticket 07).
//
// Scalar loop: each group is 4 complex elements with trivial twiddles,
// so the butterfly is just adds/subs + j-rotation.  Callgrind shows this
// beats an in-register SIMD approach (DupEven/InterleaveLower/OddEven/
// IfThenElse) by ~2x on SSE2 N=4 because the shuffle overhead per group
// outweighs the arithmetic savings when processing only 1 group/iter.
OMNIDSP_FLATTEN void ButterflyStage4FirstF32(float* data, size_t n, int forward) {
    const float j_re = forward ? 1.0f : -1.0f;
    const float j_im = -j_re;

    for (size_t g = 0; g < n; g += 4) {
        float* p = data + g * 2;
        const float x0r = p[0], x0i = p[1];
        const float x1r = p[2], x1i = p[3];
        const float x2r = p[4], x2i = p[5];
        const float x3r = p[6], x3i = p[7];

        const float ur = x0r + x1r, ui = x0i + x1i;
        const float vr = x0r - x1r, vi = x0i - x1i;
        const float pr = x2r + x3r, pi = x2i + x3i;
        const float qr = x2r - x3r, qi = x2i - x3i;

        const float jqr = j_re * qi;
        const float jqi = j_im * qr;

        p[0] = ur + pr; p[1] = ui + pi;
        p[2] = vr + jqr; p[3] = vi + jqi;
        p[4] = ur - pr; p[5] = ui - pi;
        p[6] = vr - jqr; p[7] = vi - jqi;
    }
}

// --- First-stage radix-4 butterfly (f64) ---
OMNIDSP_FLATTEN void ButterflyStage4FirstF64(double* data, size_t n, int forward) {
    const double j_re = forward ? 1.0 : -1.0;
    const double j_im = -j_re;

    for (size_t g = 0; g < n; g += 4) {
        double* p = data + g * 2;
        const double x0r = p[0], x0i = p[1];
        const double x1r = p[2], x1i = p[3];
        const double x2r = p[4], x2i = p[5];
        const double x3r = p[6], x3i = p[7];

        const double ur = x0r + x1r, ui = x0i + x1i;
        const double vr = x0r - x1r, vi = x0i - x1i;
        const double pr = x2r + x3r, pi = x2i + x3i;
        const double qr = x2r - x3r, qi = x2i - x3i;

        const double jqr = j_re * qi;
        const double jqi = j_im * qr;

        p[0] = ur + pr; p[1] = ui + pi;
        p[2] = vr + jqr; p[3] = vi + jqi;
        p[4] = ur - pr; p[5] = ui - pi;
        p[6] = vr - jqr; p[7] = vi - jqi;
    }
}

// --- Fully unrolled 4-point butterfly (f32) ---
// Complete 4-point DFT on bit-reversed input.  All twiddles are trivial
// (1+0j): the butterfly is purely adds/subs + j-rotation.  Equivalent to
// ButterflyStage4First with n=4 but without loop overhead.
// data: 4 complex elements (8 floats), bit-reversed.
// forward: 1 for forward FFT (-j rotation), 0 for inverse (+j rotation).
OMNIDSP_FLATTEN void Butterfly4F32(float* data, int forward) {
    const float x0r = data[0], x0i = data[1];
    const float x1r = data[2], x1i = data[3];
    const float x2r = data[4], x2i = data[5];
    const float x3r = data[6], x3i = data[7];

    const float ur = x0r + x1r, ui = x0i + x1i;
    const float vr = x0r - x1r, vi = x0i - x1i;
    const float pr = x2r + x3r, pi = x2i + x3i;
    const float qr = x2r - x3r, qi = x2i - x3i;

    const float jqr = forward ? qi : -qi;
    const float jqi = forward ? -qr : qr;

    data[0] = ur + pr; data[1] = ui + pi;
    data[2] = vr + jqr; data[3] = vi + jqi;
    data[4] = ur - pr; data[5] = ui - pi;
    data[6] = vr - jqr; data[7] = vi - jqi;
}

// --- Fully unrolled 4-point butterfly (f64) ---
OMNIDSP_FLATTEN void Butterfly4F64(double* data, int forward) {
    const double x0r = data[0], x0i = data[1];
    const double x1r = data[2], x1i = data[3];
    const double x2r = data[4], x2i = data[5];
    const double x3r = data[6], x3i = data[7];

    const double ur = x0r + x1r, ui = x0i + x1i;
    const double vr = x0r - x1r, vi = x0i - x1i;
    const double pr = x2r + x3r, pi = x2i + x3i;
    const double qr = x2r - x3r, qi = x2i - x3i;

    const double jqr = forward ? qi : -qi;
    const double jqi = forward ? -qr : qr;

    data[0] = ur + pr; data[1] = ui + pi;
    data[2] = vr + jqr; data[3] = vi + jqi;
    data[4] = ur - pr; data[5] = ui - pi;
    data[6] = vr - jqr; data[7] = vi - jqi;
}

// --- Fully unrolled 8-point butterfly (f32) ---
// Complete 8-point DFT on bit-reversed input.  Fuses radix-2 stage 0
// with radix-4 stage pair (1,2).  Hardcodes all twiddle values:
//   W1[1] = exp(∓πi/4), W2[1] = exp(∓πi/2), W3[1] = exp(∓3πi/4)
// eliminating twiddle loads and 3 of 6 complex multiplies (the k=0
// twiddles are all 1+0j).  The remaining 3 multiplies reduce to
// add/sub/scale with the known twiddle structure.
// data: 8 complex elements (16 floats), bit-reversed.
// forward: 1 for forward FFT (-j rotation), 0 for inverse (+j rotation).
OMNIDSP_FLATTEN void Butterfly8F32(float* data, int forward) {
    const float S = 0.70710678118654752f;  // 1/sqrt(2)

    // Load 8 complex elements (bit-reversed order).
    const float x0r = data[0],  x0i = data[1];
    const float x1r = data[2],  x1i = data[3];
    const float x2r = data[4],  x2i = data[5];
    const float x3r = data[6],  x3i = data[7];
    const float x4r = data[8],  x4i = data[9];
    const float x5r = data[10], x5i = data[11];
    const float x6r = data[12], x6i = data[13];
    const float x7r = data[14], x7i = data[15];

    // Stage 0: radix-2, half_len=1, all twiddles = 1+0j.
    const float a0r = x0r + x1r, a0i = x0i + x1i;
    const float a1r = x0r - x1r, a1i = x0i - x1i;
    const float a2r = x2r + x3r, a2i = x2i + x3i;
    const float a3r = x2r - x3r, a3i = x2i - x3i;
    const float a4r = x4r + x5r, a4i = x4i + x5i;
    const float a5r = x4r - x5r, a5i = x4i - x5i;
    const float a6r = x6r + x7r, a6i = x6i + x7i;
    const float a7r = x6r - x7r, a7i = x6i - x7i;

    // Stage pair (1,2): radix-4, quarter_len=2.
    // Sub-blocks: p0=[a0,a1], p1=[a2,a3], p2=[a4,a5], p3=[a6,a7].
    // W1 applied to p2, W2 to p1, W3 to p3.
    //
    // k=0: all twiddles = 1+0j, no multiply needed.
    // k=1: hardcoded twiddle multiplies —
    //   Forward: W1[1]=(S,-S), W2[1]=(0,-1), W3[1]=(-S,-S)
    //   Inverse: W1[1]=(S,+S), W2[1]=(0,+1), W3[1]=(-S,+S)

    // Twiddle-multiply k=1 elements.
    float tx5r, tx5i, tx3r, tx3i, tx7r, tx7i;
    if (forward) {
        // W1[1]=(S,-S) * a5: (S*(a5r+a5i), S*(a5i-a5r))
        tx5r = S * (a5r + a5i);  tx5i = S * (a5i - a5r);
        // W2[1]=(0,-1) * a3: (a3i, -a3r)
        tx3r = a3i;              tx3i = -a3r;
        // W3[1]=(-S,-S) * a7: (S*(a7i-a7r), -S*(a7r+a7i))
        tx7r = S * (a7i - a7r);  tx7i = -S * (a7r + a7i);
    } else {
        // W1[1]=(S,S) * a5: (S*(a5r-a5i), S*(a5r+a5i))
        tx5r = S * (a5r - a5i);  tx5i = S * (a5r + a5i);
        // W2[1]=(0,1) * a3: (-a3i, a3r)
        tx3r = -a3i;             tx3i = a3r;
        // W3[1]=(-S,S) * a7: (-S*(a7r+a7i), S*(a7r-a7i))
        tx7r = -S * (a7r + a7i); tx7i = S * (a7r - a7i);
    }

    // j-rotation sign: forward = -j, inverse = +j.
    const float jr = forward ? 1.0f : -1.0f;
    const float ji = -jr;

    // k=0 radix-4 butterfly: x0=a0, tx1=a2, tx2=a4, tx3=a6 (trivial).
    const float u0r = a0r + a2r, u0i = a0i + a2i;
    const float v0r = a0r - a2r, v0i = a0i - a2i;
    const float p0r = a4r + a6r, p0i = a4i + a6i;
    const float q0r = a4r - a6r, q0i = a4i - a6i;
    const float jq0r = jr * q0i, jq0i = ji * q0r;

    data[0]  = u0r + p0r;  data[1]  = u0i + p0i;   // y0[0]
    data[4]  = v0r + jq0r; data[5]  = v0i + jq0i;  // y1[0]
    data[8]  = u0r - p0r;  data[9]  = u0i - p0i;   // y2[0]
    data[12] = v0r - jq0r; data[13] = v0i - jq0i;  // y3[0]

    // k=1 radix-4 butterfly: x0=a1, tx1=W2*a3, tx2=W1*a5, tx3=W3*a7.
    const float u1r = a1r + tx3r, u1i = a1i + tx3i;
    const float v1r = a1r - tx3r, v1i = a1i - tx3i;
    const float p1r = tx5r + tx7r, p1i = tx5i + tx7i;
    const float q1r = tx5r - tx7r, q1i = tx5i - tx7i;
    const float jq1r = jr * q1i, jq1i = ji * q1r;

    data[2]  = u1r + p1r;  data[3]  = u1i + p1i;   // y0[1]
    data[6]  = v1r + jq1r; data[7]  = v1i + jq1i;  // y1[1]
    data[10] = u1r - p1r;  data[11] = u1i - p1i;   // y2[1]
    data[14] = v1r - jq1r; data[15] = v1i - jq1i;  // y3[1]
}

// --- Fully unrolled 8-point butterfly (f64) ---
OMNIDSP_FLATTEN void Butterfly8F64(double* data, int forward) {
    const double S = 0.70710678118654752440;  // 1/sqrt(2)

    const double x0r = data[0],  x0i = data[1];
    const double x1r = data[2],  x1i = data[3];
    const double x2r = data[4],  x2i = data[5];
    const double x3r = data[6],  x3i = data[7];
    const double x4r = data[8],  x4i = data[9];
    const double x5r = data[10], x5i = data[11];
    const double x6r = data[12], x6i = data[13];
    const double x7r = data[14], x7i = data[15];

    const double a0r = x0r + x1r, a0i = x0i + x1i;
    const double a1r = x0r - x1r, a1i = x0i - x1i;
    const double a2r = x2r + x3r, a2i = x2i + x3i;
    const double a3r = x2r - x3r, a3i = x2i - x3i;
    const double a4r = x4r + x5r, a4i = x4i + x5i;
    const double a5r = x4r - x5r, a5i = x4i - x5i;
    const double a6r = x6r + x7r, a6i = x6i + x7i;
    const double a7r = x6r - x7r, a7i = x6i - x7i;

    double tx5r, tx5i, tx3r, tx3i, tx7r, tx7i;
    if (forward) {
        tx5r = S * (a5r + a5i);  tx5i = S * (a5i - a5r);
        tx3r = a3i;              tx3i = -a3r;
        tx7r = S * (a7i - a7r);  tx7i = -S * (a7r + a7i);
    } else {
        tx5r = S * (a5r - a5i);  tx5i = S * (a5r + a5i);
        tx3r = -a3i;             tx3i = a3r;
        tx7r = -S * (a7r + a7i); tx7i = S * (a7r - a7i);
    }

    const double jr = forward ? 1.0 : -1.0;
    const double ji = -jr;

    const double u0r = a0r + a2r, u0i = a0i + a2i;
    const double v0r = a0r - a2r, v0i = a0i - a2i;
    const double p0r = a4r + a6r, p0i = a4i + a6i;
    const double q0r = a4r - a6r, q0i = a4i - a6i;
    const double jq0r = jr * q0i, jq0i = ji * q0r;

    data[0]  = u0r + p0r;  data[1]  = u0i + p0i;
    data[4]  = v0r + jq0r; data[5]  = v0i + jq0i;
    data[8]  = u0r - p0r;  data[9]  = u0i - p0i;
    data[12] = v0r - jq0r; data[13] = v0i - jq0i;

    const double u1r = a1r + tx3r, u1i = a1i + tx3i;
    const double v1r = a1r - tx3r, v1i = a1i - tx3i;
    const double p1r = tx5r + tx7r, p1i = tx5i + tx7i;
    const double q1r = tx5r - tx7r, q1i = tx5i - tx7i;
    const double jq1r = jr * q1i, jq1i = ji * q1r;

    data[2]  = u1r + p1r;  data[3]  = u1i + p1i;
    data[6]  = v1r + jq1r; data[7]  = v1i + jq1i;
    data[10] = u1r - p1r;  data[11] = u1i - p1i;
    data[14] = v1r - jq1r; data[15] = v1i - jq1i;
}

// --- Batch 8-point butterfly (f32) ---
// Applies Butterfly8 to all N/8 contiguous groups.  Replaces the first
// 3 iterative stages of any FFT with N >= 16, fusing radix-2 stage 0 +
// radix-4 pair (1,2) into a single twiddle-free pass.
// data: N complex elements (2*N floats), bit-reversed.
// n: FFT length (complex elements, must be >= 16 and a multiple of 8).
OMNIDSP_FLATTEN void Butterfly8BatchF32(float* data, size_t n, int forward) {
    for (size_t g = 0; g < n; g += 8) {
        Butterfly8F32(data + g * 2, forward);
    }
}

// --- Batch 8-point butterfly (f64) ---
OMNIDSP_FLATTEN void Butterfly8BatchF64(double* data, size_t n, int forward) {
    for (size_t g = 0; g < n; g += 8) {
        Butterfly8F64(data + g * 2, forward);
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace omnidsp
HWY_AFTER_NAMESPACE();

// ---- Dispatch table + extern "C" (compiled once) ----
#if HWY_ONCE
namespace omnidsp {
HWY_EXPORT(MulF32);
HWY_EXPORT(MulF64);
HWY_EXPORT(AddF32);
HWY_EXPORT(AddF64);
HWY_EXPORT(ScaleF32);
HWY_EXPORT(ScaleF64);
HWY_EXPORT(DotF32);
HWY_EXPORT(DotF64);
HWY_EXPORT(CmulF32);
HWY_EXPORT(CmulF64);
HWY_EXPORT(ButterflyStageF32);
HWY_EXPORT(ButterflyStageF64);
HWY_EXPORT(ButterflyStage4F32);
HWY_EXPORT(ButterflyStage4F64);
HWY_EXPORT(ButterflyStage4FirstF32);
HWY_EXPORT(ButterflyStage4FirstF64);
HWY_EXPORT(Butterfly4F32);
HWY_EXPORT(Butterfly4F64);
HWY_EXPORT(Butterfly8F32);
HWY_EXPORT(Butterfly8F64);
HWY_EXPORT(Butterfly8BatchF32);
HWY_EXPORT(Butterfly8BatchF64);
HWY_EXPORT(MulF32NoAlias);
HWY_EXPORT(MulF64NoAlias);
HWY_EXPORT(CmulF32NoAlias);
HWY_EXPORT(CmulF64NoAlias);

// Dispatch wrappers — HWY_DYNAMIC_DISPATCH must be called from the same
// namespace as HWY_EXPORT.  The compiler inlines these away.
void DispatchMulF32(const float* a, const float* b, float* out, size_t count) {
    HWY_DYNAMIC_DISPATCH(MulF32)(a, b, out, count);
}
void DispatchMulF64(const double* a, const double* b, double* out, size_t count) {
    HWY_DYNAMIC_DISPATCH(MulF64)(a, b, out, count);
}
void DispatchAddF32(const float* a, const float* b, float* out, size_t count) {
    HWY_DYNAMIC_DISPATCH(AddF32)(a, b, out, count);
}
void DispatchAddF64(const double* a, const double* b, double* out, size_t count) {
    HWY_DYNAMIC_DISPATCH(AddF64)(a, b, out, count);
}
void DispatchScaleF32(float* data, float scalar, size_t count) {
    HWY_DYNAMIC_DISPATCH(ScaleF32)(data, scalar, count);
}
void DispatchScaleF64(double* data, double scalar, size_t count) {
    HWY_DYNAMIC_DISPATCH(ScaleF64)(data, scalar, count);
}
float DispatchDotF32(const float* a, const float* b, size_t count) {
    return HWY_DYNAMIC_DISPATCH(DotF32)(a, b, count);
}
double DispatchDotF64(const double* a, const double* b, size_t count) {
    return HWY_DYNAMIC_DISPATCH(DotF64)(a, b, count);
}
void DispatchCmulF32(const float* a, const float* b, float* out, size_t count) {
    HWY_DYNAMIC_DISPATCH(CmulF32)(a, b, out, count);
}
void DispatchCmulF64(const double* a, const double* b, double* out, size_t count) {
    HWY_DYNAMIC_DISPATCH(CmulF64)(a, b, out, count);
}
void DispatchButterflyStageF32(float* data, const float* twiddles, size_t half_len, size_t n) {
    HWY_DYNAMIC_DISPATCH(ButterflyStageF32)(data, twiddles, half_len, n);
}
void DispatchButterflyStageF64(double* data, const double* twiddles, size_t half_len, size_t n) {
    HWY_DYNAMIC_DISPATCH(ButterflyStageF64)(data, twiddles, half_len, n);
}
void DispatchButterflyStage4F32(float* data, const float* twiddles, size_t quarter_len, size_t n, int forward) {
    HWY_DYNAMIC_DISPATCH(ButterflyStage4F32)(data, twiddles, quarter_len, n, forward);
}
void DispatchButterflyStage4F64(double* data, const double* twiddles, size_t quarter_len, size_t n, int forward) {
    HWY_DYNAMIC_DISPATCH(ButterflyStage4F64)(data, twiddles, quarter_len, n, forward);
}
void DispatchButterflyStage4FirstF32(float* data, size_t n, int forward) {
    HWY_DYNAMIC_DISPATCH(ButterflyStage4FirstF32)(data, n, forward);
}
void DispatchButterflyStage4FirstF64(double* data, size_t n, int forward) {
    HWY_DYNAMIC_DISPATCH(ButterflyStage4FirstF64)(data, n, forward);
}
void DispatchButterfly4F32(float* data, int forward) {
    HWY_DYNAMIC_DISPATCH(Butterfly4F32)(data, forward);
}
void DispatchButterfly4F64(double* data, int forward) {
    HWY_DYNAMIC_DISPATCH(Butterfly4F64)(data, forward);
}
void DispatchButterfly8F32(float* data, int forward) {
    HWY_DYNAMIC_DISPATCH(Butterfly8F32)(data, forward);
}
void DispatchButterfly8F64(double* data, int forward) {
    HWY_DYNAMIC_DISPATCH(Butterfly8F64)(data, forward);
}
void DispatchButterfly8BatchF32(float* data, size_t n, int forward) {
    HWY_DYNAMIC_DISPATCH(Butterfly8BatchF32)(data, n, forward);
}
void DispatchButterfly8BatchF64(double* data, size_t n, int forward) {
    HWY_DYNAMIC_DISPATCH(Butterfly8BatchF64)(data, n, forward);
}
void DispatchMulF32NoAlias(const float* a, const float* b, float* out, size_t count) {
    HWY_DYNAMIC_DISPATCH(MulF32NoAlias)(a, b, out, count);
}
void DispatchMulF64NoAlias(const double* a, const double* b, double* out, size_t count) {
    HWY_DYNAMIC_DISPATCH(MulF64NoAlias)(a, b, out, count);
}
void DispatchCmulF32NoAlias(const float* a, const float* b, float* out, size_t count) {
    HWY_DYNAMIC_DISPATCH(CmulF32NoAlias)(a, b, out, count);
}
void DispatchCmulF64NoAlias(const double* a, const double* b, double* out, size_t count) {
    HWY_DYNAMIC_DISPATCH(CmulF64NoAlias)(a, b, out, count);
}
}  // namespace omnidsp

extern "C" {

void omnidsp_mul_f32(const float* a, const float* b, float* out, size_t count) {
    omnidsp::DispatchMulF32(a, b, out, count);
}
void omnidsp_mul_f64(const double* a, const double* b, double* out, size_t count) {
    omnidsp::DispatchMulF64(a, b, out, count);
}
void omnidsp_add_f32(const float* a, const float* b, float* out, size_t count) {
    omnidsp::DispatchAddF32(a, b, out, count);
}
void omnidsp_add_f64(const double* a, const double* b, double* out, size_t count) {
    omnidsp::DispatchAddF64(a, b, out, count);
}
void omnidsp_scale_f32(float* data, float scalar, size_t count) {
    omnidsp::DispatchScaleF32(data, scalar, count);
}
void omnidsp_scale_f64(double* data, double scalar, size_t count) {
    omnidsp::DispatchScaleF64(data, scalar, count);
}
float omnidsp_dot_f32(const float* a, const float* b, size_t count) {
    return omnidsp::DispatchDotF32(a, b, count);
}
double omnidsp_dot_f64(const double* a, const double* b, size_t count) {
    return omnidsp::DispatchDotF64(a, b, count);
}
void omnidsp_cmul_f32(const float* a, const float* b, float* out, size_t count) {
    omnidsp::DispatchCmulF32(a, b, out, count);
}
void omnidsp_cmul_f64(const double* a, const double* b, double* out, size_t count) {
    omnidsp::DispatchCmulF64(a, b, out, count);
}
void omnidsp_butterfly_stage_f32(float* data, const float* twiddles, size_t half_len, size_t n) {
    omnidsp::DispatchButterflyStageF32(data, twiddles, half_len, n);
}
void omnidsp_butterfly_stage_f64(double* data, const double* twiddles, size_t half_len, size_t n) {
    omnidsp::DispatchButterflyStageF64(data, twiddles, half_len, n);
}
void omnidsp_butterfly_stage4_f32(float* data, const float* twiddles, size_t quarter_len, size_t n, int forward) {
    omnidsp::DispatchButterflyStage4F32(data, twiddles, quarter_len, n, forward);
}
void omnidsp_butterfly_stage4_f64(double* data, const double* twiddles, size_t quarter_len, size_t n, int forward) {
    omnidsp::DispatchButterflyStage4F64(data, twiddles, quarter_len, n, forward);
}
void omnidsp_butterfly_stage4_first_f32(float* data, size_t n, int forward) {
    omnidsp::DispatchButterflyStage4FirstF32(data, n, forward);
}
void omnidsp_butterfly_stage4_first_f64(double* data, size_t n, int forward) {
    omnidsp::DispatchButterflyStage4FirstF64(data, n, forward);
}
void omnidsp_butterfly4_f32(float* data, int forward) {
    omnidsp::DispatchButterfly4F32(data, forward);
}
void omnidsp_butterfly4_f64(double* data, int forward) {
    omnidsp::DispatchButterfly4F64(data, forward);
}
void omnidsp_butterfly8_f32(float* data, int forward) {
    omnidsp::DispatchButterfly8F32(data, forward);
}
void omnidsp_butterfly8_f64(double* data, int forward) {
    omnidsp::DispatchButterfly8F64(data, forward);
}
void omnidsp_butterfly8_batch_f32(float* data, size_t n, int forward) {
    omnidsp::DispatchButterfly8BatchF32(data, n, forward);
}
void omnidsp_butterfly8_batch_f64(double* data, size_t n, int forward) {
    omnidsp::DispatchButterfly8BatchF64(data, n, forward);
}
void omnidsp_mul_f32_noalias(const float* a, const float* b, float* out, size_t count) {
    omnidsp::DispatchMulF32NoAlias(a, b, out, count);
}
void omnidsp_mul_f64_noalias(const double* a, const double* b, double* out, size_t count) {
    omnidsp::DispatchMulF64NoAlias(a, b, out, count);
}
void omnidsp_cmul_f32_noalias(const float* a, const float* b, float* out, size_t count) {
    omnidsp::DispatchCmulF32NoAlias(a, b, out, count);
}
void omnidsp_cmul_f64_noalias(const double* a, const double* b, double* out, size_t count) {
    omnidsp::DispatchCmulF64NoAlias(a, b, out, count);
}

}  // extern "C"
#endif  // HWY_ONCE
