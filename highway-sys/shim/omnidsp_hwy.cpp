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
// Plain loads + in-register deinterleave via InterleaveEven/Odd, avoiding
// multi-cycle vld2/vst2 on ARM NEON.  On x86 both approaches compile to
// equivalent shuffle sequences.  The within-block permutation produced by
// InterleaveEven/Odd is self-canceling: the same ops reinterleave on output.
// count = number of complex elements; raw floats = count * 2.
OMNIDSP_FLATTEN void CmulF32(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
             float* out, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    OMNIDSP_UNROLL
    for (; i + N <= count; i += N) {
        const auto v0a = hn::LoadU(d, a + i * 2);
        const auto v1a = hn::LoadU(d, a + i * 2 + N);
        const auto a_re = hn::InterleaveEven(v0a, v1a);
        const auto a_im = hn::InterleaveOdd(d, v0a, v1a);
        const auto v0b = hn::LoadU(d, b + i * 2);
        const auto v1b = hn::LoadU(d, b + i * 2 + N);
        const auto b_re = hn::InterleaveEven(v0b, v1b);
        const auto b_im = hn::InterleaveOdd(d, v0b, v1b);
        const auto out_re = hn::NegMulAdd(a_im, b_im, hn::Mul(a_re, b_re));
        const auto out_im = hn::MulAdd(a_im, b_re, hn::Mul(a_re, b_im));
        hn::StoreU(hn::InterleaveEven(out_re, out_im), d, out + i * 2);
        hn::StoreU(hn::InterleaveOdd(d, out_re, out_im), d, out + i * 2 + N);
    }
    // Scalar remainder — process complex pairs.
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
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    OMNIDSP_UNROLL
    for (; i + N <= count; i += N) {
        const auto v0a = hn::LoadU(d, a + i * 2);
        const auto v1a = hn::LoadU(d, a + i * 2 + N);
        const auto a_re = hn::InterleaveEven(v0a, v1a);
        const auto a_im = hn::InterleaveOdd(d, v0a, v1a);
        const auto v0b = hn::LoadU(d, b + i * 2);
        const auto v1b = hn::LoadU(d, b + i * 2 + N);
        const auto b_re = hn::InterleaveEven(v0b, v1b);
        const auto b_im = hn::InterleaveOdd(d, v0b, v1b);
        const auto out_re = hn::NegMulAdd(a_im, b_im, hn::Mul(a_re, b_re));
        const auto out_im = hn::MulAdd(a_im, b_re, hn::Mul(a_re, b_im));
        hn::StoreU(hn::InterleaveEven(out_re, out_im), d, out + i * 2);
        hn::StoreU(hn::InterleaveOdd(d, out_re, out_im), d, out + i * 2 + N);
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
    const size_t N = hn::Lanes(d);
    const size_t full_len = half_len * 2;
    // Number of float elements per half = half_len * 2 (interleaved re/im).
    const size_t half_floats = half_len * 2;

    for (size_t group_start = 0; group_start < n; group_start += full_len) {
        float* even_ptr = data + group_start * 2;
        float* odd_ptr  = even_ptr + half_floats;

        // SIMD loop over the butterfly pairs in this group.
        size_t i = 0;
        OMNIDSP_UNROLL
        for (; i + N <= half_floats; i += N) {
            const auto tw  = hn::LoadU(d, twiddles + i);
            const auto odd = hn::LoadU(d, odd_ptr + i);
            const auto t   = hn::MulComplex(tw, odd);
            const auto ev  = hn::LoadU(d, even_ptr + i);
            hn::StoreU(hn::Add(ev, t), d, even_ptr + i);
            hn::StoreU(hn::Sub(ev, t), d, odd_ptr + i);
        }
        // Scalar tail — process remaining complex pairs.
        for (; i < half_floats; i += 2) {
            const float tw_re = twiddles[i], tw_im = twiddles[i + 1];
            const float odd_re = odd_ptr[i], odd_im = odd_ptr[i + 1];
            const float t_re = tw_re * odd_re - tw_im * odd_im;
            const float t_im = tw_re * odd_im + tw_im * odd_re;
            const float e_re = even_ptr[i], e_im = even_ptr[i + 1];
            even_ptr[i]     = e_re + t_re;
            even_ptr[i + 1] = e_im + t_im;
            odd_ptr[i]      = e_re - t_re;
            odd_ptr[i + 1]  = e_im - t_im;
        }
    }
}

// --- Butterfly stage (f64) ---
OMNIDSP_FLATTEN void ButterflyStageF64(double* data, const double* twiddles,
                       size_t half_len, size_t n) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    const size_t full_len = half_len * 2;
    const size_t half_floats = half_len * 2;

    for (size_t group_start = 0; group_start < n; group_start += full_len) {
        double* even_ptr = data + group_start * 2;
        double* odd_ptr  = even_ptr + half_floats;

        size_t i = 0;
        OMNIDSP_UNROLL
        for (; i + N <= half_floats; i += N) {
            const auto tw  = hn::LoadU(d, twiddles + i);
            const auto odd = hn::LoadU(d, odd_ptr + i);
            const auto t   = hn::MulComplex(tw, odd);
            const auto ev  = hn::LoadU(d, even_ptr + i);
            hn::StoreU(hn::Add(ev, t), d, even_ptr + i);
            hn::StoreU(hn::Sub(ev, t), d, odd_ptr + i);
        }
        for (; i < half_floats; i += 2) {
            const double tw_re = twiddles[i], tw_im = twiddles[i + 1];
            const double odd_re = odd_ptr[i], odd_im = odd_ptr[i + 1];
            const double t_re = tw_re * odd_re - tw_im * odd_im;
            const double t_im = tw_re * odd_im + tw_im * odd_re;
            const double e_re = even_ptr[i], e_im = even_ptr[i + 1];
            even_ptr[i]     = e_re + t_re;
            even_ptr[i + 1] = e_im + t_im;
            odd_ptr[i]      = e_re - t_re;
            odd_ptr[i + 1]  = e_im - t_im;
        }
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
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    OMNIDSP_UNROLL
    for (; i + N <= count; i += N) {
        const auto v0a = hn::LoadU(d, a + i * 2);
        const auto v1a = hn::LoadU(d, a + i * 2 + N);
        const auto a_re = hn::InterleaveEven(v0a, v1a);
        const auto a_im = hn::InterleaveOdd(d, v0a, v1a);
        const auto v0b = hn::LoadU(d, b + i * 2);
        const auto v1b = hn::LoadU(d, b + i * 2 + N);
        const auto b_re = hn::InterleaveEven(v0b, v1b);
        const auto b_im = hn::InterleaveOdd(d, v0b, v1b);
        const auto out_re = hn::NegMulAdd(a_im, b_im, hn::Mul(a_re, b_re));
        const auto out_im = hn::MulAdd(a_im, b_re, hn::Mul(a_re, b_im));
        hn::StoreU(hn::InterleaveEven(out_re, out_im), d, out + i * 2);
        hn::StoreU(hn::InterleaveOdd(d, out_re, out_im), d, out + i * 2 + N);
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
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    OMNIDSP_UNROLL
    for (; i + N <= count; i += N) {
        const auto v0a = hn::LoadU(d, a + i * 2);
        const auto v1a = hn::LoadU(d, a + i * 2 + N);
        const auto a_re = hn::InterleaveEven(v0a, v1a);
        const auto a_im = hn::InterleaveOdd(d, v0a, v1a);
        const auto v0b = hn::LoadU(d, b + i * 2);
        const auto v1b = hn::LoadU(d, b + i * 2 + N);
        const auto b_re = hn::InterleaveEven(v0b, v1b);
        const auto b_im = hn::InterleaveOdd(d, v0b, v1b);
        const auto out_re = hn::NegMulAdd(a_im, b_im, hn::Mul(a_re, b_re));
        const auto out_im = hn::MulAdd(a_im, b_re, hn::Mul(a_re, b_im));
        hn::StoreU(hn::InterleaveEven(out_re, out_im), d, out + i * 2);
        hn::StoreU(hn::InterleaveOdd(d, out_re, out_im), d, out + i * 2 + N);
    }
    for (; i < count; ++i) {
        const size_t j = i * 2;
        const double are = a[j], aim = a[j + 1];
        const double bre = b[j], bim = b[j + 1];
        out[j]     = are * bre - aim * bim;
        out[j + 1] = are * bim + aim * bre;
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
