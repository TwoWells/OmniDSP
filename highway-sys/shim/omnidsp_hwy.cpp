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

HWY_BEFORE_NAMESPACE();
namespace omnidsp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// --- Element-wise multiply (f32) ---
void MulF32(const float* a, const float* b,
            float* out, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
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
void MulF64(const double* a, const double* b,
            double* out, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
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
void AddF32(const float* a, const float* b,
            float* out, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
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
void AddF64(const double* a, const double* b,
            double* out, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
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
void ScaleF32(float* data, float scalar, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto vs = hn::Set(d, scalar);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto v = hn::LoadU(d, data + i);
        hn::StoreU(hn::Mul(v, vs), d, data + i);
    }
    for (; i < count; ++i) {
        data[i] *= scalar;
    }
}

// --- Scale / broadcast multiply (f64) ---
void ScaleF64(double* data, double scalar, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    const auto vs = hn::Set(d, scalar);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto v = hn::LoadU(d, data + i);
        hn::StoreU(hn::Mul(v, vs), d, data + i);
    }
    for (; i < count; ++i) {
        data[i] *= scalar;
    }
}

// --- Dot product (f32) ---
float DotF32(const float* a, const float* b, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    auto sum = hn::Zero(d);
    size_t i = 0;
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
double DotF64(const double* a, const double* b, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    auto sum = hn::Zero(d);
    size_t i = 0;
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
void CmulF32(const float* a, const float* b,
             float* out, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const size_t fcount = count * 2;
    size_t i = 0;
    for (; i + N <= fcount; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        hn::StoreU(hn::MulComplex(va, vb), d, out + i);
    }
    // Scalar remainder — process complex pairs.
    for (; i < fcount; i += 2) {
        const float are = a[i], aim = a[i + 1];
        const float bre = b[i], bim = b[i + 1];
        out[i]     = are * bre - aim * bim;
        out[i + 1] = are * bim + aim * bre;
    }
}

// --- Complex multiply (f64, interleaved re/im layout) ---
void CmulF64(const double* a, const double* b,
             double* out, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    const size_t fcount = count * 2;
    size_t i = 0;
    for (; i + N <= fcount; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        hn::StoreU(hn::MulComplex(va, vb), d, out + i);
    }
    // Scalar remainder — process complex pairs.
    for (; i < fcount; i += 2) {
        const double are = a[i], aim = a[i + 1];
        const double bre = b[i], bim = b[i + 1];
        out[i]     = are * bre - aim * bim;
        out[i + 1] = are * bim + aim * bre;
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

}  // extern "C"
#endif  // HWY_ONCE
