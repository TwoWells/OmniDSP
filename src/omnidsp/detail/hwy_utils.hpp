#ifndef OMNIDSP_DETAIL_HWY_UTILS_HPP
#define OMNIDSP_DETAIL_HWY_UTILS_HPP

// Provides helper functions for Highway, potentially backporting
// features or providing common complex operations.

#include <type_traits>  // For std::enable_if_t, etc. (if needed by SFINAE)

#include "hwy/highway.h"

// *** TARGET-SPECIFIC HELPERS (INSIDE HWY_NAMESPACE) ***
// These need to be compiled per-target by Highway.
// The implementation is now directly inside the target-specific namespace,
// allowing correct use of target-specific operations via the 'hn' alias.
HWY_BEFORE_NAMESPACE();
namespace OmniDSP {
  namespace backend {
    namespace HWY_NAMESPACE {  // Target-specific namespace

      // Alias for Highway types/functions within the current target's namespace
      namespace hn = hwy::HWY_NAMESPACE;

      // Complex conjugate implementation
      // Removed SFINAE check as IsUnsigned/TFromV availability/scoping was
      // problematic.
      template <class V>
      inline V ComplexConj(V a)
      {
        // Use hn:: alias for target-specific operations
        return hn::OddEven(hn::Neg(a), a);
      }

      // Complex multiplication implementation (Matches Highway master
      // MulComplex)
      template <class V>
      inline V MulComplex(V a, V b)
      {
        // Use hn:: alias for target-specific operations
        const auto u = hn::DupEven(a);  // real(a)
        const auto v = hn::DupOdd(a);   // imag(a)
        const auto x = hn::DupEven(b);  // real(b)
        const auto y = hn::DupOdd(b);   // imag(b)
        // Real part: ux - vy
        const auto real_part = hn::Sub(hn::Mul(u, x), hn::Mul(v, y));
        // Imaginary part: uy + vx
        const auto imag_part = hn::MulAdd(u, y, hn::Mul(v, x));
        // Interleave: [real, imag, real, imag, ...]
        return hn::OddEven(imag_part, real_part);
      }

      // Complex-Real multiplication implementation: (re + i*im) * s = (re*s) +
      // i*(im*s) Assumes 'c' is Vec2 (complex) and 'r' is Vec (real, broadcast
      // scalar)
      template <class CplxV, class RealV>
      inline CplxV MulComplexReal(CplxV c, RealV r)
      {
        // Use hn:: alias for target-specific operations
        // Assuming CplxV is Vec2 { V vec[2]; }
        return CplxV{hn::Mul(c.val[0], r), hn::Mul(c.val[1], r)};
      }

    }  // namespace HWY_NAMESPACE
  }  // namespace backend
}  // namespace OmniDSP
HWY_AFTER_NAMESPACE();
// *** END OF TARGET-SPECIFIC HELPERS ***

#endif  // OMNIDSP_DETAIL_HWY_UTILS_HPP
