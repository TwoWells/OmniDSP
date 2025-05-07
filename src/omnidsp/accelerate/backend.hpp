/**
 * @file backend.hpp (accelerate)
 * @brief Declares the concrete Accelerate backend implementation class.
 * @details This class inherits from DefaultBackend and overrides functions
 * where the Accelerate framework provides optimized FFT implementations.
 * All other functionality is inherited from DefaultBackend.
 */

#ifndef OMNIDSP_ACCELERATE_BACKEND_HPP  // Changed guard suffix to HPP
#define OMNIDSP_ACCELERATE_BACKEND_HPP

#include <memory>  // For std::unique_ptr
#include <span>    // For std::span
#include <vector>  // For vector types

#include "../default/backend.hpp"  // Inherit from DefaultBackend
// Include necessary types referenced in method signatures
#include <OmniDSP/core_types.hpp>
#include <OmniDSP/fft.hpp>  // Include public plan headers for factory return types

// Include the header declaring the Accelerate FFT plan implementations
#include "fft.hpp"  // Assumes this file exists and declares Accelerate*FFTPlanImpl

namespace OmniDSP::accelerate {

  //--------------------------------------------------------------------------
  // Accelerate Main BackendType Implementation Class
  //--------------------------------------------------------------------------

  /**
   * @brief Concrete Accelerate backend implementation. Inherits from
   * DefaultBackend. Specializes only FFT operations.
   */
  class AccelerateBackend final
      : public DefaultBackend {  // Inherits from DefaultBackend
   public:
    // --- Constructor / Destructor ---
    AccelerateBackend();
    ~AccelerateBackend() override;

    // --- Core ---
    // MUST override to identify this backend
    BackendType get_backend() const override;

    // --- Override functions optimized by Accelerate (FFT Only) ---

    // Plan Factories (MUST override to return Accelerate FFT/RFFT plans)
    [[nodiscard]] OmniExpected<std::unique_ptr<FFTPlan<C32>>>
    create_fft_plan_c32(size_t length) const override;
    [[nodiscard]] OmniExpected<std::unique_ptr<FFTPlan<C64>>>
    create_fft_plan_c64(size_t length) const override;
    [[nodiscard]] OmniExpected<std::unique_ptr<RFFTPlan<F32>>>
    create_rfft_plan_f32(size_t length) const override;
    [[nodiscard]] OmniExpected<std::unique_ptr<RFFTPlan<F64>>>
    create_rfft_plan_f64(size_t length) const override;

    // --- Inherited Methods ---
    // All other methods (one-off operations, window generation, filter design,
    // convolution plans, correlation plans, filter plans, resample plans, CQT
    // plans) are inherited directly from DefaultBackend and do not need
    // overrides here.

   private:
    // Any private members specific to the Accelerate backend's state (if any)
    // (Unlikely needed if only overriding plan factories)
  };

  std::unique_ptr<Abstract::AbstractBackend>
  Abstract::create_accelerate_backend();
}  // namespace OmniDSP::accelerate

#endif  // OMNIDSP_ACCELERATE_BACKEND_HPP
