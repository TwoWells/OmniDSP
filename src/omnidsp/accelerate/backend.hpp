/**
 * @file backend.hpp (Accelerate)
 * @brief Declares the concrete Accelerate backend implementation class.
 * @details This class inherits from Backend and overrides functions
 * where the Accelerate framework provides optimized FFT implementations.
 * All other functionality is inherited from Backend.
 */

#ifndef OMNIDSP_ACCELERATE_BACKEND_HPP  // Changed guard suffix to HPP
#define OMNIDSP_ACCELERATE_BACKEND_HPP

#include <memory>  // For std::unique_ptr
#include <span>    // For std::span
#include <vector>  // For vector types

// *** Inherit from Backend instead of AbstractBackend ***
#include "../default/backend.hpp"

// Include necessary types referenced in method signatures
#include <OmniDSP/core_types.hpp>
// #include <OmniDSP/convolution.hpp>
// #include <OmniDSP/cqt.hpp>
#include <OmniDSP/fft.hpp>
// #include <OmniDSP/filter.hpp>
// #include <OmniDSP/resample.hpp>
// #include <OmniDSP/window.hpp>

// Include the header declaring the Accelerate FFT plan implementations
#include "fft.hpp"  // Assumes this file exists and declares Accelerate*FFTPlanImpl

namespace OmniDSP::Accelerate {

  //--------------------------------------------------------------------------
  // Accelerate Main BackendType Implementation Class
  //--------------------------------------------------------------------------

  /**
   * @brief Concrete Accelerate backend implementation. Inherits from
   * Backend. Specializes only FFT operations.
   */
  class Backend final : public Default::Backend {  // Inherits from Backend
   public:
    // --- Constructor / Destructor ---
    Backend();
    ~Backend() override;

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
    // plans) are inherited directly from Backend and do not need
    // overrides here.

   private:
    // Any private members specific to the Accelerate backend's state (if any)
    // (Unlikely needed if only overriding plan factories)
  };

  std::unique_ptr<Abstract::Backend> create_accelerate_backend();
}  // namespace OmniDSP::Accelerate

#endif  // OMNIDSP_ACCELERATE_BACKEND_HPP
