/**
 * @file backend.hpp (onemkl)
 * @brief Declares the concrete oneMKL backend implementation class.
 * @details This class inherits from DefaultBackend and overrides functions
 * where oneMKL (DFTI) provides optimized implementations (FFT plans).
 * All other functionality is inherited from DefaultBackend.
 */

#ifndef OMNIDSP_ONEMKL_BACKEND_HPP
#define OMNIDSP_ONEMKL_BACKEND_HPP

#include <memory>  // For std::unique_ptr
#include <span>    // For std::span
#include <vector>  // For vector types

#include "../default/backend.hpp"  // Inherit from DefaultBackend
// Include necessary types referenced in method signatures
#include <OmniDSP/convolution.hpp>  // Included via default/backend.hpp
#include <OmniDSP/core_types.hpp>
#include <OmniDSP/cqt.hpp>  // Included via default/backend.hpp
#include <OmniDSP/fft.hpp>
#include <OmniDSP/filter.hpp>    // Included via default/backend.hpp
#include <OmniDSP/resample.hpp>  // Included via default/backend.hpp
#include <OmniDSP/window.hpp>    // Included via default/backend.hpp

// Include the headers declaring the oneMKL Plan implementations
#include "fft.hpp"

namespace OmniDSP::onemkl {

  //--------------------------------------------------------------------------
  // oneMKL Main Backend Implementation Class
  //--------------------------------------------------------------------------

  /** @brief Concrete oneMKL backend implementation. Inherits from
   * DefaultBackend. */
  class OneMKLBackend final : public default ::DefaultBackend {
   public:
    // --- Constructor / Destructor ---
    OneMKLBackend();
    ~OneMKLBackend() override;

    // --- Core ---
    // MUST override to identify this backend
    BackendType get_backend() const override;

    // --- Override functions optimized ONLY by oneMKL DFTI ---
    // --- All other functions (windows, other plans, filter design) are
    // inherited ---

    // Plan Factories (Override ONLY FFT/RFFT)
    [[nodiscard]] OmniExpected<std::unique_ptr<FFTPlan<C32>>>
    create_fft_plan_c32(size_t length) const override;  // Keep override
    [[nodiscard]] OmniExpected<std::unique_ptr<FFTPlan<C64>>>
    create_fft_plan_c64(size_t length) const override;  // Keep override
    [[nodiscard]] OmniExpected<std::unique_ptr<RFFTPlan<F32>>>
    create_rfft_plan_f32(size_t length) const override;  // Keep override
    [[nodiscard]] OmniExpected<std::unique_ptr<RFFTPlan<F64>>>
    create_rfft_plan_f64(size_t length) const override;  // Keep override

    // *** REMOVED ALL OTHER FUNCTION DECLARATIONS (Windows, Resample, Conv,
    // Corr, Filter Plans) *** They are automatically inherited from
    // DefaultBackend.

   private:
    // Any private members specific to the oneMKL backend's state (if any)
  };

  // Factory function declaration remains the same
  // Note: The name create_backend() might be ambiguous if you have multiple
  // backend factories declared in the same scope. Consider renaming to
  // create_onemkl_backend() here and ensure the declaration in
  // interface/backend.hpp matches. For now, keeping it as create_backend()
  // assuming it's declared correctly elsewhere.
  std::unique_ptr<Abstract::AbstractBackend> create_backend();

}  // namespace OmniDSP::onemkl

#endif  // OMNIDSP_ONEMKL_BACKEND_HPP
