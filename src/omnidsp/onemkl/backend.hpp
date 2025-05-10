/**
 * @file backend.hpp (OneMKL)
 * @brief Declares the concrete oneMKL backend implementation class.
 * @details This class inherits from Default::Backend and overrides functions
 * where oneMKL (DFTI) provides optimized implementations (FFT plan
 * implementations). All other functionality is inherited from Default::Backend.
 */

#ifndef OMNIDSP_ONEMKL_BACKEND_HPP
#define OMNIDSP_ONEMKL_BACKEND_HPP

#include <memory>  // For std::unique_ptr

// Inherit from Default::Backend to get default implementations for
// non-overridden functions.
#include "default/backend.hpp"

// Include necessary types referenced in method signatures from the public API
#include <OmniDSP/core_types.hpp>  // For Status, OmniExpected, F32, C32, C64 etc.
// Public Plan headers are not strictly needed for declaring _impl overrides,
// but Abstract Plan Impl types are (usually via default/backend.hpp ->
// interface/backend.hpp). #include <OmniDSP/fft.hpp>

// Include the headers declaring the oneMKL Plan *Implementations*
// These define classes like OneMKL::FFTPlanImpl<T>
#include "fft.hpp"  // Defines OneMKL::FFTPlanImpl, OneMKL::RFFTPlanImpl

namespace OmniDSP::OneMKL {

  /** @brief Concrete oneMKL backend implementation. Inherits from
   * Default::Backend. */
  class Backend final : public Default::Backend {
   public:
    // --- Constructor / Destructor ---
    Backend();
    ~Backend() override;

    // --- Core ---
    BackendType get_backend() const override;

    // --- Override Plan Impl Factories for oneMKL DFTI ---
    // These methods override the corresponding virtual methods from
    // Abstract::Backend (which Default::Backend also overrides) to provide
    // oneMKL-specific implementations.
    [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C32>>>
    create_fft_plan_impl_c32(size_t length) const override;
    [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C64>>>
    create_fft_plan_impl_c64(size_t length) const override;

    [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F32>>>
    create_rfft_plan_impl_f32(size_t length) const override;
    [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F64>>>
    create_rfft_plan_impl_f64(size_t length) const override;

    // All other methods (one-off operations, window generation, other plan impl
    // factories, filter design) are inherited from Default::Backend.

   private:
    // Any private members specific to the oneMKL backend's state (if any).
  };

  // Factory function declaration is in "omnidsp/interface/backend.hpp"
  // std::unique_ptr<Abstract::Backend> create_onemkl_backend(); // Definition
  // in .cpp

}  // namespace OmniDSP::OneMKL

#endif  // OMNIDSP_ONEMKL_BACKEND_HPP
