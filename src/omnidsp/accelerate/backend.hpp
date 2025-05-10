/**
 * @file backend.hpp (Accelerate)
 * @brief Declares the concrete Accelerate backend implementation class.
 * @details This class inherits from Default::Backend and overrides functions
 * where the Accelerate framework provides optimized FFT plan implementations.
 * All other functionality is inherited from Default::Backend.
 */

#ifndef OMNIDSP_ACCELERATE_BACKEND_HPP
#define OMNIDSP_ACCELERATE_BACKEND_HPP

#include <memory>  // For std::unique_ptr

// Inherit from Default::Backend
#include "default/backend.hpp"

// Include necessary types referenced in method signatures
#include <OmniDSP/core_types.hpp>  // For Status, OmniExpected, F32, C32, C64 etc.
// Public Plan headers are not strictly needed for declaring _impl overrides
// #include <OmniDSP/fft.hpp>

// Include the header declaring the Accelerate FFT plan *Implementations*
// These define classes like Accelerate::FFTPlanImpl<T>
#include "fft.hpp"  // Assumes this file defines Accelerate::FFTPlanImpl, Accelerate::RFFTPlanImpl

namespace OmniDSP::Accelerate {

  /**
   * @brief Concrete Accelerate backend implementation. Inherits from
   * Default::Backend. Specializes only FFT plan implementations.
   */
  class Backend final : public Default::Backend {
   public:
    // --- Constructor / Destructor ---
    Backend();
    ~Backend() override;

    // --- Core ---
    BackendType get_backend() const override;

    // --- Override Plan Impl Factories for Accelerate vDSP (FFT Only) ---
    // These methods override the corresponding virtual methods from
    // Abstract::Backend (which Default::Backend also overrides) to provide
    // Accelerate-specific implementations.
    [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C32>>>
    create_fft_plan_impl_c32(size_t length) const override;
    [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C64>>>
    create_fft_plan_impl_c64(size_t length) const override;

    [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F32>>>
    create_rfft_plan_impl_f32(size_t length) const override;
    [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F64>>>
    create_rfft_plan_impl_f64(size_t length) const override;

    // All other methods are inherited from Default::Backend.

   private:
    // Any private members specific to the Accelerate backend's state (if any).
  };

  // Factory function declaration is in "omnidsp/interface/backend.hpp"
  // std::unique_ptr<Abstract::Backend> create_accelerate_backend(); //
  // Definition in .cpp

}  // namespace OmniDSP::Accelerate

#endif  // OMNIDSP_ACCELERATE_BACKEND_HPP
