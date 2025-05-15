/**
 * @file resample.hpp // New conceptual path: processor/resample.hpp
 * @brief Defines the interface for Resample Processor objects
 * under the OmniDSP::Processor namespace.
 */

#ifndef OMNIDSP_PROCESSOR_RESAMPLE_HPP  // Updated include guard
#define OMNIDSP_PROCESSOR_RESAMPLE_HPP

#include <cmath>  // For std::fabs, etc. if needed by output_length calculation
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>  // For std::runtime_error
#include <string>     // For error messages or future use
#include <utility>    // For std::move
#include <vector>     // For std::vector if used in future API

#include "OmniDSP/core_types.hpp"  // For Status, OmniExpected, F32, F64, Utils::IsComplex_v etc.
#include "OmniDSP/design/resample.hpp"  // For Design::Resample (parameter to create method)
#include "OmniDSP/omnidsp_export.hpp"  // For OMNIDSP_EXPORT (used for explicit instantiations)
// For Abstract::Backend and Abstract::ResampleProcessorImpl:
#include "interface/backend.hpp"

namespace OmniDSP {

  // New 'Processor' namespace
  namespace Processor {

    /**
     * @brief Processor object for resampling operations.
     * @tparam T_Data The data type, typically real (e.g., F32, F64).
     */
    template <typename T_Data>
    class Resample {
      static_assert(
          !Utils::IsComplex_v<T_Data>,
          "Processor::Resample currently supports real types (F32 or F64).");

     public:
      ~Resample();                          // Destructor
      Resample(Resample&& other) noexcept;  // Move constructor
      Resample& operator=(
          Resample&& other) noexcept;      // Move assignment operator
      Resample(const Resample&) = delete;  // Disable copy constructor
      Resample& operator=(const Resample&)
          = delete;  // Disable copy assignment operator

      /**
       * @brief Executes the resampling operation.
       * @param input A span representing the input signal.
       * @param output A span representing the output buffer. Must be large
       * enough.
       * @return OmniStatus::Success on success, or an error code.
       */
      [[nodiscard]] OmniStatus execute(
          std::span<const T_Data> input, std::span<T_Data> output);

      /**
       * @brief Resets the internal state of the resampler (e.g., clears filter
       * delay lines).
       * @return OmniStatus::Success on success, or an error code.
       */
      OmniStatus reset();

      /** @return The input sample rate in Hz. */
      double get_input_rate() const;

      /** @return The output sample rate in Hz. */
      double get_output_rate() const;

      /**
       * @brief Calculates the expected output length for a given input length.
       * @param input_length The length of the input signal.
       * @return The expected length of the output signal.
       */
      size_t get_output_length(size_t input_length) const;

      /**
       * @brief Creates a Resample Processor using the specified backend and
       * design parameters.
       * @param backend The abstract backend interface to use for creating the
       * implementation.
       * @param design The resampling design specification.
       * @return An OmniExpected containing a unique_ptr to the
       * Processor::Resample on success, or an error OmniStatus.
       */
      [[nodiscard]] static OmniExpected<
          std::unique_ptr<Resample<T_Data>>>  // Updated return type
      create(
          const Abstract::Backend&
              backend,  // Correctly using Abstract::Backend
          const Design::Resample& design)
      {
        OmniExpected<std::unique_ptr<Abstract::ResampleProcessorImpl<T_Data>>>
            pimpl_expected;

        // Type dispatch to call the correct backend implementation method
        if constexpr (std::is_same_v<T_Data, F32>) {
          pimpl_expected = backend.create_resample_processor_impl_f32(design);
        }
        else if constexpr (std::is_same_v<T_Data, F64>) {
          pimpl_expected = backend.create_resample_processor_impl_f64(design);
        }
        // Add other types if complex resampling is supported by backends
        // For now, sticking to the static_assert for real types.
        else {
          // This case should ideally be caught by the class-level
          // static_assert, but as a fallback:
          return std::unexpected(OmniStatus::UnsupportedType);
        }

        if (!pimpl_expected) {
          return std::unexpected(pimpl_expected.error());
        }

        // Call the static create_from_impl method of this class
        return create_from_impl(std::move(pimpl_expected.value()));
      }

      /**
       * @brief (Internal) Creates a public Resample Processor from a private
       * implementation.
       * @param pimpl A unique_ptr to the backend-specific abstract
       * ResampleProcessorImpl.
       * @return A unique_ptr to the Processor::Resample. Returns nullptr if
       * pimpl is null.
       */
      static std::unique_ptr<Resample<T_Data>>
      create_from_impl(  // Updated return type
          std::unique_ptr<Abstract::ResampleProcessorImpl<T_Data>>
              pimpl)  // Correctly using Abstract::ResampleProcessorImpl
      {
        if (!pimpl) {
          return nullptr;
        }
        // Use private constructor for controlled creation
        // The private constructor is defined below.
        return std::unique_ptr<Resample<T_Data>>(
            new Resample<T_Data>(std::move(pimpl)));
      }

     private:
      // Private constructor, called by create_from_impl.
      explicit Resample(
          std::unique_ptr<Abstract::ResampleProcessorImpl<T_Data>> pimpl)
          : pimpl_(std::move(pimpl))
      {
        if (!pimpl_) {
          // This check is good practice, even if create_from_impl also checks.
          // It ensures the object is always in a valid state post-construction
          // if called directly (though not intended).
          throw std::runtime_error(  // Or a more specific OmniDSP exception
              "Processor::Resample constructed with null implementation "
              "pointer.");
        }
      }
      std::unique_ptr<Abstract::ResampleProcessorImpl<T_Data>>
          pimpl_;  // Pointer to implementation
    };

  }  // namespace Processor
}  // namespace OmniDSP

#endif  // OMNIDSP_PROCESSOR_RESAMPLE_HPP
