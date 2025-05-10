/**
 * @file cqt.hpp
 * @brief Defines the interface for Constant-Q Transform (CQT) Plan objects.
 */

#ifndef OMNIDSP_CQT_HPP  // Changed guard name
#define OMNIDSP_CQT_HPP

#include <complex>
#include <cstddef>
#include <memory>
#include <span>
#include <utility>  // For std::move
#include <vector>

#include "OmniDSP/omnidsp_export.hpp"
#include "core_types.hpp"  // For Status, OmniExpected, Utils::IsComplex_v, Utils::GetComplexType, F32, F64
#include "window.hpp"  // For WindowSetup

// Include Abstract::Backend for the static create method
#include "../src/omnidsp/interface/backend.hpp"  // Defines Abstract::Backend

// Forward declare backend Impl class (already present)
namespace OmniDSP::Abstract {
  template <typename T>
  class CQTPlanImpl;
}  // namespace OmniDSP::Abstract

namespace OmniDSP {

  // class OmniDSP; // Forward declaration if needed by friends

  /**
   * @brief Plan object for executing Constant-Q Transforms (CQT).
   * @tparam T The underlying real floating-point type (e.g., F32, F64).
   */
  template <typename T>  // T is real type here
  class OMNIDSP_EXPORT CQTPlan {
    static_assert(
        !Utils::IsComplex_v<T>, "CQTPlan requires a real type (F32 or F64).");
    using Complex = Utils::GetComplexType<T>;
    // friend class OmniDSP;

   public:
    ~CQTPlan();                                    // Definition in .cpp
    CQTPlan(CQTPlan&& other) noexcept;             // Definition in .cpp
    CQTPlan& operator=(CQTPlan&& other) noexcept;  // Definition in .cpp
    CQTPlan(const CQTPlan&) = delete;
    CQTPlan& operator=(const CQTPlan&) = delete;

    [[nodiscard]] Status execute(
        std::span<const T> input, std::span<Complex> output) const;
    size_t get_num_bins() const;
    size_t get_num_output_frames(size_t input_length) const;
    size_t get_hop_length() const;

    /**
     * @brief Creates a CQTPlan using the provided backend and parameters.
     * @param backend A reference to the backend implementation to use.
     * @param sample_rate The sample rate of the input signal.
     * @param min_freq The minimum frequency for the CQT.
     * @param max_freq The maximum frequency for the CQT.
     * @param bins_per_octave The number of bins per octave.
     * @param window_setup The WindowSetup to use for the CQT analysis windows.
     * @return An OmniExpected containing a unique_ptr to the CQTPlan on
     * success, or a Status on failure.
     */
    [[nodiscard]] static OmniExpected<std::unique_ptr<CQTPlan<T>>> create(
        const Abstract::Backend& backend,
        T sample_rate,  // Changed from Utils::GetRealType<T> as T is already
                        // real
        T min_freq,
        T max_freq,
        int bins_per_octave,
        const WindowSetup& window_setup  // Changed from WindowSpec
    )
    {
      OmniExpected<std::unique_ptr<Abstract::CQTPlanImpl<T>>> pimpl_expected;
      if constexpr (std::is_same_v<T, F32>) {
        pimpl_expected = backend.create_cqt_plan_impl_f32(
            sample_rate, min_freq, max_freq, bins_per_octave, window_setup);
      }
      else if constexpr (std::is_same_v<T, F64>) {
        pimpl_expected = backend.create_cqt_plan_impl_f64(
            sample_rate, min_freq, max_freq, bins_per_octave, window_setup);
      }
      else {
        // static_assert(always_false<T>::value, "Unsupported real type for
        // CQTPlan::create");
        return std::unexpected(Status::UnsupportedFeature);
      }

      if (!pimpl_expected) {
        return std::unexpected(pimpl_expected.error());
      }

      auto plan
          = CQTPlan<T>::create_from_impl(std::move(pimpl_expected.value()));
      if (!plan) {
        return std::unexpected(Status::Failure);  // Or AllocationError
      }
      return plan;
    }

    static std::unique_ptr<CQTPlan<T>> create_from_impl(
        std::unique_ptr<Abstract::CQTPlanImpl<T>> pimpl)
    {
      if (!pimpl) {
        return nullptr;
      }
      return std::unique_ptr<CQTPlan<T>>(new CQTPlan<T>(std::move(pimpl)));
    }

   private:
    explicit CQTPlan(std::unique_ptr<Abstract::CQTPlanImpl<T>> pimpl);
    std::unique_ptr<Abstract::CQTPlanImpl<T>> pimpl_;
  };

}  // namespace OmniDSP

#endif  // OMNIDSP_CQT_HPP
