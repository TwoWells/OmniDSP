/**
 * @file cqt.hpp
 * @brief Defines the interface for Constant-Q Transform (CQT) Plan objects.
 */

#ifndef OMNIDSP_CQT_H
#define OMNIDSP_CQT_H

#include <complex>
#include <cstddef>
#include <memory>
#include <span>
#include <utility>  // For std::move
#include <vector>

#include "OmniDSP/omnidsp_export.hpp"
#include "core_types.hpp"
#include "window.hpp"

// Forward declare backend Impl class
namespace OmniDSP::Abstract {
  template <typename T>
  class CQTPlanImpl;
}  // namespace OmniDSP::Abstract

namespace OmniDSP {

  class OmniDSP;

  /**
   * @brief Plan object for executing Constant-Q Transforms (CQT).
   * @tparam T The underlying real floating-point type (e.g., float, double).
   */
  template <typename T>  // T is real type here
  class OMNIDSP_EXPORT CQTPlan {
    // *** UPDATED Namespace ***
    static_assert(
        !Utils::IsComplex_v<T>, "CQTPlan requires a real type (F32 or F64).");
    // *** UPDATED Namespace ***
    using Complex = Utils::GetComplexType<T>;
    friend class OmniDSP;  // Keep friend for OmniDSP if needed
    // Removed friend declarations for Backend and Backend

   public:
    ~CQTPlan();
    CQTPlan(CQTPlan&& other) noexcept;
    CQTPlan& operator=(CQTPlan&& other) noexcept;
    CQTPlan(const CQTPlan&) = delete;
    CQTPlan& operator=(const CQTPlan&) = delete;

    [[nodiscard]] Status execute(
        std::span<const T> input, std::span<Complex> output) const;
    size_t get_num_bins() const;
    size_t get_num_output_frames(size_t input_length) const;
    size_t get_hop_length() const;

    /**
     * @brief Public static helper for factory methods to create instances.
     * @param pimpl A unique_ptr to the backend-specific implementation object.
     * @return A unique_ptr to the newly created public CQTPlan. Returns nullptr
     * if pimpl is null.
     */
    static std::unique_ptr<CQTPlan<T>> create_from_impl(
        std::unique_ptr<Abstract::CQTPlanImpl<T>> pimpl)
    {
      if (!pimpl) {
        return nullptr;
      }
      return std::unique_ptr<CQTPlan<T>>(new CQTPlan<T>(std::move(pimpl)));
    }

   private:
    /** @brief Private constructor, called by create_from_impl. */
    explicit CQTPlan(std::unique_ptr<Abstract::CQTPlanImpl<T>> pimpl);

    // Removed the private static helper and MakeUniqueEnabler struct

    std::unique_ptr<Abstract::CQTPlanImpl<T>> pimpl_;
  };

  // Definitions MUST be provided in a .cpp file

}  // namespace OmniDSP

#endif  // OMNIDSP_CQT_H
