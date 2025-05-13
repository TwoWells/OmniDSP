/**
 * @file cqt.hpp
 * @brief Defines the interface for Constant-Q Transform (CQT) Processor objects
 * and related design specifications (Design::CQT, Design::CQTOctave).
 */

#ifndef OMNIDSP_CQT_HPP
#define OMNIDSP_CQT_HPP

#include <complex>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "OmniDSP/core_types.hpp"
#include "OmniDSP/design/cqt.hpp"
#include "OmniDSP/omnidsp_export.hpp"
#include "OmniDSP/window.hpp"  // For WindowSetup
// This include defines ::OmniDSP::Abstract::Backend and
// ::OmniDSP::Abstract::CQTPlanImpl
#include "interface/backend.hpp"

namespace OmniDSP {

  template <typename T>
  class OMNIDSP_EXPORT CQTPlan {
    static_assert(
        !Utils::IsComplex_v<T>,
        "CQTPlan/Processor requires a real type (F32 or F64).");
    using Complex = Utils::GetComplexType<T>;

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

    // Use fully qualified name for Abstract::Backend
    [[nodiscard]] static OmniExpected<std::unique_ptr<CQTPlan<T>>> create(
        const ::OmniDSP::Abstract::Backend& backend, const Design::CQT& design);

    // Use fully qualified name for Abstract::CQTPlanImpl
    static std::unique_ptr<CQTPlan<T>> create_from_impl(
        std::unique_ptr<::OmniDSP::Abstract::CQTPlanImpl<T>> pimpl)
    {
      if (!pimpl) {
        return nullptr;
      }
      return std::unique_ptr<CQTPlan<T>>(new CQTPlan<T>(std::move(pimpl)));
    }

   private:
    // Use fully qualified name for Abstract::CQTPlanImpl
    explicit CQTPlan(
        std::unique_ptr<::OmniDSP::Abstract::CQTPlanImpl<T>> pimpl);
    std::unique_ptr<::OmniDSP::Abstract::CQTPlanImpl<T>> pimpl_;
  };

}  // namespace OmniDSP

#endif  // OMNIDSP_CQT_HPP
