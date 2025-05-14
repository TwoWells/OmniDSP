/**
 * @file omnidsp.cpp
 * @brief Implements the non-template OmniDSP class methods (factory,
 * constructor, etc.) and the OmniDSP::Builder methods.
 * Also contains explicit template instantiations for OmniDSP template methods.
 */

#include "OmniDSP/omnidsp.hpp"  // Main header, now includes omnidsp.tpp for template definitions

// Specific backend factory function headers (needed for create_xyz_backend())
#include "default/backend.hpp"     // For Abstract::create_default_backend()
#include "dispatcher/backend.hpp"  // For Dispatcher::Backend

// Conditionally include other backend headers for their factory functions
#include "OmniDSP/omnidsp_config.hpp"  // For OMNIDSP_BACKEND_*_ENABLED macros

#if OMNIDSP_BACKEND_ACCELERATE_ENABLED
#include "accelerate/backend.hpp"  // For Abstract::create_accelerate_backend()
#endif
#if OMNIDSP_BACKEND_ONEMKL_ENABLED
#include "onemkl/backend.hpp"  // For Abstract::create_onemkl_backend()
#endif
#if OMNIDSP_BACKEND_INTELIPP_ENABLED
#include "intelipp/backend.hpp"  // For Abstract::create_intelipp_backend()
#endif

// Standard Library Includes
#include <spdlog/spdlog.h>  // For logging

#include <map>        // For std::map
#include <optional>   // For std::optional
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <vector>     // For std::vector

// Include Params headers for explicit instantiations
#include "OmniDSP/params/convolution.hpp"
#include "OmniDSP/params/cqt.hpp"
#include "OmniDSP/params/fft.hpp"
#include "OmniDSP/params/fir_filter.hpp"
#include "OmniDSP/params/iir_filter.hpp"
#include "OmniDSP/params/resample.hpp"

namespace OmniDSP {

  //--------------------------------------------------------------------------
  // OmniDSP::Builder Method Definitions
  //--------------------------------------------------------------------------
  OmniDSP::Builder::Builder(std::optional<BackendType> primary_type)
      : primary_backend_type_(primary_type.value_or(BackendType::Default)),
        category_overrides_map_()
  {
    auto logger = spdlog::get("OmniDSP");
    if (logger) {
      if (primary_type.has_value()) {
        logger->debug(
            "OmniDSP::Builder initialized with primary backend: {}",
            get_backend_name(primary_type.value()));
      }
      else {
        logger->debug(
            "OmniDSP::Builder initialized. Primary backend defaults to: {}",
            get_backend_name(primary_backend_type_));
      }
    }
  }

  OmniDSP::Builder& OmniDSP::Builder::add_override(
      OperationCategory category, BackendType type)
  {
    category_overrides_map_[category] = type;
    auto logger = spdlog::get("OmniDSP");
    if (logger)
      logger->debug(
          "OmniDSP::Builder: Override added for category {} with backend {}",
          static_cast<int>(category),
          get_backend_name(type));
    return *this;
  }

  [[nodiscard]] OmniExpected<OmniDSP> OmniDSP::Builder::build() const
  {
    auto logger = spdlog::get("OmniDSP");
    if (logger)
      logger->debug(
          "OmniDSP::Builder::build() called. Primary: {}, Overrides count: {}",
          get_backend_name(primary_backend_type_),
          category_overrides_map_.size());
    return OmniDSP::create(primary_backend_type_, category_overrides_map_);
  }

  //--------------------------------------------------------------------------
  // OmniDSP Non-Template Method Definitions
  //--------------------------------------------------------------------------
  OmniDSP::OmniDSP(std::unique_ptr<Abstract::Backend> impl)
      : pimpl_(std::move(impl))
  {
    if (!pimpl_) {
      auto logger = spdlog::get("OmniDSP");
      if (logger)
        logger->critical(
            "OmniDSP constructed with a null backend implementation pointer. "
            "This indicates a severe internal error.");
      throw std::runtime_error(
          "OmniDSP cannot be constructed with a null backend implementation.");
    }
  }

  OmniDSP::~OmniDSP() = default;
  OmniDSP::OmniDSP(OmniDSP&& other) noexcept = default;
  OmniDSP& OmniDSP::operator=(OmniDSP&& other) noexcept = default;

  BackendType OmniDSP::get_backend() const
  {
    if (!pimpl_) {
      auto logger = spdlog::get("OmniDSP");
      if (logger)
        logger->error(
            "OmniDSP::get_backend() called on an invalid (e.g., moved-from) "
            "instance.");
      throw std::runtime_error(
          "Attempted to use an invalid (moved-from or improperly constructed) "
          "OmniDSP instance.");
    }
    return pimpl_->get_backend();
  }

  // --- Static Factory Functions ---
  namespace {
    std::unique_ptr<Abstract::Backend> create_concrete_backend_instance(
        BackendType type, spdlog::logger* logger)
    {
      switch (type) {
        case BackendType::Default:
          if constexpr (OMNIDSP_BACKEND_DEFAULT_ENABLED) {
            if (logger)
              logger->debug("Factory: Creating Default backend instance.");
            return Abstract::create_default_backend();
          }
          else {
            if (logger)
              logger->error(
                  "Factory: Default backend requested but "
                  "OMNIDSP_BACKEND_DEFAULT_ENABLED is false. This is a "
                  "critical configuration error.");
            return nullptr;
          }
        case BackendType::Accelerate:
          if constexpr (OMNIDSP_BACKEND_ACCELERATE_ENABLED) {
            if (logger)
              logger->debug("Factory: Creating Accelerate backend instance.");
            return Abstract::create_accelerate_backend();
          }
          else {
            if (logger)
              logger->warn(
                  "Factory: Accelerate backend requested but not enabled in "
                  "this build (OMNIDSP_BACKEND_ACCELERATE_ENABLED=0).");
            return nullptr;
          }
        case BackendType::OneMKL:
          if constexpr (OMNIDSP_BACKEND_ONEMKL_ENABLED) {
            if (logger)
              logger->debug("Factory: Creating OneMKL backend instance.");
            return Abstract::create_onemkl_backend();
          }
          else {
            if (logger)
              logger->warn(
                  "Factory: OneMKL backend requested but not enabled in this "
                  "build (OMNIDSP_BACKEND_ONEMKL_ENABLED=0).");
            return nullptr;
          }
        case BackendType::IntelIPP:
          if constexpr (OMNIDSP_BACKEND_INTELIPP_ENABLED) {
            if (logger)
              logger->debug("Factory: Creating IntelIPP backend instance.");
            return Abstract::create_intelipp_backend();
          }
          else {
            if (logger)
              logger->warn(
                  "Factory: IntelIPP backend requested but not enabled in this "
                  "build (OMNIDSP_BACKEND_INTELIPP_ENABLED=0).");
            return nullptr;
          }
        default:
          if (logger)
            logger->error(
                "Factory: Unknown or unsupported backend type requested: {}",
                static_cast<int>(type));
          return nullptr;
      }
    }
  }  // namespace

  [[nodiscard]] OmniExpected<OmniDSP> OmniDSP::create(BackendType backend_type)
  {
    return create(backend_type, {});
  }

  [[nodiscard]] OmniExpected<OmniDSP> OmniDSP::create(
      BackendType primary_backend_type,
      const std::map<OperationCategory, BackendType>& category_overrides)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }

    std::unique_ptr<Abstract::Backend> primary_backend_pimpl = nullptr;
    try {
      primary_backend_pimpl = create_concrete_backend_instance(
          primary_backend_type, logger.get());
      if (!primary_backend_pimpl) {
        logger->error(
            "Failed to create primary backend of type: {}. Check build "
            "configuration and backend availability.",
            get_backend_name(primary_backend_type));
        return std::unexpected(OmniStatus::BackendError);
      }
    }
    catch (const std::exception& e) {
      logger->error(
          "Exception during primary backend creation (type {}): {}",
          get_backend_name(primary_backend_type),
          e.what());
      return std::unexpected(OmniStatus::BackendError);
    }
    catch (...) {
      logger->error(
          "Unknown exception during primary backend creation (type {}).",
          get_backend_name(primary_backend_type));
      return std::unexpected(OmniStatus::Failure);
    }

    if (category_overrides.empty()) {
      logger->info(
          "OmniDSP::create called with no overrides. Using primary backend: {}",
          get_backend_name(primary_backend_pimpl->get_backend()));
      return OmniDSP(std::move(primary_backend_pimpl));
    }

    logger->info(
        "OmniDSP::create called with overrides. Primary backend: {}. "
        "Configuring DispatcherBackend...",
        get_backend_name(primary_backend_pimpl->get_backend()));
    std::map<OperationCategory, std::shared_ptr<Abstract::Backend>>
        dispatcher_overrides_map;
    bool all_overrides_created_successfully = true;

    for (const auto& pair : category_overrides) {
      OperationCategory cat = pair.first;
      BackendType override_type = pair.second;
      logger->debug(
          "  Attempting to create override backend of type {} for category "
          "{}...",
          get_backend_name(override_type),
          static_cast<int>(cat));
      std::unique_ptr<Abstract::Backend> override_pimpl
          = create_concrete_backend_instance(override_type, logger.get());
      if (override_pimpl) {
        dispatcher_overrides_map[cat] = std::move(override_pimpl);
        logger->info(
            "    Successfully created override backend {} for category {}.",
            get_backend_name(dispatcher_overrides_map[cat]->get_backend()),
            static_cast<int>(cat));
      }
      else {
        logger->error(
            "    Failed to create override backend of type {} for category {}. "
            "Dispatcher setup cannot proceed with this override.",
            get_backend_name(override_type),
            static_cast<int>(cat));
        all_overrides_created_successfully = false;
        break;
      }
    }

    if (!all_overrides_created_successfully) {
      logger->error(
          "One or more specified backend overrides could not be created. "
          "OmniDSP instance creation failed.");
      return std::unexpected(OmniStatus::BackendError);
    }

    try {
      auto dispatcher_pimpl = std::make_unique<Dispatcher::Backend>(
          std::move(primary_backend_pimpl), dispatcher_overrides_map);
      logger->info(
          "DispatcherBackend configured. Creating OmniDSP instance with "
          "DispatcherBackend.");
      return OmniDSP(std::move(dispatcher_pimpl));
    }
    catch (const std::exception& e) {
      logger->error(
          "Exception during Dispatcher::Backend construction: {}", e.what());
      return std::unexpected(OmniStatus::BackendError);
    }
    catch (...) {
      logger->error(
          "Unknown exception during Dispatcher::Backend construction.");
      return std::unexpected(OmniStatus::Failure);
    }
  }

  //--------------------------------------------------------------------------
  // Explicit Template Instantiations for OmniDSP methods
  //--------------------------------------------------------------------------
  // One-off operations
  template OMNIDSP_EXPORT OmniExpected<std::vector<F32>> OmniDSP::convolve<F32>(
      const std::vector<F32>&,
      const std::vector<F32>&,
      ConvolutionType,
      ConvolutionMethod) const;
  template OMNIDSP_EXPORT OmniExpected<std::vector<F64>> OmniDSP::convolve<F64>(
      const std::vector<F64>&,
      const std::vector<F64>&,
      ConvolutionType,
      ConvolutionMethod) const;
  template OMNIDSP_EXPORT OmniExpected<std::vector<C32>> OmniDSP::convolve<C32>(
      const std::vector<C32>&,
      const std::vector<C32>&,
      ConvolutionType,
      ConvolutionMethod) const;
  template OMNIDSP_EXPORT OmniExpected<std::vector<C64>> OmniDSP::convolve<C64>(
      const std::vector<C64>&,
      const std::vector<C64>&,
      ConvolutionType,
      ConvolutionMethod) const;

  template OMNIDSP_EXPORT OmniExpected<std::vector<F32>>
  OmniDSP::correlate<F32>(
      const std::vector<F32>&,
      const std::vector<F32>&,
      ConvolutionType,
      ConvolutionMethod) const;
  template OMNIDSP_EXPORT OmniExpected<std::vector<F64>>
  OmniDSP::correlate<F64>(
      const std::vector<F64>&,
      const std::vector<F64>&,
      ConvolutionType,
      ConvolutionMethod) const;
  template OMNIDSP_EXPORT OmniExpected<std::vector<C32>>
  OmniDSP::correlate<C32>(
      const std::vector<C32>&,
      const std::vector<C32>&,
      ConvolutionType,
      ConvolutionMethod) const;
  template OMNIDSP_EXPORT OmniExpected<std::vector<C64>>
  OmniDSP::correlate<C64>(
      const std::vector<C64>&,
      const std::vector<C64>&,
      ConvolutionType,
      ConvolutionMethod) const;

  template OMNIDSP_EXPORT OmniExpected<std::vector<C32>> OmniDSP::fft<C32>(
      const std::vector<C32>&) const;
  template OMNIDSP_EXPORT OmniExpected<std::vector<C64>> OmniDSP::fft<C64>(
      const std::vector<C64>&) const;
  template OMNIDSP_EXPORT OmniExpected<std::vector<C32>> OmniDSP::ifft<C32>(
      const std::vector<C32>&) const;
  template OMNIDSP_EXPORT OmniExpected<std::vector<C64>> OmniDSP::ifft<C64>(
      const std::vector<C64>&) const;
  template OMNIDSP_EXPORT OmniExpected<std::vector<C32>> OmniDSP::rfft<F32>(
      const std::vector<F32>&) const;  // T is F32, output is C32
  template OMNIDSP_EXPORT OmniExpected<std::vector<C64>> OmniDSP::rfft<F64>(
      const std::vector<F64>&) const;  // T is F64, output is C64
  template OMNIDSP_EXPORT OmniExpected<std::vector<F32>> OmniDSP::irfft<F32>(
      const std::vector<C32>&,
      size_t) const;  // T is F32 (output), input is C32
  template OMNIDSP_EXPORT OmniExpected<std::vector<F64>> OmniDSP::irfft<F64>(
      const std::vector<C64>&,
      size_t) const;  // T is F64 (output), input is C64

  template OMNIDSP_EXPORT OmniExpected<std::vector<F32>>
  OmniDSP::generate_window<F32>(const WindowSetup&) const;
  template OMNIDSP_EXPORT OmniExpected<std::vector<F64>>
  OmniDSP::generate_window<F64>(const WindowSetup&) const;

  // Deprecated window functions
  template OMNIDSP_EXPORT OmniExpected<std::vector<F32>>
      OmniDSP::bartlett_window<F32>(size_t) const;
  template OMNIDSP_EXPORT OmniExpected<std::vector<F64>>
      OmniDSP::bartlett_window<F64>(size_t) const;
  template OMNIDSP_EXPORT OmniExpected<std::vector<F32>>
      OmniDSP::blackman_window<F32>(size_t) const;
  template OMNIDSP_EXPORT OmniExpected<std::vector<F64>>
      OmniDSP::blackman_window<F64>(size_t) const;
  template OMNIDSP_EXPORT OmniExpected<std::vector<F32>>
      OmniDSP::flattop_window<F32>(size_t) const;
  template OMNIDSP_EXPORT OmniExpected<std::vector<F64>>
      OmniDSP::flattop_window<F64>(size_t) const;
  template OMNIDSP_EXPORT OmniExpected<std::vector<F32>>
  OmniDSP::gaussian_window<F32>(size_t, double) const;
  template OMNIDSP_EXPORT OmniExpected<std::vector<F64>>
  OmniDSP::gaussian_window<F64>(size_t, double) const;
  template OMNIDSP_EXPORT OmniExpected<std::vector<F32>>
      OmniDSP::hamming_window<F32>(size_t) const;
  template OMNIDSP_EXPORT OmniExpected<std::vector<F64>>
      OmniDSP::hamming_window<F64>(size_t) const;
  template OMNIDSP_EXPORT OmniExpected<std::vector<F32>>
      OmniDSP::hann_window<F32>(size_t) const;
  template OMNIDSP_EXPORT OmniExpected<std::vector<F64>>
      OmniDSP::hann_window<F64>(size_t) const;
  template OMNIDSP_EXPORT OmniExpected<std::vector<F32>>
  OmniDSP::kaiser_window<F32>(size_t, double) const;
  template OMNIDSP_EXPORT OmniExpected<std::vector<F64>>
  OmniDSP::kaiser_window<F64>(size_t, double) const;
  template OMNIDSP_EXPORT OmniExpected<std::vector<F32>>
      OmniDSP::rectangular_window<F32>(size_t) const;
  template OMNIDSP_EXPORT OmniExpected<std::vector<F64>>
      OmniDSP::rectangular_window<F64>(size_t) const;
  template OMNIDSP_EXPORT OmniExpected<std::vector<F32>>
      OmniDSP::triangular_window<F32>(size_t) const;
  template OMNIDSP_EXPORT OmniExpected<std::vector<F64>>
      OmniDSP::triangular_window<F64>(size_t) const;

  // Plan and Processor Factories
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<FFTPlan<C32>>>
  OmniDSP::create_plan(const Params::FFT&) const;
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<FFTPlan<C64>>>
  OmniDSP::create_plan(const Params::FFT&) const;
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<RFFTPlan<F32>>>
  OmniDSP::create_plan(const Params::RFFT&) const;
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<RFFTPlan<F64>>>
  OmniDSP::create_plan(const Params::RFFT&) const;
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<ConvolutionPlan<F32>>>
  OmniDSP::create_plan(
      const Params::Convolution&, const std::vector<F32>&) const;
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<ConvolutionPlan<F64>>>
  OmniDSP::create_plan(
      const Params::Convolution&, const std::vector<F64>&) const;
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<ConvolutionPlan<C32>>>
  OmniDSP::create_plan(
      const Params::Convolution&, const std::vector<C32>&) const;
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<ConvolutionPlan<C64>>>
  OmniDSP::create_plan(
      const Params::Convolution&, const std::vector<C64>&) const;

  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<FIRFilterProcessor<F32>>>
  OmniDSP::create_processor(const Params::FIRFilter&) const;
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<FIRFilterProcessor<F64>>>
  OmniDSP::create_processor(const Params::FIRFilter&) const;
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<FIRFilterProcessor<C32>>>
  OmniDSP::create_processor(const Params::FIRFilter&) const;
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<FIRFilterProcessor<C64>>>
  OmniDSP::create_processor(const Params::FIRFilter&) const;
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<FIRFilterProcessor<F32>>>
  OmniDSP::create_processor(const Coefs::FIRFilter<F32>&) const;
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<FIRFilterProcessor<F64>>>
  OmniDSP::create_processor(const Coefs::FIRFilter<F64>&) const;
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<FIRFilterProcessor<C32>>>
  OmniDSP::create_processor(const Coefs::FIRFilter<C32>&) const;
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<FIRFilterProcessor<C64>>>
  OmniDSP::create_processor(const Coefs::FIRFilter<C64>&) const;

  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<IIRFilterProcessor<F32>>>
  OmniDSP::create_processor(const Params::IIRFilter&) const;
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<IIRFilterProcessor<F64>>>
  OmniDSP::create_processor(const Params::IIRFilter&) const;
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<IIRFilterProcessor<F32>>>
  OmniDSP::create_processor(const Coefs::IIRFilterSOS&) const;
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<IIRFilterProcessor<F64>>>
  OmniDSP::create_processor(const Coefs::IIRFilterSOS&) const;

  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<ResampleProcessor<F32>>>
  OmniDSP::create_processor(const Params::Resample&) const;
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<ResampleProcessor<F64>>>
  OmniDSP::create_processor(const Params::Resample&) const;
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<ResampleProcessor<F32>>>
  OmniDSP::create_processor(const Design::Resample&) const;
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<ResampleProcessor<F64>>>
  OmniDSP::create_processor(const Design::Resample&) const;

  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<CQTProcessor<F32>>>
  OmniDSP::create_processor(const Params::CQT&) const;
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<CQTProcessor<F64>>>
  OmniDSP::create_processor(const Params::CQT&) const;
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<CQTProcessor<F32>>>
  OmniDSP::create_processor(const Design::CQT&) const;
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<CQTProcessor<F64>>>
  OmniDSP::create_processor(const Design::CQT&) const;

  // Filter Design Methods
  template OMNIDSP_EXPORT OmniExpected<Coefs::FIRFilter<F32>>
  OmniDSP::design_fir_filter<F32>(const Params::FIRFilter&) const;
  template OMNIDSP_EXPORT OmniExpected<Coefs::FIRFilter<F64>>
  OmniDSP::design_fir_filter<F64>(const Params::FIRFilter&) const;
  template OMNIDSP_EXPORT OmniExpected<Coefs::IIRFilterSOS>
  OmniDSP::design_iir_filter<F32>(const Params::IIRFilter&) const;
  template OMNIDSP_EXPORT OmniExpected<Coefs::IIRFilterSOS>
  OmniDSP::design_iir_filter<F64>(const Params::IIRFilter&) const;

}  // namespace OmniDSP
