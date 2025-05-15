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
#include <spdlog/fmt/ostr.h>  // For logging custom types with operator<<
#include <spdlog/spdlog.h>    // For logging

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

// Ensure core_types is included for enums and their operator<<
#include "OmniDSP/core_types.hpp"

namespace OmniDSP {

  //--------------------------------------------------------------------------
  // OmniDSP::Builder Method Definitions
  //--------------------------------------------------------------------------
  OmniDSP::Builder::Builder(std::optional<BackendType> primary_type)
      : primary_backend_type_(primary_type.value_or(BackendType::Default)),
        category_overrides_map_()
  {
    auto logger = spdlog::get("OmniDSP");
    if (logger
        && logger->should_log(spdlog::level::debug)) {  // Check log level
      if (primary_type.has_value()) {
        logger->debug(
            "OmniDSP::Builder initialized with primary backend: {}",
            primary_type.value());  // Log enum directly
      }
      else {
        logger->debug(
            "OmniDSP::Builder initialized. Primary backend defaults to: {}",
            primary_backend_type_);  // Log enum directly
      }
    }
  }

  OmniDSP::Builder& OmniDSP::Builder::add_override(
      OperationCategory category, BackendType type)
  {
    category_overrides_map_[category] = type;
    auto logger = spdlog::get("OmniDSP");
    if (logger
        && logger->should_log(spdlog::level::debug)) {  // Check log level
      logger->debug(
          "OmniDSP::Builder: Override added for category {} with backend {}",
          category,  // Log enum directly
          type);     // Log enum directly
    }
    return *this;
  }

  [[nodiscard]] OmniExpected<OmniDSP> OmniDSP::Builder::build() const
  {
    auto logger = spdlog::get("OmniDSP");
    if (logger
        && logger->should_log(spdlog::level::debug)) {  // Check log level
      logger->debug(
          "OmniDSP::Builder::build() called. Primary: {}, Overrides count: {}",
          primary_backend_type_,  // Log enum directly
          category_overrides_map_.size());
    }
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
      if (logger)  // No need to check log level for critical
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
      if (logger)  // No need to check log level for error
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
  namespace {  // Anonymous namespace for internal helper
    std::unique_ptr<Abstract::Backend> create_concrete_backend_instance(
        BackendType type, spdlog::logger* logger_ptr)  // Pass logger pointer
    {
      switch (type) {
        case BackendType::Default:
          if constexpr (OMNIDSP_BACKEND_DEFAULT_ENABLED) {
            if (logger_ptr && logger_ptr->should_log(spdlog::level::debug))
              logger_ptr->debug("Factory: Creating Default backend instance.");
            return Abstract::create_default_backend();
          }
          else {
            if (logger_ptr)  // No need to check log level for error
              logger_ptr->error(
                  "Factory: Default backend requested but "
                  "OMNIDSP_BACKEND_DEFAULT_ENABLED is false. This is a "
                  "critical configuration error.");
            return nullptr;
          }
        case BackendType::Accelerate:
          if constexpr (OMNIDSP_BACKEND_ACCELERATE_ENABLED) {
            if (logger_ptr && logger_ptr->should_log(spdlog::level::debug))
              logger_ptr->debug(
                  "Factory: Creating Accelerate backend instance.");
            return Abstract::create_accelerate_backend();
          }
          else {
            if (logger_ptr && logger_ptr->should_log(spdlog::level::warn))
              logger_ptr->warn(
                  "Factory: Accelerate backend requested but not enabled in "
                  "this build (OMNIDSP_BACKEND_ACCELERATE_ENABLED=0).");
            return nullptr;
          }
        case BackendType::OneMKL:
          if constexpr (OMNIDSP_BACKEND_ONEMKL_ENABLED) {
            if (logger_ptr && logger_ptr->should_log(spdlog::level::debug))
              logger_ptr->debug("Factory: Creating OneMKL backend instance.");
            return Abstract::create_onemkl_backend();
          }
          else {
            if (logger_ptr && logger_ptr->should_log(spdlog::level::warn))
              logger_ptr->warn(
                  "Factory: OneMKL backend requested but not enabled in this "
                  "build (OMNIDSP_BACKEND_ONEMKL_ENABLED=0).");
            return nullptr;
          }
        case BackendType::IntelIPP:
          if constexpr (OMNIDSP_BACKEND_INTELIPP_ENABLED) {
            if (logger_ptr && logger_ptr->should_log(spdlog::level::debug))
              logger_ptr->debug("Factory: Creating IntelIPP backend instance.");
            return Abstract::create_intelipp_backend();
          }
          else {
            if (logger_ptr && logger_ptr->should_log(spdlog::level::warn))
              logger_ptr->warn(
                  "Factory: IntelIPP backend requested but not enabled in this "
                  "build (OMNIDSP_BACKEND_INTELIPP_ENABLED=0).");
            return nullptr;
          }
        default:           // Dispatcher is not created here, it's a wrapper
          if (logger_ptr)  // No need to check log level for error
            logger_ptr->error(
                "Factory: Unknown or unsupported backend type requested for "
                "concrete instance: {}",
                type);  // Log enum directly
          return nullptr;
      }
    }
  }  // namespace

  [[nodiscard]] OmniExpected<OmniDSP> OmniDSP::create(BackendType backend_type)
  {
    return create(
        backend_type, {});  // Delegate to the more general create function
  }

  [[nodiscard]] OmniExpected<OmniDSP> OmniDSP::create(
      BackendType primary_backend_type,
      const std::map<OperationCategory, BackendType>& category_overrides)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      // This is a critical point; if logging isn't set up, we might have
      // issues. For robustness, one might consider initializing a default
      // logger here if absolutely necessary, but typically, logger setup is an
      // application-level concern. For now, we'll proceed, and if logger is
      // null, logging calls will be no-ops or handled by spdlog's default.
      // However, it's better if the logger is guaranteed to be available.
      // If spdlog::get returns nullptr, subsequent logger->... calls will
      // crash. It's safer to ensure a logger exists or handle the nullptr case
      // explicitly. For this example, let's assume spdlog::default_logger() is
      // a safe fallback if "OmniDSP" is not found.
      logger = spdlog::default_logger();
    }

    std::unique_ptr<Abstract::Backend> primary_backend_pimpl = nullptr;
    try {
      primary_backend_pimpl = create_concrete_backend_instance(
          primary_backend_type, logger.get());  // Pass the logger pointer
      if (!primary_backend_pimpl) {
        if (logger)
          logger->error(  // Check logger before use
              "Failed to create primary backend of type: {}. Check build "
              "configuration and backend availability.",
              primary_backend_type);  // Log enum directly
        return std::unexpected(OmniStatus::BackendError);
      }
    }
    catch (const std::exception& e) {
      if (logger)
        logger->error(  // Check logger
            "Exception during primary backend creation (type {}): {}",
            primary_backend_type,  // Log enum directly
            e.what());
      return std::unexpected(OmniStatus::BackendError);
    }
    catch (...) {
      if (logger)
        logger->error(  // Check logger
            "Unknown exception during primary backend creation (type {}).",
            primary_backend_type);  // Log enum directly
      return std::unexpected(OmniStatus::Failure);
    }

    if (category_overrides.empty()) {
      if (logger
          && logger->should_log(spdlog::level::info)) {  // Check log level
        logger->info(
            "OmniDSP::create called with no overrides. Using primary backend: "
            "{}",
            primary_backend_pimpl->get_backend());  // Log enum directly
      }
      return OmniDSP(std::move(primary_backend_pimpl));
    }

    if (logger && logger->should_log(spdlog::level::info)) {  // Check log level
      logger->info(
          "OmniDSP::create called with overrides. Primary backend: {}. "
          "Configuring DispatcherBackend...",
          primary_backend_pimpl->get_backend());  // Log enum directly
    }

    std::map<OperationCategory, std::shared_ptr<Abstract::Backend>>
        dispatcher_overrides_map;
    bool all_overrides_created_successfully = true;

    for (const auto& pair : category_overrides) {
      OperationCategory cat = pair.first;
      BackendType override_type = pair.second;
      if (logger
          && logger->should_log(spdlog::level::debug)) {  // Check log level
        logger->debug(
            "  Attempting to create override backend of type {} for category "
            "{}...",
            override_type,  // Log enum directly
            cat);           // Log enum directly
      }
      std::unique_ptr<Abstract::Backend> override_pimpl
          = create_concrete_backend_instance(
              override_type, logger.get());  // Pass logger
      if (override_pimpl) {
        dispatcher_overrides_map[cat] = std::move(override_pimpl);
        if (logger
            && logger->should_log(spdlog::level::info)) {  // Check log level
          logger->info(
              "    Successfully created override backend {} for category {}.",
              dispatcher_overrides_map[cat]
                  ->get_backend(),  // Log enum directly
              cat);                 // Log enum directly
        }
      }
      else {
        if (logger)
          logger->error(  // No need to check log level for error
              "    Failed to create override backend of type {} for category "
              "{}. "
              "Dispatcher setup cannot proceed with this override.",
              override_type,  // Log enum directly
              cat);           // Log enum directly
        all_overrides_created_successfully = false;
        break;
      }
    }

    if (!all_overrides_created_successfully) {
      if (logger)
        logger->error(  // No need to check log level for error
            "One or more specified backend overrides could not be created. "
            "OmniDSP instance creation failed.");
      return std::unexpected(OmniStatus::BackendError);
    }

    try {
      auto dispatcher_pimpl = std::make_unique<Dispatcher::Backend>(
          std::move(primary_backend_pimpl), dispatcher_overrides_map);
      if (logger
          && logger->should_log(spdlog::level::info)) {  // Check log level
        logger->info(
            "DispatcherBackend configured. Creating OmniDSP instance with "
            "DispatcherBackend.");
      }
      return OmniDSP(std::move(dispatcher_pimpl));
    }
    catch (const std::exception& e) {
      if (logger)
        logger->error(  // No need to check log level for error
            "Exception during Dispatcher::Backend construction: {}",
            e.what());
      return std::unexpected(OmniStatus::BackendError);
    }
    catch (...) {
      if (logger)
        logger->error(  // No need to check log level for error
            "Unknown exception during Dispatcher::Backend construction.");
      return std::unexpected(OmniStatus::Failure);
    }
  }

  //--------------------------------------------------------------------------
  // Explicit Template Instantiations for OmniDSP methods
  //--------------------------------------------------------------------------
  // (These remain unchanged as they don't directly involve the enum-to-string
  // logging) One-off operations
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
      const std::vector<F32>&) const;
  template OMNIDSP_EXPORT OmniExpected<std::vector<C64>> OmniDSP::rfft<F64>(
      const std::vector<F64>&) const;
  template OMNIDSP_EXPORT OmniExpected<std::vector<F32>> OmniDSP::irfft<F32>(
      const std::vector<C32>&, size_t) const;
  template OMNIDSP_EXPORT OmniExpected<std::vector<F64>> OmniDSP::irfft<F64>(
      const std::vector<C64>&, size_t) const;

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

  // Plan::FFT Factories
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<Plan::FFT<C32>>>
  OmniDSP::create_plan(const Params::FFT&) const;
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<Plan::FFT<C64>>>
  OmniDSP::create_plan(const Params::FFT&) const;

  // Plan::RFFT Factories
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<Plan::RFFT<F32>>>
  OmniDSP::create_plan(const Params::RFFT&) const;
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<Plan::RFFT<F64>>>
  OmniDSP::create_plan(const Params::RFFT&) const;

  // Plan::Convolution Factories
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<Plan::Convolution<F32>>>
  OmniDSP::create_plan(
      const Params::Convolution&, const std::vector<F32>&) const;
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<Plan::Convolution<F64>>>
  OmniDSP::create_plan(
      const Params::Convolution&, const std::vector<F64>&) const;
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<Plan::Convolution<C32>>>
  OmniDSP::create_plan(
      const Params::Convolution&, const std::vector<C32>&) const;
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<Plan::Convolution<C64>>>
  OmniDSP::create_plan(
      const Params::Convolution&, const std::vector<C64>&) const;

  // Plan::Correlation Factories
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<Plan::Correlation<F32>>>
  OmniDSP::create_plan(
      const Params::Correlation&, const std::vector<F32>&) const;
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<Plan::Correlation<F64>>>
  OmniDSP::create_plan(
      const Params::Correlation&, const std::vector<F64>&) const;
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<Plan::Correlation<C32>>>
  OmniDSP::create_plan(
      const Params::Correlation&, const std::vector<C32>&) const;
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<Plan::Correlation<C64>>>
  OmniDSP::create_plan(
      const Params::Correlation&, const std::vector<C64>&) const;

  // Processor Factories
  template OMNIDSP_EXPORT
      OmniExpected<std::unique_ptr<Processor::FIRFilter<F32>>>
      OmniDSP::create_processor(const Params::FIRFilter&) const;
  template OMNIDSP_EXPORT
      OmniExpected<std::unique_ptr<Processor::FIRFilter<F64>>>
      OmniDSP::create_processor(const Params::FIRFilter&) const;
  template OMNIDSP_EXPORT
      OmniExpected<std::unique_ptr<Processor::FIRFilter<C32>>>
      OmniDSP::create_processor(const Params::FIRFilter&) const;
  template OMNIDSP_EXPORT
      OmniExpected<std::unique_ptr<Processor::FIRFilter<C64>>>
      OmniDSP::create_processor(const Params::FIRFilter&) const;
  template OMNIDSP_EXPORT
      OmniExpected<std::unique_ptr<Processor::FIRFilter<F32>>>
      OmniDSP::create_processor(const Coefs::FIRFilter<F32>&) const;
  template OMNIDSP_EXPORT
      OmniExpected<std::unique_ptr<Processor::FIRFilter<F64>>>
      OmniDSP::create_processor(const Coefs::FIRFilter<F64>&) const;
  template OMNIDSP_EXPORT
      OmniExpected<std::unique_ptr<Processor::FIRFilter<C32>>>
      OmniDSP::create_processor(const Coefs::FIRFilter<C32>&) const;
  template OMNIDSP_EXPORT
      OmniExpected<std::unique_ptr<Processor::FIRFilter<C64>>>
      OmniDSP::create_processor(const Coefs::FIRFilter<C64>&) const;

  template OMNIDSP_EXPORT
      OmniExpected<std::unique_ptr<Processor::IIRFilter<F32>>>
      OmniDSP::create_processor(const Params::IIRFilter&) const;
  template OMNIDSP_EXPORT
      OmniExpected<std::unique_ptr<Processor::IIRFilter<F64>>>
      OmniDSP::create_processor(const Params::IIRFilter&) const;
  template OMNIDSP_EXPORT
      OmniExpected<std::unique_ptr<Processor::IIRFilter<F32>>>
      OmniDSP::create_processor(const Coefs::IIRFilterSOS&) const;
  template OMNIDSP_EXPORT
      OmniExpected<std::unique_ptr<Processor::IIRFilter<F64>>>
      OmniDSP::create_processor(const Coefs::IIRFilterSOS&) const;

  template OMNIDSP_EXPORT
      OmniExpected<std::unique_ptr<Processor::Resample<F32>>>
      OmniDSP::create_processor(const Params::Resample&) const;
  template OMNIDSP_EXPORT
      OmniExpected<std::unique_ptr<Processor::Resample<F64>>>
      OmniDSP::create_processor(const Params::Resample&) const;
  template OMNIDSP_EXPORT
      OmniExpected<std::unique_ptr<Processor::Resample<F32>>>
      OmniDSP::create_processor(const Design::Resample&) const;
  template OMNIDSP_EXPORT
      OmniExpected<std::unique_ptr<Processor::Resample<F64>>>
      OmniDSP::create_processor(const Design::Resample&) const;

  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<Processor::CQT<F32>>>
  OmniDSP::create_processor(const Params::CQT&) const;
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<Processor::CQT<F64>>>
  OmniDSP::create_processor(const Params::CQT&) const;
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<Processor::CQT<F32>>>
  OmniDSP::create_processor(const Design::CQT&) const;
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<Processor::CQT<F64>>>
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
