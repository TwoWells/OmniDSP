/**
 * @file transform.cpp
 * @brief Implements constructors and stream operators for transform
 * specification structs.
 */
#include "OmniDSP/specs/transform.hpp"  // Corresponding header

#include <ostream>    // For std::ostream definitions
#include <stdexcept>  // For std::invalid_argument
#include <string>     // For std::to_string, std::string

// spdlog for logging in constructors
#include "spdlog/spdlog.h"
// fmt is included via the header for ostream_formatter, but if direct
// formatting was needed here, it would be included.

namespace OmniDSP {
  namespace Transform {

    // --- FFT ---
    FFT::FFT(size_t len) : length(len)
    {
      if (length == 0) {
        std::string msg = "Transform::FFT: length (" + std::to_string(length)
                          + ") must be positive.";
        auto logger = spdlog::get("OmniDSP");
        if (!logger) {
          logger = spdlog::default_logger();
        }
        if (logger) {
          logger->error(msg);
        }
        throw std::invalid_argument(msg);
      }
    }

    std::ostream& operator<<(std::ostream& os, const FFT& spec)
    {
      os << "Transform::FFT(length: " << spec.length << ")";
      return os;
    }

    // --- RFFT ---
    RFFT::RFFT(size_t len) : real_length(len)
    {
      if (real_length == 0) {
        std::string msg = "Transform::RFFT: real_length ("
                          + std::to_string(real_length) + ") must be positive.";
        auto logger = spdlog::get("OmniDSP");
        if (!logger) {
          logger = spdlog::default_logger();
        }
        if (logger) {
          logger->error(msg);
        }
        throw std::invalid_argument(msg);
      }
    }

    std::ostream& operator<<(std::ostream& os, const RFFT& spec)
    {
      os << "Transform::RFFT(real_length: " << spec.real_length << ")";
      return os;
    }

    // --- DFT ---
    DFT::DFT(size_t len) : length(len)
    {
      if (length == 0) {
        std::string msg = "Transform::DFT: length (" + std::to_string(length)
                          + ") must be positive.";
        auto logger = spdlog::get("OmniDSP");
        if (!logger) {
          logger = spdlog::default_logger();
        }
        if (logger) {
          logger->error(msg);
        }
        throw std::invalid_argument(msg);
      }
    }

    std::ostream& operator<<(std::ostream& os, const DFT& spec)
    {
      os << "Transform::DFT(length: " << spec.length << ")";
      return os;
    }

    // --- STFT ---
    STFT::STFT(
        size_t frame_len,
        size_t hop_len,
        size_t fft_len,
        OmniDSP::Window::Hann win_spec,
        bool center)
        : frame_length(frame_len),
          hop_length(hop_len),
          fft_length(fft_len),
          window_spec(win_spec),
          center_frames(center)
    {
      auto logger = spdlog::get("OmniDSP");
      if (!logger) {
        logger = spdlog::default_logger();
      }
      std::string msg;

      if (frame_length == 0) {
        msg = "Transform::STFT: frame_length must be positive.";
        if (logger) {
          logger->error(msg);
        }
        throw std::invalid_argument(msg);
      }
      if (hop_length == 0) {
        msg = "Transform::STFT: hop_length must be positive.";
        if (logger) {
          logger->error(msg);
        }
        throw std::invalid_argument(msg);
      }
      if (fft_length == 0) {
        msg = "Transform::STFT: fft_length must be positive.";
        if (logger) {
          logger->error(msg);
        }
        throw std::invalid_argument(msg);
      }
      if (fft_length < frame_length) {
        msg = "Transform::STFT: fft_length (" + std::to_string(fft_length)
              + ") should not be less than frame_length ("
              + std::to_string(frame_length) + ").";
        if (logger) {
          logger->warn(msg);
        }
      }
      // Window::Hann constructor handles its own validation if any.
    }

    std::ostream& operator<<(std::ostream& os, const STFT& spec)
    {
      os << "Transform::STFT(frame_length: " << spec.frame_length
         << ", hop_length: " << spec.hop_length
         << ", fft_length: " << spec.fft_length << ", window: "
         << spec.window_spec  // Assumes Window::Hann has operator<<
         << ", center_frames: " << (spec.center_frames ? "true" : "false")
         << ")";
      return os;
    }

    // --- DCT_Type ---
    std::ostream& operator<<(std::ostream& os, const DCT_Type& type)
    {
      switch (type) {
        case DCT_Type::DCT_I:
          os << "DCT_I";
          break;
        case DCT_Type::DCT_II:
          os << "DCT_II (Standard)";
          break;
        case DCT_Type::DCT_III:
          os << "DCT_III (Inverse of II)";
          break;
        case DCT_Type::DCT_IV:
          os << "DCT_IV";
          break;
        default:
          os << "Unknown DCT_Type";
          break;
      }
      return os;
    }

    // --- DCT ---
    DCT::DCT(size_t len, DCT_Type dct_type) : length(len), type(dct_type)
    {
      if (length == 0) {
        std::string msg = "Transform::DCT: length must be positive.";
        auto logger = spdlog::get("OmniDSP");
        if (!logger) {
          logger = spdlog::default_logger();
        }
        if (logger) {
          logger->error(msg);
        }
        throw std::invalid_argument(msg);
      }
    }

    std::ostream& operator<<(std::ostream& os, const DCT& spec)
    {
      os << "Transform::DCT(length: " << spec.length << ", type: " << spec.type
         << ")";
      return os;
    }

    // --- DST_Type ---
    std::ostream& operator<<(std::ostream& os, const DST_Type& type)
    {
      switch (type) {
        case DST_Type::DST_I:
          os << "DST_I";
          break;
        case DST_Type::DST_II:
          os << "DST_II (Standard)";
          break;
        case DST_Type::DST_III:
          os << "DST_III (Inverse of II)";
          break;
        case DST_Type::DST_IV:
          os << "DST_IV";
          break;
        default:
          os << "Unknown DST_Type";
          break;
      }
      return os;
    }

    // --- DST ---
    DST::DST(size_t len, DST_Type dst_type) : length(len), type(dst_type)
    {
      if (length == 0) {
        std::string msg = "Transform::DST: length must be positive.";
        auto logger = spdlog::get("OmniDSP");
        if (!logger) {
          logger = spdlog::default_logger();
        }
        if (logger) {
          logger->error(msg);
        }
        throw std::invalid_argument(msg);
      }
    }

    std::ostream& operator<<(std::ostream& os, const DST& spec)
    {
      os << "Transform::DST(length: " << spec.length << ", type: " << spec.type
         << ")";
      return os;
    }

    // --- DWT ---
    DWT::DWT(std::string name, int num_levels, std::string ext_mode)
        : wavelet_name(std::move(name)),
          levels(num_levels),
          mode(std::move(ext_mode))
    {
      auto logger = spdlog::get("OmniDSP");
      if (!logger) {
        logger = spdlog::default_logger();
      }

      if (wavelet_name.empty()) {
        std::string msg = "Transform::DWT: wavelet_name cannot be empty.";
        if (logger) {
          logger->error(msg);
        }
        throw std::invalid_argument(msg);
      }
      if (levels <= 0) {
        std::string msg = "Transform::DWT: levels (" + std::to_string(levels)
                          + ") must be positive.";
        if (logger) {
          logger->error(msg);
        }
        throw std::invalid_argument(msg);
      }
      if (mode != "symmetric" && mode != "periodic"
          && mode != "constant_extension" && mode != "zero") {
        std::string msg = "Transform::DWT: mode ('" + mode
                          + "') is not a recognized extension mode.";
        if (logger) {
          logger->warn(msg);
        }
      }
    }

    std::ostream& operator<<(std::ostream& os, const DWT& spec)
    {
      os << "Transform::DWT(wavelet: " << spec.wavelet_name
         << ", levels: " << spec.levels << ", mode: " << spec.mode << ")";
      return os;
    }

    // --- DHT ---
    DHT::DHT(size_t len) : length(len)
    {
      if (length == 0) {
        std::string msg = "Transform::DHT: length must be positive.";
        auto logger = spdlog::get("OmniDSP");
        if (!logger) {
          logger = spdlog::default_logger();
        }
        if (logger) {
          logger->error(msg);
        }
        throw std::invalid_argument(msg);
      }
    }

    std::ostream& operator<<(std::ostream& os, const DHT& spec)
    {
      os << "Transform::DHT(length: " << spec.length << ")";
      return os;
    }

  }  // namespace Transform
}  // namespace OmniDSP
