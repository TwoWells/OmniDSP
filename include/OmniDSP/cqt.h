#ifndef OMNIDSP_CQT_H
#define OMNIDSP_CQT_H

// --- Standard Includes ---
#include <cmath>
#include <complex>
#include <memory>     // Needed for std::unique_ptr
#include <stdexcept>  // For exceptions
#include <vector>

// --- Project Includes ---
#include "core_types.h"  // Include core types (Precision) - NOT omnidsp.h
#include "fft.h"         // Need fft.h for FFTNorm and FFTPlan

namespace OmniDSP {  // <<< Namespace opened for this header's content

// --- Forward Declarations ---
template <typename T>
struct CQTPlanImpl;

// --- CQTPlan Class ---
template <typename T>
class CQTPlan {
 public:
  // Constructor Declaration with namespace qualifier for FFTNorm:
  CQTPlan(double sr, double fmin, int n_bins, int bins_per_octave, Precision p,
          OmniDSP::FFTNorm norm);
  ~CQTPlan();
  CQTPlan(const CQTPlan&) = delete;
  CQTPlan& operator=(const CQTPlan&) = delete;
  CQTPlan(CQTPlan&&) noexcept;
  CQTPlan& operator=(CQTPlan&&) noexcept;
  void execute(const std::vector<std::complex<T>>& input,
               std::vector<std::complex<T>>& output);
  int getNumBins() const;
  double getMinFrequency() const;
  double getSamplingRate() const;

 private:
  std::unique_ptr<CQTPlanImpl<T>> pimpl_;
};

// --- Convenience Function Declaration ---
template <typename T>
void cqt(
    const std::vector<std::complex<T>>& input,
    std::vector<std::complex<T>>& output, double sr, double fmin, int n_bins,
    int bins_per_octave,
    OmniDSP::FFTNorm norm = OmniDSP::FFTNorm::Backward);  // Added OmniDSP::

}  // namespace OmniDSP

#endif  // OMNIDSP_CQT_H
