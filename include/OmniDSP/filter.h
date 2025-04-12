// Example: include/OmniDSP/filter.h
#ifndef OMNIDSP_FILTER_H
#define OMNIDSP_FILTER_H

#include <vector>

namespace OmniDSP {

/**
 * @brief Applies a low-pass FIR filter.
 * @param signal Input signal.
 * @param coefficients Filter coefficients.
 * @return Filtered signal.
 * @throws std::runtime_error // Or other appropriate exception on error.
 */
template <typename T>
std::vector<T> applyFirFilter(const std::vector<T>& signal, const std::vector<T>& coefficients);

// You might also add functions for designing coefficients here,
// or create a filter class to manage state and coefficients.

} // namespace OmniDSP

#endif // OMNIDSP_FILTER_H