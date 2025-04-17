/**
 * @file tests/cpp/backend/onemkl.cpp
 * @brief Unit tests specifically for the MKL backend convolution/correlation
 * (if any).
 */
#include <OmniDSP/convolution.h>
#include <gtest/gtest.h>

#include <vector>

// Only compile these tests if the MKL backend is active
#ifdef USE_ONEMKL

// Add any tests here that are specific to validating MKL behavior,
// beyond the general correctness tests in backend_conv_test.cpp.
// For example, performance benchmarks or tests with MKL-specific edge cases.

// Example Placeholder Test (can be removed if no MKL-specific tests yet)
TEST(BackendConvMKLTest, Placeholder) {
  SUCCEED() << "No MKL-specific convolution tests yet.";
}

#endif  // USE_ONEMKL
