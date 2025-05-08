/**
 * @file tests/cpp/backend/stub.cpp
 * @brief Unit tests specifically for the Default backend
 * convolution/correlation behavior.
 */
#include <gtest/gtest.h>

#include <OmniDSP/convolution.hpp>  // Include the public API
#include <stdexcept>
#include <vector>

// Only compile these tests if the Default backend is active
#if !defined(USE_ACCELERATE) && !defined(USE_ONEMKL)

// Test fixture (can inherit if common setup is needed, but maybe not necessary
// here)
class BackendConvStubTest : public ::testing::Test {};

// Test specifically if stub backend throws (this is the test moved from the
// main file)
TEST_F(BackendConvStubTest, PublicApiThrowsError)
{
  std::vector<double> signal_d = {1.0, 2.0, 3.0};
  std::vector<double> kernel_d = {1.0, 0.5};
  // These call the public API which should dispatch to the throwing stub
  // backend impl
  EXPECT_THROW(OmniDSP::convolve1d(signal_d, kernel_d), std::runtime_error);
  EXPECT_THROW(OmniDSP::correlate1d(signal_d, kernel_d), std::runtime_error);

  std::vector<float> signal_f = {1.0f, 2.0f, 3.0f};
  std::vector<float> kernel_f = {1.0f, 0.5f};
  EXPECT_THROW(OmniDSP::convolve1d(signal_f, kernel_f), std::runtime_error);
  EXPECT_THROW(OmniDSP::correlate1d(signal_f, kernel_f), std::runtime_error);
}

#endif  // Stub backend test guard
