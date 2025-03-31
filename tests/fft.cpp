// tests/fft.cpp
// Tests for forward Complex-to-Complex FFT functions (fft, fft_inplace, FFTPlan::execute)

#include <gtest/gtest.h>
#include <OmniFFT/omnifft.h> // Use correct include path
#include <vector>
#include <complex>
#include <limits> // For numeric_limits
#include <cmath>

// --- Test Fixture ---
class FFT_C2C_Forward_Test : public ::testing::Test {
protected:
    // Using a power-of-2 size for compatibility with all backends/modes initially
    const size_t N = 16;
    const double double_tolerance = std::numeric_limits<double>::epsilon() * N * 10; // Looser tolerance for FFT
    const float float_tolerance = std::numeric_limits<float>::epsilon() * N * 10;

    // Helper to compare complex vectors with tolerance
    template<typename T>
    void ExpectComplexVectorNear(const std::vector<std::complex<T>>& expected,
                                 const std::vector<std::complex<T>>& actual,
                                 T tolerance, const std::string& msg = "") {
        ASSERT_EQ(expected.size(), actual.size()) << msg << " - Size mismatch";
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_NEAR(expected[i].real(), actual[i].real(), tolerance) << msg << " - Mismatch at index " << i << " (real)";
            EXPECT_NEAR(expected[i].imag(), actual[i].imag(), tolerance) << msg << " - Mismatch at index " << i << " (imag)";
        }
    }
};

// --- Test Cases ---

TEST_F(FFT_C2C_Forward_Test, KnownTransform_Delta_Double) {
    using Complex = std::complex<double>;
    std::vector<Complex> input(N, {0.0, 0.0});
    input[0] = {1.0, 0.0}; // Delta function at index 0
    std::vector<Complex> spectrum;
    std::vector<Complex> expected(N, {1.0, 0.0}); // FFT of delta is constant 1

    try {
        // Test convenience function (uses BACKWARD norm -> scale=1)
        OmniFFT::fft(input, spectrum);
        ExpectComplexVectorNear(expected, spectrum, double_tolerance, "fft(delta)");
    } catch (const std::exception& e) {
        FAIL() << "FFT operation threw exception: " << e.what();
    }
}

TEST_F(FFT_C2C_Forward_Test, KnownTransform_DC_Float) {
    using Complex = std::complex<float>;
    std::vector<Complex> input(N, {1.0f, 0.0f}); // DC signal
    std::vector<Complex> spectrum;
    std::vector<Complex> expected(N, {0.0f, 0.0f});
    expected[0] = {static_cast<float>(N), 0.0f}; // FFT of DC=1 is N at freq 0

    try {
        // Test convenience function (uses BACKWARD norm -> scale=1)
        OmniFFT::fft(input, spectrum);
        ExpectComplexVectorNear(expected, spectrum, float_tolerance, "fft(DC)");
    } catch (const std::exception& e) {
        FAIL() << "FFT operation threw exception: " << e.what();
    }
}

TEST_F(FFT_C2C_Forward_Test, InPlace_Double) {
    using Complex = std::complex<double>;
    std::vector<Complex> input(N);
    for (size_t i = 0; i < N; ++i) input[i] = Complex(i, -static_cast<double>(i) / 2.0);
    std::vector<Complex> input_copy = input; // Keep original for comparison later if needed
    std::vector<Complex> spectrum_oop;

    try {
        // Calculate expected OOP result first (using BACKWARD norm)
        OmniFFT::fft(input_copy, spectrum_oop);

        // Test in-place convenience function
        OmniFFT::fft_inplace(input); // Modifies input
        ExpectComplexVectorNear(spectrum_oop, input, double_tolerance, "fft_inplace");

    } catch (const std::exception& e) {
        FAIL() << "FFT operation threw exception: " << e.what();
    }
}

TEST_F(FFT_C2C_Forward_Test, PlanExecute_OrthoNorm_Double) {
    using Complex = std::complex<double>;
    std::vector<Complex> input(N, {0.0, 0.0});
    input[0] = {1.0, 0.0}; // Delta function
    std::vector<Complex> spectrum(N);
    // Expected for Ortho norm: FFT output is scaled by 1/sqrt(N)
    double scale = 1.0 / std::sqrt(static_cast<double>(N));
    std::vector<Complex> expected(N, {scale, 0.0});

    try {
        OmniFFT::FFTPlan<double> plan(N, OmniFFT::Precision::DOUBLE, OmniFFT::Direction::FORWARD, OmniFFT::Domain::COMPLEX, OmniFFT::NormMode::ORTHO);
        plan.execute(input.data(), spectrum.data());
        ExpectComplexVectorNear(expected, spectrum, double_tolerance, "FFTPlan ORTHO");
    } catch (const std::exception& e) {
        FAIL() << "FFT operation threw exception: " << e.what();
    }
}

TEST_F(FFT_C2C_Forward_Test, PlanExecute_ForwardNorm_Double) {
    using Complex = std::complex<double>;
    std::vector<Complex> input(N, {0.0, 0.0});
    input[0] = {1.0, 0.0}; // Delta function
    std::vector<Complex> spectrum(N);
    // Expected for Forward norm: FFT output is scaled by 1/N
    double scale = 1.0 / static_cast<double>(N);
    std::vector<Complex> expected(N, {scale, 0.0});

    try {
        OmniFFT::FFTPlan<double> plan(N, OmniFFT::Precision::DOUBLE, OmniFFT::Direction::FORWARD, OmniFFT::Domain::COMPLEX, OmniFFT::NormMode::FORWARD);
        plan.execute(input.data(), spectrum.data());
        ExpectComplexVectorNear(expected, spectrum, double_tolerance, "FFTPlan FORWARD");
    } catch (const std::exception& e) {
        FAIL() << "FFT operation threw exception: " << e.what();
    }
}

// TODO: Add more tests:
// - Different lengths (if supported beyond power-of-2 by backend)
// - Specific sine/cosine inputs
// - Error condition tests (e.g., wrong domain plan)