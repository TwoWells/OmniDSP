// tests/irfft.cpp
// Tests for inverse Complex-to-Real FFT functions (irfft, FFTPlan::execute_irfft)

#include <gtest/gtest.h>
#include <OmniFFT/omnifft.h> // Use correct include path
#include <vector>
#include <complex>
#include <limits>
#include <numeric> // For std::iota
#include <cmath>

// --- Test Fixture ---
class FFT_C2R_Inverse_Test : public ::testing::Test {
protected:
    // Using a power-of-2 size required by Accelerate REAL domain implementation
    const size_t N = 16;
    const size_t Nc = N / 2 + 1;
    const double double_tolerance = std::numeric_limits<double>::epsilon() * N * 10;
    const float float_tolerance = std::numeric_limits<float>::epsilon() * N * 10;

    // Helper to compare real vectors with tolerance
    template<typename T>
    void ExpectRealVectorNear(const std::vector<T>& expected,
                              const std::vector<T>& actual,
                              T tolerance, const std::string& msg = "") {
        ASSERT_EQ(expected.size(), actual.size()) << msg << " - Size mismatch";
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_NEAR(expected[i], actual[i], tolerance) << msg << " - Mismatch at index " << i;
        }
    }

    // Helper to generate a real test signal
    template<typename T>
    std::vector<T> generate_real_signal(size_t len) {
         std::vector<T> signal(len);
         // Simple ramp for testing
         for (size_t i = 0; i < len; ++i) {
             signal[i] = static_cast<T>(i);
         }
        // Or use std::iota: std::iota(signal.begin(), signal.end(), static_cast<T>(0));
        return signal;
    }
};

// --- Test Cases ---

TEST_F(FFT_C2R_Inverse_Test, Identity_BackwardNorm_Double) {
    using Real = double;
    using Complex = std::complex<Real>;
    std::vector<Real> original = generate_real_signal<Real>(N);
    std::vector<Complex> spectrum(Nc);
    std::vector<Real> reconstructed(N);

    try {
        // Use BACKWARD norm for both forward (rfft) and inverse (irfft)
        // Convenience functions use BACKWARD by default.
        OmniFFT::rfft(original, spectrum);
        OmniFFT::irfft(spectrum, reconstructed);
        // With BACKWARD norm, irfft(rfft(x)) should equal x
        // (assuming MKL backend, or correct internal scaling in Accelerate impl)
        ExpectRealVectorNear(original, reconstructed, double_tolerance, "rfft/irfft BACKWARD");

    } catch (const std::exception& e) {
        FAIL() << "FFT operation threw exception: " << e.what();
    }
}

TEST_F(FFT_C2R_Inverse_Test, Identity_OrthoNorm_Float) {
    using Real = float;
    using Complex = std::complex<Real>;
    std::vector<Real> original = generate_real_signal<Real>(N);
    std::vector<Complex> spectrum(Nc);
    std::vector<Real> reconstructed(N);
    const float tol = float_tolerance * 10; // Use slightly looser tolerance maybe

    try {
        // Use ORTHO norm via FFTPlan for both forward and inverse
        OmniFFT::FFTPlan<Real> plan_fwd(N, OmniFFT::Precision::SINGLE, OmniFFT::Direction::FORWARD, OmniFFT::Domain::REAL, OmniFFT::NormMode::ORTHO);
        OmniFFT::FFTPlan<Real> plan_inv(N, OmniFFT::Precision::SINGLE, OmniFFT::Direction::INVERSE, OmniFFT::Domain::REAL, OmniFFT::NormMode::ORTHO);

        ASSERT_EQ(plan_fwd.getComplexLength(), Nc);
        ASSERT_EQ(plan_inv.getLength(), N);

        plan_fwd.execute_rfft(original.data(), spectrum.data());
        plan_inv.execute_irfft(spectrum.data(), reconstructed.data());

        // With ORTHO norm, irfft(rfft(x)) should equal x
        ExpectRealVectorNear(original, reconstructed, tol, "rfft/irfft ORTHO");

    } catch (const std::exception& e) {
        FAIL() << "FFT operation threw exception: " << e.what();
    }
}

TEST_F(FFT_C2R_Inverse_Test, Identity_ForwardNorm_Double) {
    using Real = double;
    using Complex = std::complex<Real>;
    std::vector<Real> original = generate_real_signal<Real>(N);
    std::vector<Complex> spectrum(Nc);
    std::vector<Real> reconstructed(N);

    try {
        // Use FORWARD norm via FFTPlan for both forward and inverse
        OmniFFT::FFTPlan<Real> plan_fwd(N, OmniFFT::Precision::DOUBLE, OmniFFT::Direction::FORWARD, OmniFFT::Domain::REAL, OmniFFT::NormMode::FORWARD);
        OmniFFT::FFTPlan<Real> plan_inv(N, OmniFFT::Precision::DOUBLE, OmniFFT::Direction::INVERSE, OmniFFT::Domain::REAL, OmniFFT::NormMode::FORWARD);

        ASSERT_EQ(plan_fwd.getComplexLength(), Nc);
        ASSERT_EQ(plan_inv.getLength(), N);

        plan_fwd.execute_rfft(original.data(), spectrum.data());
        plan_inv.execute_irfft(spectrum.data(), reconstructed.data());

        // With FORWARD norm, irfft(rfft(x)) should equal x
        ExpectRealVectorNear(original, reconstructed, double_tolerance, "rfft/irfft FORWARD");

    } catch (const std::exception& e) {
        FAIL() << "FFT operation threw exception: " << e.what();
    }
}

TEST_F(FFT_C2R_Inverse_Test, KnownTransform_RealDC_Double) {
    using Real = double;
    using Complex = std::complex<Real>;
    // Input spectrum for a DC=1 signal (using BACKWARD norm from rfft)
    std::vector<Complex> spectrum(Nc, {0.0, 0.0});
    spectrum[0] = {static_cast<Real>(N), 0.0}; // DC component is N
    std::vector<Real> output;
    std::vector<Real> expected(N, 1.0); // Expected DC signal

    try {
        OmniFFT::irfft(spectrum, output);
        ASSERT_EQ(output.size(), N);
        ExpectRealVectorNear(expected, output, double_tolerance, "irfft(DC Spectrum)");
    } catch (const std::exception& e) {
        FAIL() << "IRFFT operation threw exception: " << e.what();
    }
}

// TODO: Add more tests:
// - Known spectrum for Cosine -> Real Cosine output
// - Different power-of-2 lengths
// - Edge case N=1
// - Error condition tests (non-power-of-2 for Accelerate, wrong domain/direction plan)
// - Test with input spectrum that does NOT have Hermitian symmetry? (Output should not be purely real) - Low priority