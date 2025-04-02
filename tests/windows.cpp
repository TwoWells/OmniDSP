#include <gtest/gtest.h>
#include <OmniDSP/omnidsp.h> // Include OmniDSP header
#include <OmniDSP/windows.h> // Include Window header
#include <vector>
#include <numeric>  // std::iota
#include <cmath>    // cos, M_PI
#include <limits>   // numeric_limits

// --- Test Fixture ---
class FFT_Window_Test : public ::testing::Test {
protected:
    // Helper function to generate a simple test signal
    template <typename T>
    std::vector<T> generate_test_signal(size_t length) {
        std::vector<T> signal(length);
        if constexpr (std::is_same_v<T, float>) {
            std::iota(signal.begin(), signal.end(), 0.0f); // Use 0.0f for float
        } else {
            std::iota(signal.begin(), signal.end(), 0);
        }
        return signal;
    }

    // Helper function to compare vectors with a tolerance
    template <typename T>
    void ExpectVectorNear(const std::vector<T>& expected,
                          const std::vector<T>& actual,
                          T tolerance,
                          const std::string& msg = "") {
        ASSERT_EQ(expected.size(), actual.size()) << msg << " - Size mismatch";
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_NEAR(expected[i], actual[i], tolerance) << msg
                << " - Mismatch at index " << i;
        }
    }

    // Define a reasonable tolerance based on the data type
    template <typename T>
    T get_tolerance() {
        return std::numeric_limits<T>::epsilon() * 100;
    }
};

// --- Test Cases ---

TEST_F(FFT_Window_Test, HannWindow_Double) {
    using T = double;
    size_t N = 10;
    std::vector<T> input = generate_test_signal<T>(N);
    std::vector<T> windowed = OmniDSP::Window::hann(input);
    std::vector<T> expected(N);
    T tolerance = get_tolerance<T>();

    for (size_t n = 0; n < N; ++n) {
        expected[n] = input[n] * (0.5 - 0.5 * cos(2.0 * M_PI * n / (N - 1)));
    }

    ExpectVectorNear(expected, windowed, tolerance, "Hann Window (double)");
}

TEST_F(FFT_Window_Test, HammingWindow_Float) {
    using T = float;
    size_t N = 10;
    std::vector<T> input = generate_test_signal<T>(N);
    std::vector<T> windowed = OmniDSP::Window::hamming(input);
    std::vector<T> expected(N);
    T tolerance = get_tolerance<T>();

    for (size_t n = 0; n < N; ++n) {
        expected[n] = input[n] * (0.54f - 0.46f * cos(2.0f * M_PI * n / (N - 1)));
    }

    ExpectVectorNear(expected, windowed, tolerance, "Hamming Window (float)");
}

TEST_F(FFT_Window_Test, KaiserWindow_Double) {
    using T = double;
    size_t N = 10;
    T beta = 8.6;
    std::vector<T> input = generate_test_signal<T>(N);
    std::vector<T> windowed = OmniDSP::Window::kaiser(input, beta);
    std::vector<T> expected(N);
    T tolerance = get_tolerance<T>();

    for (size_t n = 0; n < N; ++n) {
        T factor = (n * 2.0) / (N - 1) - 1.0;
        T sqrt_term = sqrt(1.0 - factor * factor);
        expected[n] = input[n] * std::cyl_bessel_i(0, beta * sqrt_term) / std::cyl_bessel_i(0, beta);
    }

    ExpectVectorNear(expected, windowed, tolerance, "Kaiser Window (double)");
}

TEST_F(FFT_Window_Test, FlattopWindow_Float) {
    using T = float;
    size_t N = 10;
    std::vector<T> input = generate_test_signal<T>(N);
    std::vector<T> windowed = OmniDSP::Window::flattop(input);
    std::vector<T> expected(N);
    T tolerance = get_tolerance<T>();

    constexpr T a0 = 0.21557895f;
    constexpr T a1 = 0.41663158f;
    constexpr T a2 = 0.277263158f;
    constexpr T a3 = 0.083578947f;
    constexpr T a4 = 0.006947368f;

    for (size_t n = 0; n < N; ++n) {
        expected[n] = input[n] * (a0 - a1 * cos(2.0f * M_PI * n / (N - 1)) +
                                   a2 * cos(4.0f * M_PI * n / (N - 1)) -
                                   a3 * cos(6.0f * M_PI * n / (N - 1)) +
                                   a4 * cos(8.0f * M_PI * n / (N - 1)));
    }

    ExpectVectorNear(expected, windowed, tolerance, "Flattop Window (float)");
}

TEST_F(FFT_Window_Test, EmptyInput_ThrowsException) {
    using T = double;
    std::vector<T> input; // Empty vector

    // Test that each window function throws an exception for empty input
    ASSERT_THROW(OmniDSP::Window::hann(input), std::invalid_argument);
    ASSERT_THROW(OmniDSP::Window::hamming(input), std::invalid_argument);
    ASSERT_THROW(OmniDSP::Window::kaiser(input, 8.6), std::invalid_argument);
    ASSERT_THROW(OmniDSP::Window::flattop(input), std::invalid_argument);
}