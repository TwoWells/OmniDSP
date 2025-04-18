#ifndef OMNIDSP_TESTS_CPP_TEST_DATA_UTILS_H
#define OMNIDSP_TESTS_CPP_TEST_DATA_UTILS_H

#include <complex>
#include <stdexcept>  // For std::runtime_error
#include <string>
#include <vector>

namespace TestDataUtils {

/**
 * @brief Provides access to test reference data loaded from
 * test_references.txt.
 *
 * Data is loaded implicitly from the file specified by the
 * OMNIDSP_TEST_REF_FILE_PATH preprocessor definition upon the first call
 * to any getter function. Subsequent calls return cached data.
 *
 * All getter functions throw std::runtime_error if the reference file cannot be
 * found/read, or if the requested data key does not exist in the loaded data.
 */

/**
 * @brief Gets the expected double vector associated with the given key.
 * @param key The identifier string from the reference file (e.g.,
 * "WINDOW_HANN_N5_D").
 * @return Const reference to the loaded data vector.
 * @throws std::runtime_error If the key is not found or data cannot be loaded.
 */
const std::vector<double>& getExpectedDoubleVec(const std::string& key);

/**
 * @brief Gets the expected float vector associated with the given key.
 * @param key The identifier string from the reference file (e.g.,
 * "WINDOW_HANN_N5_F").
 * @return Const reference to the loaded data vector.
 * @throws std::runtime_error If the key is not found or data cannot be loaded.
 */
const std::vector<float>& getExpectedFloatVec(const std::string& key);

/**
 * @brief Gets the expected complex double vector associated with the given key.
 * Used primarily for FFT input/output reference data.
 * @param key The identifier string from the reference file (e.g.,
 * "FFT_INPUT_COMPLEX_D").
 * @return Const reference to the loaded data vector.
 * @throws std::runtime_error If the key is not found or data cannot be loaded.
 */
const std::vector<std::complex<double>>& getExpectedComplexDoubleVec(
    const std::string& key);

/**
 * @brief Gets the expected complex float vector associated with the given key.
 * Used primarily for FFT input/output reference data.
 * @param key The identifier string from the reference file (e.g.,
 * "FFT_INPUT_COMPLEX_F").
 * @return Const reference to the loaded data vector.
 * @throws std::runtime_error If the key is not found or data cannot be loaded.
 */
const std::vector<std::complex<float>>& getExpectedComplexFloatVec(
    const std::string& key);

/**
 * @brief Gets the expected complex double matrix for CQT comparison (Librosa).
 * Corresponds to the "librosa_cqt_complex_d" key in the reference file.
 * @return Const reference to the loaded data matrix.
 * @throws std::runtime_error If the CQT data is not found or cannot be loaded.
 */
const std::vector<std::vector<std::complex<double>>>& getExpectedCQTD();

/**
 * @brief Gets the expected complex float matrix for CQT comparison (Librosa).
 * Corresponds to the "librosa_cqt_complex_f" key in the reference file.
 * @return Const reference to the loaded data matrix.
 * @throws std::runtime_error If the CQT data is not found or cannot be loaded.
 */
const std::vector<std::vector<std::complex<float>>>& getExpectedCQTF();

// Add more getters here if other specific data types/structures are needed.

}  // namespace TestDataUtils

#endif  // OMNIDSP_TESTS_CPP_TEST_DATA_UTILS_H
