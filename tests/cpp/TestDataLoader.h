#ifndef TEST_DATA_LOADER_H
#define TEST_DATA_LOADER_H

#include <complex>    // For std::complex
#include <stdexcept>  // For std::runtime_error
#include <string>     // For std::string
#include <vector>     // For std::vector

/**
 * @brief Provides utility functions to load test reference data from text
 * files.
 *
 * Assumes data files are organized under a base directory specified by CMake
 * (OMNIDSP_TEST_DATA_DIR), then grouped by suite name.
 * Filenames within suite directories should follow the convention:
 * <TestCaseName>_<Purpose>_<TypeSuffix>.txt
 */
namespace TestDataLoader {

/**
 * @brief Constructs the full path to a test data file.
 * @param suite_name The name of the test suite (e.g., "fft", "window"),
 * corresponding to a subdirectory.
 * @param filename The specific data filename (e.g., "MyTest_input_d.txt").
 * @return The full path to the data file.
 */
std::string getTestDataPath(const std::string &suite_name,
                            const std::string &filename);

/**
 * @brief Loads a 1D vector of data from a text file.
 *
 * Handles real types (double, float) assuming one value per line/whitespace
 * separated. Handles complex types (complex<double>, complex<float>) assuming
 * "real imag" pairs per line/whitespace separated.
 *
 * @tparam T The data type to load (double, float, std::complex<double>,
 * std::complex<float>).
 * @param suite_name The name of the test suite subdirectory.
 * @param filename The specific data filename.
 * @return std::vector<T> The loaded data.
 * @throws std::runtime_error If the file cannot be opened or a read error
 * occurs.
 */
template <typename T>
std::vector<T> loadVectorData(const std::string &suite_name,
                              const std::string &filename);

/**
 * @brief Loads a 2D matrix (vector of vectors) of complex data from a text
 * file.
 *
 * Assumes each line in the file represents a row.
 * Within each row, complex numbers are represented as "real imag" pairs,
 * separated by a specific delimiter (e.g., tab '\t').
 *
 * @tparam T The underlying floating point type (double or float).
 * @param suite_name The name of the test suite subdirectory.
 * @param filename The specific data filename (e.g.,
 * "FilterBank_expected_cd.txt").
 * @return std::vector<std::vector<std::complex<T>>> The loaded 2D data.
 * @throws std::runtime_error If the file cannot be opened or a parsing error
 * occurs.
 */
template <typename T>
std::vector<std::vector<std::complex<T>>> loadComplexMatrixData(
    const std::string &suite_name, const std::string &filename);

// --- Explicit Template Specialization Declarations ---
// Ensures these types are supported and can be defined in the .cpp file.

template <>
std::vector<double> loadVectorData<double>(const std::string &suite_name,
                                           const std::string &filename);
template <>
std::vector<float> loadVectorData<float>(const std::string &suite_name,
                                         const std::string &filename);
template <>
std::vector<std::complex<double>> loadVectorData<std::complex<double>>(
    const std::string &suite_name, const std::string &filename);
template <>
std::vector<std::complex<float>> loadVectorData<std::complex<float>>(
    const std::string &suite_name, const std::string &filename);

template <>
std::vector<std::vector<std::complex<double>>> loadComplexMatrixData<double>(
    const std::string &suite_name, const std::string &filename);
template <>
std::vector<std::vector<std::complex<float>>> loadComplexMatrixData<float>(
    const std::string &suite_name, const std::string &filename);

}  // namespace TestDataLoader

#endif  // TEST_DATA_LOADER_H
