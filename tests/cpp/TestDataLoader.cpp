/**
 * @file TestDataLoader.cpp
 * @brief Implements utility functions to load test reference data from text
 * files.
 *
 * Uses std::filesystem for robust path handling and throws std::runtime_error
 * with concise messages on failure, suitable for catching in test cases.
 */
#include "TestDataLoader.h"  // Include the header file declaration

#include <complex>
#include <filesystem>  // For std::filesystem::path
#include <fstream>     // For std::ifstream
#include <iostream>    // For std::cerr (error output), std::cout (debug output)
#include <limits>      // For std::numeric_limits
#include <sstream>     // For std::stringstream, std::getline
#include <stdexcept>   // For std::runtime_error
#include <string>
#include <system_error>  // For std::errc with from_chars (if used)
#include <utility>       // For std::pair (though using struct now)
#include <vector>

// Check if the CMake variable is defined
#ifndef OMNIDSP_TEST_DATA_DIR
#error "OMNIDSP_TEST_DATA_DIR is not defined by CMake! Cannot locate test data."
#endif

namespace TestDataLoader {

  // Base path defined by CMake
  const std::filesystem::path test_data_base_path = OMNIDSP_TEST_DATA_DIR;

  // --- Helper Struct and Functions ---

  // Structure to hold parsed dimensions
  struct Dimensions {
    size_t rows = 0;
    size_t cols = 0;  // 0 or 1 indicates vector
    bool is_matrix = false;
  };

  // Helper to trim leading/trailing whitespace
  std::string trim(const std::string &str)
  {
    size_t first = str.find_first_not_of(" \t\n\r");
    if (std::string::npos == first)
      return "";  // Handle empty or all-whitespace string
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, (last - first + 1));
  }

  // Helper to parse dimension string like "1024" or "84x12"
  // Throws std::runtime_error on invalid format.
  Dimensions parseDimString(
      const std::string &dim_str_trimmed, const std::string &context_path)
  {
    Dimensions dims;
    // Check for empty string after potential trimming
    if (dim_str_trimmed.empty()) {
      throw std::runtime_error(
          "Empty dimension string found in header of '" + context_path + "'");
    }

    size_t x_pos = dim_str_trimmed.find('x');

    if (x_pos == std::string::npos) {
      // Vector format: "# <rows>"
      dims.is_matrix = false;
      dims.cols = 1;  // Treat vector as having 1 column conceptually
      try {
        // Check for non-digit characters before converting
        if (dim_str_trimmed.find_first_not_of("0123456789")
            != std::string::npos) {
          throw std::invalid_argument(
              "Non-digit characters found in vector dimension");
        }
        // Use stoull for potentially large sizes
        dims.rows = std::stoull(dim_str_trimmed);
      }
      catch (const std::exception &e) {
        throw std::runtime_error(
            "Invalid vector dimension format in header of '" + context_path
            + "': '" + dim_str_trimmed + "' (" + e.what() + ")");
      }
    }
    else {
      // Matrix format: "# <rows>x<columns>"
      dims.is_matrix = true;
      std::string rows_str = trim(dim_str_trimmed.substr(0, x_pos));
      std::string cols_str = trim(dim_str_trimmed.substr(x_pos + 1));
      try {
        if (rows_str.empty()
            || rows_str.find_first_not_of("0123456789") != std::string::npos) {
          throw std::invalid_argument("Invalid character or empty rows string");
        }
        if (cols_str.empty()
            || cols_str.find_first_not_of("0123456789") != std::string::npos) {
          throw std::invalid_argument("Invalid character or empty cols string");
        }
        dims.rows = std::stoull(rows_str);
        dims.cols = std::stoull(cols_str);
      }
      catch (const std::exception &e) {
        throw std::runtime_error(
            "Invalid matrix dimension format in header of '" + context_path
            + "': '" + dim_str_trimmed + "' (" + e.what() + ")");
      }
      if (dims.rows == 0 || dims.cols == 0) {
        // Allow empty matrix definition if rows or cols is 0
        if (dims.rows == 0 && dims.cols == 0) {
          // Both 0 is okay for empty matrix
        }
        else if (dims.rows == 0 || dims.cols == 0) {
          // One is 0, the other isn't - this is ambiguous, treat as error for
          // now
          throw std::runtime_error(
              "Matrix dimensions cannot be partially zero (one zero, one "
              "non-zero) in header of '"
              + context_path + "'");
        }
      }
    }
    // Allow empty vector only if explicitly 0 rows were parsed successfully
    if (dims.rows == 0 && dims.is_matrix && dims.cols != 0) {
      throw std::runtime_error(
          "Matrix cannot have zero rows but non-zero columns in header of '"
          + context_path + "'");
    }

    return dims;
  }

  // Helper function to parse the header line and return dimensions
  // Throws std::runtime_error on failure.
  Dimensions parseHeader(std::ifstream &infile, const std::string &context_path)
  {
    std::string header_line;
    if (!std::getline(infile, header_line)) {
      // Check if the file is completely empty or just couldn't read the line
      if (infile.eof() && infile.gcount() == 0) {
        throw std::runtime_error(
            "Empty file or missing header line in file: '" + context_path
            + "'");
      }
      else {
        throw std::runtime_error(
            "Failed to read header line from file: '" + context_path + "'");
      }
    }

    std::string trimmed_header = trim(header_line);

    if (trimmed_header.empty() || trimmed_header[0] != '#') {
      throw std::runtime_error(
          "Invalid or missing header line (must start with '#') in file: '"
          + context_path + "'");
    }

    // Remove '#' and trim again
    std::string dim_part = trim(trimmed_header.substr(1));

    return parseDimString(dim_part, context_path);
  }

  // --- Public Functions ---

  std::string getTestDataPath(
      const std::string &suite_name, const std::string &filename)
  {
    if (test_data_base_path.empty()) {
      throw std::runtime_error(
          "OMNIDSP_TEST_DATA_DIR CMake variable is empty or invalid.");
    }
    // Use std::filesystem::path's operator/ for joining paths robustly
    std::filesystem::path full_path
        = test_data_base_path / suite_name / filename;
    // Return the path as a standard string
    return full_path.string();
  }

  // --- loadVectorData Specializations ---

  template <> std::vector<double> loadVectorData<double>(
      const std::string &suite_name, const std::string &filename)
  {
    std::string full_path_str = getTestDataPath(suite_name, filename);
    std::ifstream infile(full_path_str);
    if (!infile.is_open()) {
      // Throw concise error on open failure
      throw std::runtime_error(
          "Failed to open test data file: '" + full_path_str + "'");
    }

    Dimensions dims;
    try {
      dims = parseHeader(infile, full_path_str);
    }
    catch (const std::runtime_error &e) {
      throw std::runtime_error(
          "Error parsing header in file '" + full_path_str + "': " + e.what());
    }

    if (dims.is_matrix) {
      throw std::runtime_error(
          "Expected vector format ('# rows') but found matrix format ('# "
          "rowsxcols') in header of '"
          + full_path_str + "'");
    }

    std::vector<double> data;
    if (dims.rows == 0) {
      // std::cout << "[TestDataLoader] Loaded 0 doubles (empty vector) from "
      // << filename << std::endl;
      return data;  // Return empty vector
    }
    data.reserve(dims.rows);
    double value;
    size_t count = 0;
    while (count < dims.rows && (infile >> value)) {
      data.push_back(value);
      count++;
    }

    if (count != dims.rows) {
      std::stringstream ss;
      ss << "Data size mismatch in '" << full_path_str << "': Header specified "
         << dims.rows << " rows, but read " << count << ".";
      if (infile.fail() && !infile.eof()) {
        ss << " Read error likely occurred (e.g., non-numeric data).";
      }
      else if (!infile.eof()) {
        ss << " Unexpected extra data found after reading expected rows.";
      }
      else {
        ss << " Reached end of file prematurely.";
      }
      throw std::runtime_error(ss.str());
    }
    // Optional: Check for extra non-whitespace data after reading expected
    // count
    std::string remaining;
    if (infile >> remaining && !remaining.empty()) {
      std::cerr << "[TestDataLoader WARNING] Extra data ('" << remaining
                << "...') found after reading expected " << dims.rows
                << " doubles from '" << full_path_str << "'" << std::endl;
    }

    // std::cout << "[TestDataLoader] Loaded " << data.size() << " doubles from
    // "
    // << filename << std::endl;
    return data;
  }

  template <> std::vector<float> loadVectorData<float>(
      const std::string &suite_name, const std::string &filename)
  {
    std::string full_path_str = getTestDataPath(suite_name, filename);
    std::ifstream infile(full_path_str);
    if (!infile.is_open()) {
      throw std::runtime_error(
          "Failed to open test data file: '" + full_path_str + "'");
    }

    Dimensions dims;
    try {
      dims = parseHeader(infile, full_path_str);
    }
    catch (const std::runtime_error &e) {
      throw std::runtime_error(
          "Error parsing header in file '" + full_path_str + "': " + e.what());
    }

    if (dims.is_matrix) {
      throw std::runtime_error(
          "Expected vector format ('# rows') but found matrix format ('# "
          "rowsxcols') in header of '"
          + full_path_str + "'");
    }

    std::vector<float> data;
    if (dims.rows == 0) {
      // std::cout << "[TestDataLoader] Loaded 0 floats (empty vector) from " <<
      // filename << std::endl;
      return data;
    }
    data.reserve(dims.rows);
    float value;
    size_t count = 0;
    while (count < dims.rows && (infile >> value)) {
      data.push_back(value);
      count++;
    }

    if (count != dims.rows) {
      std::stringstream ss;
      ss << "Data size mismatch in '" << full_path_str << "': Header specified "
         << dims.rows << " rows, but read " << count << ".";
      if (infile.fail() && !infile.eof()) {
        ss << " Read error likely occurred (e.g., non-numeric data).";
      }
      else if (!infile.eof()) {
        ss << " Unexpected extra data found after reading expected rows.";
      }
      else {
        ss << " Reached end of file prematurely.";
      }
      throw std::runtime_error(ss.str());
    }
    std::string remaining;
    if (infile >> remaining && !remaining.empty()) {
      std::cerr << "[TestDataLoader WARNING] Extra data ('" << remaining
                << "...') found after reading expected " << dims.rows
                << " floats from '" << full_path_str << "'" << std::endl;
    }

    // std::cout << "[TestDataLoader] Loaded " << data.size() << " floats from "
    // << filename << std::endl;
    return data;
  }

  template <>
  std::vector<std::complex<double>> loadVectorData<std::complex<double>>(
      const std::string &suite_name, const std::string &filename)
  {
    std::string full_path_str = getTestDataPath(suite_name, filename);
    std::ifstream infile(full_path_str);
    if (!infile.is_open()) {
      throw std::runtime_error(
          "Failed to open test data file: '" + full_path_str + "'");
    }

    Dimensions dims;
    try {
      dims = parseHeader(infile, full_path_str);
    }
    catch (const std::runtime_error &e) {
      throw std::runtime_error(
          "Error parsing header in file '" + full_path_str + "': " + e.what());
    }

    if (dims.is_matrix) {
      throw std::runtime_error(
          "Expected vector format ('# rows') but found matrix format ('# "
          "rowsxcols') in header of '"
          + full_path_str + "'");
    }

    std::vector<std::complex<double>> data;
    if (dims.rows == 0) {
      // std::cout << "[TestDataLoader] Loaded 0 complex doubles (empty vector)
      // from " << filename << std::endl;
      return data;
    }
    data.reserve(dims.rows);
    double real_part, imag_part;
    size_t count = 0;
    while (count < dims.rows && (infile >> real_part >> imag_part)) {
      data.emplace_back(real_part, imag_part);
      count++;
    }

    if (count != dims.rows) {
      std::stringstream ss;
      ss << "Data size mismatch in '" << full_path_str << "': Header specified "
         << dims.rows << " complex pairs, but read " << count << ".";
      if (infile.fail() && !infile.eof()) {
        infile.clear();  // Clear fail bit to check for remaining non-whitespace
                         // chars
        std::string remaining;
        if (infile >> remaining && !remaining.empty()) {
          ss << " Read error likely occurred (expected pairs).";
        }  // else likely just whitespace or EOF
      }
      else if (!infile.eof()) {
        ss << " Unexpected extra data found after reading expected pairs.";
      }
      else {
        ss << " Reached end of file prematurely (expected pairs).";
      }
      throw std::runtime_error(ss.str());
    }
    // Check for extra data
    std::string remaining;
    if (infile >> remaining && !remaining.empty()) {
      std::cerr << "[TestDataLoader WARNING] Extra data ('" << remaining
                << "...') found after reading expected " << dims.rows
                << " complex doubles from '" << full_path_str << "'"
                << std::endl;
    }

    // std::cout << "[TestDataLoader] Loaded " << data.size() << " complex
    // doubles from " << filename << std::endl;
    return data;
  }

  template <>
  std::vector<std::complex<float>> loadVectorData<std::complex<float>>(
      const std::string &suite_name, const std::string &filename)
  {
    std::string full_path_str = getTestDataPath(suite_name, filename);
    std::ifstream infile(full_path_str);
    if (!infile.is_open()) {
      throw std::runtime_error(
          "Failed to open test data file: '" + full_path_str + "'");
    }

    Dimensions dims;
    try {
      dims = parseHeader(infile, full_path_str);
    }
    catch (const std::runtime_error &e) {
      throw std::runtime_error(
          "Error parsing header in file '" + full_path_str + "': " + e.what());
    }

    if (dims.is_matrix) {
      throw std::runtime_error(
          "Expected vector format ('# rows') but found matrix format ('# "
          "rowsxcols') in header of '"
          + full_path_str + "'");
    }

    std::vector<std::complex<float>> data;
    if (dims.rows == 0) {
      // std::cout << "[TestDataLoader] Loaded 0 complex floats (empty vector)
      // from " << filename << std::endl;
      return data;
    }
    data.reserve(dims.rows);
    float real_part, imag_part;
    size_t count = 0;
    while (count < dims.rows && (infile >> real_part >> imag_part)) {
      data.emplace_back(real_part, imag_part);
      count++;
    }

    if (count != dims.rows) {
      std::stringstream ss;
      ss << "Data size mismatch in '" << full_path_str << "': Header specified "
         << dims.rows << " complex pairs, but read " << count << ".";
      if (infile.fail() && !infile.eof()) {
        infile.clear();
        std::string remaining;
        if (infile >> remaining && !remaining.empty()) {
          ss << " Read error likely occurred (expected pairs).";
        }
      }
      else if (!infile.eof()) {
        ss << " Unexpected extra data found after reading expected pairs.";
      }
      else {
        ss << " Reached end of file prematurely (expected pairs).";
      }
      throw std::runtime_error(ss.str());
    }
    std::string remaining;
    if (infile >> remaining && !remaining.empty()) {
      std::cerr << "[TestDataLoader WARNING] Extra data ('" << remaining
                << "...') found after reading expected " << dims.rows
                << " complex floats from '" << full_path_str << "'"
                << std::endl;
    }

    // std::cout << "[TestDataLoader] Loaded " << data.size() << " complex
    // floats from " << filename << std::endl;
    return data;
  }

  // --- loadComplexMatrixData Specializations ---

  // Common implementation function for loading complex matrix data
  template <typename T>
  std::vector<std::vector<std::complex<T>>> loadComplexMatrixDataImpl(
      const std::string &suite_name, const std::string &filename)
  {
    std::string full_path_str = getTestDataPath(suite_name, filename);
    std::ifstream infile(full_path_str);
    if (!infile.is_open()) {
      throw std::runtime_error(
          "Failed to open test data file: '" + full_path_str + "'");
    }

    Dimensions dims;
    try {
      dims = parseHeader(infile, full_path_str);
    }
    catch (const std::runtime_error &e) {
      throw std::runtime_error(
          "Error parsing header in file '" + full_path_str + "': " + e.what());
    }

    if (!dims.is_matrix) {
      throw std::runtime_error(
          "Expected matrix format ('# rowsxcols') but found vector format ('# "
          "rows') in header of '"
          + full_path_str + "'");
    }

    std::vector<std::vector<std::complex<T>>> matrix_data;
    // Handle empty matrix case defined by header "# 0x0" or "# 0xN" or "# Nx0"
    if (dims.rows == 0 || dims.cols == 0) {
      // std::cout << "[TestDataLoader] Loaded 0x0 complex matrix (empty) from "
      // << filename << std::endl;
      return matrix_data;  // Return empty matrix
    }
    matrix_data.reserve(dims.rows);

    std::vector<std::complex<T>> linear_data;  // Load linearly first
    const size_t total_elements = dims.rows * dims.cols;
    linear_data.reserve(total_elements);

    T real_part, imag_part;
    size_t count = 0;

    // Read all complex pairs linearly using whitespace separation
    while (count < total_elements && (infile >> real_part >> imag_part)) {
      linear_data.emplace_back(real_part, imag_part);
      count++;
    }

    // Validation after reading
    if (count != total_elements) {
      std::stringstream ss;
      ss << "Data size mismatch in '" << full_path_str << "': Header specified "
         << dims.rows << "x" << dims.cols << " (" << total_elements
         << " complex elements), but read " << count << ".";
      if (infile.fail() && !infile.eof()) {
        infile.clear();
        std::string remaining;
        if (infile >> remaining && !remaining.empty()) {
          ss << " Read error likely occurred (expected pairs).";
        }
      }
      else if (!infile.eof()) {
        ss << " Unexpected extra data found after reading expected pairs.";
      }
      else {
        ss << " Reached end of file prematurely (expected pairs).";
      }
      throw std::runtime_error(ss.str());
    }
    std::string remaining;
    if (infile >> remaining && !remaining.empty()) {
      std::cerr << "[TestDataLoader WARNING] Extra data ('" << remaining
                << "...') found after reading expected " << total_elements
                << " complex numbers from '" << full_path_str << "'"
                << std::endl;
    }

    // Reshape the linear data into the matrix
    size_t linear_idx = 0;
    for (size_t r = 0; r < dims.rows; ++r) {
      std::vector<std::complex<T>> row;
      row.reserve(dims.cols);
      auto start_iter = linear_data.begin() + r * dims.cols;
      auto end_iter = start_iter + dims.cols;
      row.assign(start_iter, end_iter);
      matrix_data.push_back(std::move(row));
    }

    // std::cout << "[TestDataLoader] Loaded " << matrix_data.size() << "x"
    //           << (matrix_data.empty() ? 0 : matrix_data[0].size())
    //           << " complex matrix from " << filename << std::endl;
    return matrix_data;
  }

  template <>
  std::vector<std::vector<std::complex<double>>> loadComplexMatrixData<double>(
      const std::string &suite_name, const std::string &filename)
  {
    return loadComplexMatrixDataImpl<double>(suite_name, filename);
  }

  template <>
  std::vector<std::vector<std::complex<float>>> loadComplexMatrixData<float>(
      const std::string &suite_name, const std::string &filename)
  {
    return loadComplexMatrixDataImpl<float>(suite_name, filename);
  }

}  // namespace TestDataLoader
