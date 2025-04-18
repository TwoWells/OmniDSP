#include "test_data_utils.h"

#include <complex>
#include <cstdlib>    // For std::stod, std::stof
#include <exception>  // For std::exception
#include <fstream>    // For std::ifstream
#include <iostream>   // For std::cerr (optional error logging)
#include <map>
#include <mutex>    // For std::call_once, std::once_flag
#include <sstream>  // For std::stringstream
#include <string>
#include <vector>

// Preprocessor check for the reference file path defined by CMake
#ifndef OMNIDSP_TEST_REF_FILE_PATH
#error \
    "OMNIDSP_TEST_REF_FILE_PATH is not defined. Ensure CMakeLists.txt defines this path."
#endif

namespace TestDataUtils {

// Anonymous namespace for internal state (static variables)
namespace {
// Static storage for the loaded data
std::map<std::string, std::vector<double>> data_d;
std::map<std::string, std::vector<float>> data_f;
// NEW: Static storage for complex vectors
std::map<std::string, std::vector<std::complex<double>>> data_cd;
std::map<std::string, std::vector<std::complex<float>>> data_cf;
// Static storage for CQT matrices (remains the same)
std::vector<std::vector<std::complex<double>>> cqt_data_d;
std::vector<std::vector<std::complex<float>>> cqt_data_f;

// Flag and mutex for thread-safe lazy loading
std::once_flag load_flag;
bool load_successful = false;
std::string load_error_message = "";

// Helper to trim whitespace from start and end of string
std::string trim(const std::string& str) {
  size_t first = str.find_first_not_of(" \t\n\r");
  if (std::string::npos == first) {
    return str;
  }
  size_t last = str.find_last_not_of(" \t\n\r");
  return str.substr(first, (last - first + 1));
}

// Internal helper function to load data (called only once)
void loadDataFromFile() {
  const std::string filename = OMNIDSP_TEST_REF_FILE_PATH;
  std::ifstream infile(filename);

  if (!infile.is_open()) {
    load_error_message = "Failed to open reference data file. Path used: '" +
                         filename +
                         "'. Check CMake definition and file copy command.";
    load_successful = false;
    return;  // Exit early if file cannot be opened
  }

  std::string line, current_key;
  bool reading_data = false;
  bool is_complex_1d = false;  // NEW flag for 1D complex vectors
  bool is_complex_2d = false;  // Flag for 2D complex matrices (CQT)
  bool is_float = false;       // Track if current key implies float data
  size_t expected_rows = 0, expected_cols = 0;
  std::vector<double>
      flat_data_buffer;  // Use double buffer for all numeric parsing

  try {
    while (std::getline(infile, line)) {
      line = trim(line);  // Trim whitespace

      // Skip empty lines and simple comments
      if (line.empty() ||
          (line[0] == '#' && line.find("# START") != 0 &&
           line.find("# END") != 0 && line.find("# SHAPE") != 0)) {
        continue;
      }

      if (line.rfind("# START ", 0) == 0) {
        if (reading_data) {
          throw std::runtime_error(
              "Found new START marker before END marker for key: " +
              current_key);
        }
        current_key = trim(line.substr(8));
        reading_data = true;
        // Determine data type based on key
        is_complex_2d = (current_key.find("cqt_complex") != std::string::npos);
        // Assume complex if key contains COMPLEX or FFT/RFFT/IFFT and isn't 2D
        // CQT
        is_complex_1d =
            !is_complex_2d &&
            (current_key.find("COMPLEX") != std::string::npos ||
             current_key.find("FFT_EXPECTED") != std::string::npos ||
             current_key.find("RFFT_EXPECTED") != std::string::npos ||
             current_key.find("IFFT_EXPECTED") != std::string::npos);
        is_float = (current_key.find("_F") != std::string::npos);

        expected_rows = 0;
        expected_cols = 0;
        flat_data_buffer.clear();

      } else if (line.rfind("# END ", 0) == 0) {
        if (!reading_data) {
          throw std::runtime_error(
              "Found END marker without active START marker.");
        }
        std::string end_key = trim(line.substr(6));
        if (end_key != current_key) {
          throw std::runtime_error("Mismatched END marker. Expected: " +
                                   current_key + ", Got: " + end_key);
        }

        // --- Process the collected data buffer ---
        if (is_complex_2d) {  // Process 2D Complex Matrix (CQT)
          if (expected_rows == 0 || expected_cols == 0) {
            throw std::runtime_error(
                "Missing or invalid # SHAPE info for key: " + current_key);
          }
          size_t expected_parts = expected_rows * expected_cols * 2;
          if (flat_data_buffer.size() != expected_parts) {
            throw std::runtime_error(
                "Incorrect number of complex parts read for 2D key '" +
                current_key + "'. Expected " + std::to_string(expected_parts) +
                ", Got " + std::to_string(flat_data_buffer.size()));
          }

          if (current_key == "librosa_cqt_complex_d") {
            cqt_data_d.assign(expected_rows,
                              std::vector<std::complex<double>>(expected_cols));
            for (size_t r = 0; r < expected_rows; ++r) {
              for (size_t c = 0; c < expected_cols; ++c) {
                size_t base_idx = (r * expected_cols + c) * 2;
                cqt_data_d[r][c] = {flat_data_buffer[base_idx],
                                    flat_data_buffer[base_idx + 1]};
              }
            }
          } else if (current_key == "librosa_cqt_complex_f") {
            // Note: generate_references.py doesn't create this key currently
            cqt_data_f.assign(expected_rows,
                              std::vector<std::complex<float>>(expected_cols));
            for (size_t r = 0; r < expected_rows; ++r) {
              for (size_t c = 0; c < expected_cols; ++c) {
                size_t base_idx = (r * expected_cols + c) * 2;
                cqt_data_f[r][c] = {
                    static_cast<float>(flat_data_buffer[base_idx]),
                    static_cast<float>(flat_data_buffer[base_idx + 1])};
              }
            }
          } else {
            std::cerr << "Warning: Skipping unknown 2D complex key '"
                      << current_key << "'" << std::endl;
          }
        } else if (is_complex_1d) {  // Process 1D Complex Vector (FFT)
          size_t num_complex_values = flat_data_buffer.size() / 2;
          if (flat_data_buffer.size() % 2 != 0) {
            throw std::runtime_error(
                "Odd number of values read for 1D complex key '" + current_key +
                "'. Expected pairs of real/imag.");
          }

          if (is_float) {  // complex64
            std::vector<std::complex<float>> complex_vec_f(num_complex_values);
            for (size_t i = 0; i < num_complex_values; ++i) {
              complex_vec_f[i] = {
                  static_cast<float>(flat_data_buffer[i * 2]),
                  static_cast<float>(flat_data_buffer[i * 2 + 1])};
            }
            data_cf[current_key] = std::move(complex_vec_f);
          } else {  // complex128
            std::vector<std::complex<double>> complex_vec_d(num_complex_values);
            for (size_t i = 0; i < num_complex_values; ++i) {
              complex_vec_d[i] = {flat_data_buffer[i * 2],
                                  flat_data_buffer[i * 2 + 1]};
            }
            data_cd[current_key] = std::move(complex_vec_d);
          }
        } else {           // Process 1D Real Vector (Windows, etc.)
          if (is_float) {  // float32
            std::vector<float> current_vec_f;
            current_vec_f.reserve(flat_data_buffer.size());
            for (double val : flat_data_buffer) {
              current_vec_f.push_back(static_cast<float>(val));
            }
            data_f[current_key] = std::move(current_vec_f);
          } else {                                   // float64 (double)
            data_d[current_key] = flat_data_buffer;  // Move vector content
          }
        }

        // Reset state for next block
        reading_data = false;
        is_complex_1d = false;
        is_complex_2d = false;
        is_float = false;
        current_key = "";

      } else if (reading_data && line.rfind("# SHAPE: ", 0) == 0) {
        if (!is_complex_2d) {  // SHAPE only expected for 2D complex data
                               // currently
          throw std::runtime_error(
              "# SHAPE marker found for non-complex-2d key: " + current_key);
        }
        std::stringstream ss(line.substr(9));
        if (!(ss >> expected_rows >> expected_cols) || expected_rows == 0 ||
            expected_cols == 0) {
          throw std::runtime_error("Failed to parse valid # SHAPE info: " +
                                   line);
        }
      } else if (reading_data) {  // Read numerical data value(s)
        std::string value_str_real = line;
        double real_part = 0.0;
        double imag_part = 0.0;

        // Remove potential comment part from real line
        size_t comment_pos = value_str_real.find('#');
        if (comment_pos != std::string::npos) {
          value_str_real = trim(value_str_real.substr(0, comment_pos));
        }
        // Trim trailing 'f' if present (we parse as double anyway)
        if (!value_str_real.empty() && value_str_real.back() == 'f') {
          value_str_real.pop_back();
        }

        try {
          real_part = std::stod(value_str_real);
        } catch (const std::exception& e) {
          throw std::runtime_error("Invalid number format for real part '" +
                                   line + "' for key '" + current_key +
                                   "': " + e.what());
        }

        if (is_complex_1d || is_complex_2d) {
          // Read the next line for the imaginary part
          std::string line_imag;
          if (!std::getline(infile, line_imag)) {
            throw std::runtime_error(
                "Unexpected end of file while reading imaginary part for key "
                "'" +
                current_key + "'.");
          }
          std::string value_str_imag = trim(line_imag);
          // Remove potential comment part from imag line
          comment_pos = value_str_imag.find('#');
          if (comment_pos != std::string::npos) {
            value_str_imag = trim(value_str_imag.substr(0, comment_pos));
          }
          // Trim trailing 'f' if present
          if (!value_str_imag.empty() && value_str_imag.back() == 'f') {
            value_str_imag.pop_back();
          }

          try {
            imag_part = std::stod(value_str_imag);
          } catch (const std::exception& e) {
            throw std::runtime_error("Invalid number format for imag part '" +
                                     line_imag + "' for key '" + current_key +
                                     "': " + e.what());
          }
          flat_data_buffer.push_back(real_part);
          flat_data_buffer.push_back(imag_part);
        } else {
          // Real data, just store the real part
          flat_data_buffer.push_back(real_part);
        }
      }
    }  // End while loop

    if (reading_data) {
      throw std::runtime_error(
          "Reached end of file while reading data for key: " + current_key);
    }
    load_successful = true;  // Mark loading as successful only if no exceptions

  } catch (const std::exception& e) {
    load_error_message =
        "Error loading reference data from '" + filename + "': " + e.what();
    load_successful = false;
    // Clear potentially partially loaded data to ensure consistent error state
    data_d.clear();
    data_f.clear();
    data_cd.clear();  // Clear new maps too
    data_cf.clear();
    cqt_data_d.clear();
    cqt_data_f.clear();
  }
  infile.close();
}

// Helper function to ensure data is loaded (thread-safe)
void ensureDataLoaded() {
  std::call_once(load_flag, loadDataFromFile);
  // After call_once, check if loading failed and throw if it did
  if (!load_successful) {
    throw std::runtime_error("Reference data loading failed. Reason: " +
                             (load_error_message.empty()
                                  ? "Unknown error during loading."
                                  : load_error_message));
  }
}

}  // Anonymous namespace

// --- Public Getter Implementations ---

const std::vector<double>& getExpectedDoubleVec(const std::string& key) {
  ensureDataLoaded();  // Ensure data is loaded (thread-safe)
  try {
    return data_d.at(key);  // .at() throws if key not found
  } catch (const std::out_of_range& oor) {
    throw std::runtime_error("Reference data key not found (double): '" + key +
                             "'");
  }
}

const std::vector<float>& getExpectedFloatVec(const std::string& key) {
  ensureDataLoaded();  // Ensure data is loaded (thread-safe)
  try {
    return data_f.at(key);  // .at() throws if key not found
  } catch (const std::out_of_range& oor) {
    throw std::runtime_error("Reference data key not found (float): '" + key +
                             "'");
  }
}

// --- NEW Getter Implementations for Complex Vectors ---

const std::vector<std::complex<double>>& getExpectedComplexDoubleVec(
    const std::string& key) {
  ensureDataLoaded();  // Ensure data is loaded (thread-safe)
  try {
    return data_cd.at(key);  // .at() throws if key not found
  } catch (const std::out_of_range& oor) {
    throw std::runtime_error(
        "Reference data key not found (complex double): '" + key + "'");
  }
}

const std::vector<std::complex<float>>& getExpectedComplexFloatVec(
    const std::string& key) {
  ensureDataLoaded();  // Ensure data is loaded (thread-safe)
  try {
    return data_cf.at(key);  // .at() throws if key not found
  } catch (const std::out_of_range& oor) {
    throw std::runtime_error("Reference data key not found (complex float): '" +
                             key + "'");
  }
}

// --- CQT Matrix Getters (Unchanged) ---

const std::vector<std::vector<std::complex<double>>>& getExpectedCQTD() {
  ensureDataLoaded();  // Ensure data is loaded (thread-safe)
  if (cqt_data_d.empty() &&
      load_successful) {  // Check load_successful to avoid hiding load errors
    // This check is mostly redundant if loadDataFromFile throws on error,
    // but provides a specific message if the CQT data wasn't found/parsed.
    throw std::runtime_error(
        "Complex double CQT reference data ('librosa_cqt_complex_d') not "
        "found or failed to load.");
  }
  return cqt_data_d;
}

const std::vector<std::vector<std::complex<float>>>& getExpectedCQTF() {
  ensureDataLoaded();  // Ensure data is loaded (thread-safe)
  if (cqt_data_f.empty() && load_successful) {
    // generate_references.py doesn't create this key currently
    throw std::runtime_error(
        "Complex float CQT reference data ('librosa_cqt_complex_f') not "
        "found or failed to load.");
  }
  return cqt_data_f;
}

}  // namespace TestDataUtils
