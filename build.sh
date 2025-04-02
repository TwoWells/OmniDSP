#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# EDIT THIS PATH if needed, used only if ONEAPI_ROOT is not set (fallback to find setvars)
DEFAULT_ONEAPI_SETVARS_PATH="/opt/intel/oneapi/setvars.sh"
# --- End Configuration ---

# --- Default Build Options ---
BUILD_TYPE="Release"
BACKEND_PREF="auto" # Options: auto, mkl, accelerate
RUN_TESTS="ON"      # Options: ON, OFF
INSTALL_PREFIX=""   # e.g., "../install"
CMAKE_GENERATOR=""  # e.g., "Ninja"

# --- Helper Functions ---
print_usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -t, --type <TYPE>      Build type: Release (default), Debug"
  echo "  -b, --backend <PREF>   Backend preference: auto (default), mkl, accelerate"
  echo "      --no-tests           Disable running tests (default: ON)"
  echo "  -p, --prefix <PATH>    Install prefix path (default: none)"
  echo "  -G <GENERATOR>       Specify CMake generator (e.g., \"Ninja\")"
  echo "  -h, --help             Show this help message"
}

# --- Argument Parsing ---
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -t|--type)
      BUILD_TYPE="$2"
      shift # past argument
      shift # past value
      ;;
    -b|--backend)
      BACKEND_PREF="$2"
      shift # past argument
      shift # past value
      ;;
    --no-tests)
      RUN_TESTS="OFF"
      shift # past argument
      ;;
    -p|--prefix)
      INSTALL_PREFIX="$2"
      shift # past argument
      shift # past value
      ;;
    -G)
      CMAKE_GENERATOR="$2"
      shift # past argument
      shift # past value
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)    # unknown option
      echo "Unknown option: $1"
      print_usage
      exit 1
      ;;
  esac
done

# Validate Build Type
if [[ "$BUILD_TYPE" != "Release" && "$BUILD_TYPE" != "Debug" ]]; then
  echo "Error: Invalid build type '$BUILD_TYPE'. Use Release or Debug."
  exit 1
fi

# Validate Backend Preference
if [[ "$BACKEND_PREF" != "auto" && "$BACKEND_PREF" != "mkl" && "$BACKEND_PREF" != "accelerate" ]]; then
  echo "Error: Invalid backend preference '$BACKEND_PREF'. Use auto, mkl, or accelerate."
  exit 1
fi

echo "--- Build Configuration ---"
echo "Build Type:       $BUILD_TYPE"
echo "Backend Pref:     $BACKEND_PREF"
echo "Run Tests:        $RUN_TESTS"
if [[ -n "$INSTALL_PREFIX" ]]; then
  echo "Install Prefix:   $INSTALL_PREFIX"
fi
if [[ -n "$CMAKE_GENERATOR" ]]; then
  echo "CMake Generator:  $CMAKE_GENERATOR"
fi
echo "-------------------------"

# --- Source Intel oneAPI environment if MKL is preferred ---
if [[ "$BACKEND_PREF" == "mkl" ]]; then
  # Check if SETVARS_COMPLETED is already set
  if [[ -n "${SETVARS_COMPLETED}" ]]; then
      echo "SETVARS_COMPLETED environment variable found ('${SETVARS_COMPLETED}'). Assuming environment is configured."
  else
      echo "SETVARS_COMPLETED not set. Attempting to source setvars.sh..."
      # Try finding setvars.sh via ONEAPI_ROOT first, then default path
      SETVARS_PATH=""
      if [[ -n "${ONEAPI_ROOT}" && -f "${ONEAPI_ROOT}/setvars.sh" ]]; then
          SETVARS_PATH="${ONEAPI_ROOT}/setvars.sh"
      elif [[ -f "$DEFAULT_ONEAPI_SETVARS_PATH" ]]; then
           SETVARS_PATH="$DEFAULT_ONEAPI_SETVARS_PATH"
      fi

      if [[ -n "$SETVARS_PATH" ]]; then
        echo "Sourcing Intel oneAPI environment script: $SETVARS_PATH"
        source "$SETVARS_PATH" intel64 || { echo "Error sourcing setvars.sh"; exit 1; }
        # Verify if SETVARS_COMPLETED got set after sourcing
        if [[ -z "${SETVARS_COMPLETED}" ]]; then
            echo "Warning: Sourced setvars.sh but SETVARS_COMPLETED is still not set."
        else
             echo "Environment configured by setvars.sh (SETVARS_COMPLETED=${SETVARS_COMPLETED})."
        fi
      else
        echo "Warning: MKL backend preferred, SETVARS_COMPLETED not set, and could not find setvars.sh via ONEAPI_ROOT or default path."
        echo "         Build might fail if MKL is not found via other means."
        # exit 1 # Optional: Exit if setvars cannot be found/run
      fi
  fi
fi

# --- Clean and Create Build Directory ---
echo "Cleaning and creating build directory..."
if [[ -d "build" ]]; then
  rm -rf "build"
fi
mkdir "build"
cd "build"

# --- Construct CMake Command ---
CMAKE_ARGS=("../") # Point to parent directory where CMakeLists.txt is

CMAKE_ARGS+=("-DCMAKE_BUILD_TYPE=$BUILD_TYPE")

if [[ "$BACKEND_PREF" == "mkl" ]]; then
  CMAKE_ARGS+=("-DFFT_PREFER_ONEMKL=ON")
  if [[ "$(uname)" == "Darwin" ]]; then # On Apple, explicitly disable Accelerate if MKL is preferred
      CMAKE_ARGS+=("-DFFT_PREFER_ACCELERATE=OFF")
  fi
elif [[ "$BACKEND_PREF" == "accelerate" ]]; then
   CMAKE_ARGS+=("-DFFT_PREFER_ONEMKL=OFF")
   CMAKE_ARGS+=("-DFFT_PREFER_ACCELERATE=ON") # Explicitly request
fi

if [[ -n "$INSTALL_PREFIX" ]]; then
  ABS_INSTALL_PREFIX=$(mkdir -p "$../$INSTALL_PREFIX" && realpath --relative-to=. "$../$INSTALL_PREFIX")
  CMAKE_ARGS+=("-DCMAKE_INSTALL_PREFIX=$ABS_INSTALL_PREFIX")
fi

if [[ -n "$CMAKE_GENERATOR" ]]; then
  CMAKE_ARGS+=("-G" "$CMAKE_GENERATOR")
fi

# --- Run CMake Configuration ---
echo "Running CMake configure..."
echo "cmake ${CMAKE_ARGS[@]}" # Print the command
cmake "${CMAKE_ARGS[@]}"

# --- Run CMake Build ---
echo "Running CMake build..."
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)
cmake --build . --config "$BUILD_TYPE" --parallel $NPROC # Use available cores

# --- Run Tests ---
if [[ "$RUN_TESTS" == "ON" ]]; then
  echo "Running tests..."
  ctest -C "$BUILD_TYPE" --output-on-failure
else
  echo "Skipping tests."
fi

# --- Return to original directory ---
cd ..

echo "Build script finished successfully."
exit 0