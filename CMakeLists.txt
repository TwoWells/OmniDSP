cmake_minimum_required(VERSION 3.15)
project(OmniFFT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Option to prefer a backend ---
option(FFT_PREFER_ACCELERATE "Prefer Apple Accelerate if available" OFF)
option(FFT_PREFER_ONEMKL "Prefer Intel oneMKL if available" OFF)

# --- Backend Detection and Source File Selection ---
set(FFT_BACKEND "NONE")
set(FFT_IMPL_SOURCE "") # Variable to hold the implementation file path

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND NOT FFT_PREFER_ONEMKL)
    find_library(ACCELERATE_LIBRARY Accelerate)
    if(ACCELERATE_LIBRARY)
        message(STATUS "Found Apple Accelerate framework.")
        set(FFT_BACKEND "ACCELERATE")
        set(FFT_IMPL_SOURCE src/fft_impl_accelerate.cpp) # Select Accelerate impl
    else()
         message(WARNING "Apple Accelerate framework not found on Darwin system.")
    endif()
    unset(ACCELERATE_LIBRARY CACHE)
endif()

if(NOT FFT_BACKEND STREQUAL "ACCELERATE" OR FFT_PREFER_ONEMKL)
    find_package(MKL)
    if(MKL_FOUND)
        message(STATUS "Found Intel oneMKL.")
        set(FFT_BACKEND "ONEMKL")
        set(FFT_IMPL_SOURCE src/fft_impl_onemkl.cpp) # Select oneMKL impl
    else()
        message(STATUS "Intel oneMKL not found.")
        if(FFT_PREFER_ONEMKL)
             message(WARNING "Preferred oneMKL backend not found.")
        endif()
    endif()
endif()

# --- Select Stub Implementation if no backend found ---
if(FFT_BACKEND STREQUAL "NONE")
     message(WARNING "No suitable FFT backend (Accelerate or oneMKL) found or selected. Using stub implementation.")
     set(FFT_IMPL_SOURCE src/fft_impl_stub.cpp) # Select Stub impl
endif()

# --- Define the library target using the selected implementation ---
# Check if the selected implementation source file exists
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${FFT_IMPL_SOURCE}")
    message(FATAL_ERROR "Selected FFT implementation source file not found: ${FFT_IMPL_SOURCE}")
endif()

add_library(omnifft
    src/omnifft.cpp      # Platform-independent wrapper
    ${FFT_IMPL_SOURCE}   # The selected platform-specific implementation
)
target_include_directories(omnifft PUBLIC include) # Make public header available

# --- Apply backend-specific settings to the library target ---
if(FFT_BACKEND STREQUAL "ACCELERATE")
    target_link_libraries(omnifft PUBLIC Accelerate::Accelerate)
    target_compile_definitions(omnifft PRIVATE USE_ACCELERATE)
elseif(FFT_BACKEND STREQUAL "ONEMKL")
    target_include_directories(omnifft PRIVATE ${MKL_INCLUDE_DIRS})
    if(TARGET MKL::MKL)
         target_link_libraries(omnifft PUBLIC MKL::MKL)
    elseif(MKL_LIBRARIES)
         target_link_libraries(omnifft PUBLIC ${MKL_LIBRARIES})
    else()
         message(WARNING "FindMKL found MKL but did not define MKL::MKL target or MKL_LIBRARIES. Manual linking might be required.")
    endif()
    target_compile_definitions(omnifft PRIVATE USE_ONEMKL)
# elseif(FFT_BACKEND STREQUAL "NONE")
    # No specific settings needed for the stub backend
endif()


# --- Example Executable (Optional) ---
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/main.cpp")
    add_executable(omnifft_example examples/main.cpp)
    target_link_libraries(omnifft_example PRIVATE omnifft)
endif()

# --- Installation (Optional) ---
# install(TARGETS omnifft DESTINATION lib)
# install(FILES include/omnifft.h DESTINATION include)