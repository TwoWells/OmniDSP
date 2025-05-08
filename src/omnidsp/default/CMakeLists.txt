# OmniDSP/src/omnidsp/default/CMakeLists.txt
# Configures the Default backend for the 'OmniDSP' library.

message(STATUS "  Configuring Default backend...")

# --- Define Default Backend Source Files ---
# List all .cpp files for the default backend implementation.
set(DEFAULT_BACKEND_SOURCES_INTERNAL # Use a distinct variable name
    "${CMAKE_CURRENT_SOURCE_DIR}/backend.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fft.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/cqt.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/convolution.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/resample.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/filter.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/window.cpp"
)

# --- Add Sources to the 'OmniDSP' library target ---
# The 'OmniDSP' target is defined in the parent CMakeLists.txt (src/omnidsp/CMakeLists.txt).
target_sources(OmniDSP PRIVATE ${DEFAULT_BACKEND_SOURCES_INTERNAL})

# --- Include Directories for Default Backend ---
# Add current directory for backend-specific headers like "backend.hpp".
target_include_directories(OmniDSP PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

# --- Dependencies for Default Backend ---
# Example: Boost (headers only for some parts) and Highway (already linked generally)
# The original default.cmake checked for Boost_FOUND and TARGET Boost::headers.
# Assuming Boost::headers is found by cmake/depend.cmake if needed.
if(TARGET Boost::headers)
    # If default backend sources directly include Boost headers in a way that
    # requires Boost::headers to be a usage requirement of this backend part.
    target_link_libraries(OmniDSP PRIVATE Boost::headers)
    message(STATUS "    Default backend: Added Boost::headers dependency to OmniDSP target.")
endif()

# Highway (hwy::hwy) is assumed to be linked by the parent src/omnidsp/CMakeLists.txt
# as a common dependency. If it's *only* for default, link it here.
# if(TARGET hwy::hwy)
# target_link_libraries(OmniDSP PRIVATE hwy::hwy)
# endif()


# --- Signal Success and Set Configuration Variable ---
# This variable is used by src/omnidsp/CMakeLists.txt for omnidsp_config.h
# and for the final list of enabled backends.
set(OMNIDSP_BACKEND_DEFAULT_ENABLED 1 PARENT_SCOPE)

# Add a compile definition specific to this backend's implementation details if necessary
# target_compile_definitions(OmniDSP PRIVATE OMNIDSP_USING_DEFAULT_IMPL_DETAILS)

message(STATUS "    Default backend configured successfully.")
