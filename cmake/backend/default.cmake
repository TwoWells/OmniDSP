# ==============================================================================
# cmake/backend/default.cmake
# ==============================================================================
# Sets variables for the Default backend.

message(STATUS "  Configuring Default backend...")

# --- Set Backend Source Files ---
set(DEFAULT_BACKEND_SOURCES
    "${PROJECT_SOURCE_DIR}/src/omnidsp/default/backend.cpp"
    "${PROJECT_SOURCE_DIR}/src/omnidsp/default/fft.cpp"
    "${PROJECT_SOURCE_DIR}/src/omnidsp/default/cqt.cpp"
    "${PROJECT_SOURCE_DIR}/src/omnidsp/default/convolution.cpp"
    "${PROJECT_SOURCE_DIR}/src/omnidsp/default/resample.cpp"
    "${PROJECT_SOURCE_DIR}/src/omnidsp/default/filter.cpp"
    "${PROJECT_SOURCE_DIR}/src/omnidsp/default/window.cpp"
)

# --- Set Backend Include Directories ---
set(TEMP_DEFAULT_INCLUDES "")
if(Boost_FOUND AND TARGET Boost::headers)
    get_target_property(BOOST_INC Boost::headers INTERFACE_INCLUDE_DIRECTORIES)
    if(BOOST_INC)
        list(APPEND TEMP_DEFAULT_INCLUDES ${BOOST_INC})
    endif()
endif()
set(DEFAULT_BACKEND_INCLUDE_DIRS ${TEMP_DEFAULT_INCLUDES})

# --- Set Backend Link Libraries/Targets ---
set(TEMP_DEFAULT_LINK_LIBS "")
if(TARGET hwy::hwy)
    list(APPEND TEMP_DEFAULT_LINK_LIBS hwy::hwy)
endif()
set(DEFAULT_BACKEND_LINK_LIBS ${TEMP_DEFAULT_LINK_LIBS})

# --- Signal Success and Set Config Variables ---
set(OMNIDSP_HAS_DEFAULT TRUE)
set(OMNIDSP_HAS_BACKEND_DEFAULT_VALUE 1)
message(STATUS "      Default backend configured.")
