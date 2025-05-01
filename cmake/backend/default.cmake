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
    # "${PROJECT_SOURCE_DIR}/src/omnidsp/default/filter_design.cpp" # REMOVED - File does not exist
)

# --- Set Backend Include Directories ---
set(TEMP_DEFAULT_INCLUDES "")
if(Boost_FOUND AND TARGET Boost::headers) # Use Boost::headers target
    get_target_property(BOOST_INC Boost::headers INTERFACE_INCLUDE_DIRECTORIES)
    if(BOOST_INC)
        list(APPEND TEMP_DEFAULT_INCLUDES ${BOOST_INC})
    # else() # Optional: Fallback to Boost_INCLUDE_DIRS if property fails
    #    if(DEFINED Boost_INCLUDE_DIRS AND Boost_INCLUDE_DIRS)
    #        list(APPEND TEMP_DEFAULT_INCLUDES ${Boost_INCLUDE_DIRS})
    #    endif()
    endif()
endif()
set(DEFAULT_BACKEND_INCLUDE_DIRS ${TEMP_DEFAULT_INCLUDES}) # PARENT_SCOPE removed

# --- Set Backend Link Libraries/Targets ---
set(TEMP_DEFAULT_LINK_LIBS "")
# Add hwy::hwy as it's used by the default backend implementation files
if(TARGET hwy::hwy)
    list(APPEND TEMP_DEFAULT_LINK_LIBS hwy::hwy)
endif()
# Add Boost::headers if needed by implementation details (or rely on main target link)
# list(APPEND TEMP_DEFAULT_LINK_LIBS Boost::headers)
set(DEFAULT_BACKEND_LINK_LIBS ${TEMP_DEFAULT_LINK_LIBS}) # PARENT_SCOPE removed

# --- Signal Success ---
set(OMNIDSP_HAS_DEFAULT TRUE) # PARENT_SCOPE removed
message(STATUS "      Default backend configured.")
