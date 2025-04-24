# cmake/compiler_settings.cmake
# ==============================
# Configures global compiler settings for the project. This includes setting the
# C++ standard, enabling export of compile commands, and adding
# compiler-specific definitions (like _USE_MATH_DEFINES for MSVC).
#
# Variables Set (Globally): - CMAKE_EXPORT_COMPILE_COMMANDS (BOOL): ON -
# CMAKE_CXX_STANDARD (STRING): 23 - CMAKE_CXX_STANDARD_REQUIRED (BOOL): ON -
# CMAKE_CXX_EXTENSIONS (BOOL): OFF - Adds _USE_MATH_DEFINES definition if MSVC
# is detected.
# ==============================

# Enable generation of compile_commands.json for IDEs/tools like clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
message(STATUS "Enabled export of compile commands (compile_commands.json).")

# Set C++ standard requirements
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Prefer standard features over compiler
                              # extensions
message(
  STATUS
    "Set CMAKE_CXX_STANDARD to ${CMAKE_CXX_STANDARD} (Required, No Extensions)."
)

# Add specific definitions required for certain compilers/platforms
if(MSVC)
  # Add math defines globally for MSVC compiler (e.g., M_PI)
  add_compile_definitions(_USE_MATH_DEFINES)
  message(STATUS "MSVC detected, adding _USE_MATH_DEFINES compile definition.")
endif()

message(STATUS "Loaded compiler settings from cmake/compiler_settings.cmake")
